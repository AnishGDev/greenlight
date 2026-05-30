"""
engine.py — Person A, STEPS 1-2: Open Targets + Europe PMC -> EvidencePackage.

What this gives you:
  - resolve a gene symbol -> Ensembl id, and a disease name -> EFO id
  - pull typed target-disease evidence from Open Targets
  - for every literature-linked row, fetch the abstract from Europe PMC
    and extract discrete claims (verbatim source spans) with an LLM
  - apply the cutoff filter (NO FUTURE LEAK) — dates come from Europe PMC's
    firstPublicationDate where available, else the OT publicationYear
  - return a schema-valid EvidencePackage

Run offline first:   python engine.py --mock
Run live (your box):  python engine.py --target PCSK9 --disease hypercholesterolemia

NOTE: Open Targets' GraphQL schema evolves. Validate the query field names
against the live browser before the live run:
  https://api.platform.opentargets.org/api/v4/graphql/browser
The mock path lets the rest of your pipeline work regardless.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import date
from typing import Optional

try:
    from tenacity import (
        retry,
        retry_if_exception,
        stop_after_attempt,
        wait_exponential,
    )
except ModuleNotFoundError:
    def retry(*_args, **_kwargs):
        def decorator(fn):
            return fn
        return decorator

    def retry_if_exception(predicate):
        return predicate

    def stop_after_attempt(_attempts: int):
        return None

    def wait_exponential(**_kwargs):
        return None

from schemas import (
    EvidenceItem, EvidencePackage, EvidenceType, SourceDB, make_pair_id,
)

OT_URL = "https://api.platform.opentargets.org/api/v4/graphql"
EPMC_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
LLM_MODEL_DEFAULT = "gpt-4o-mini"   # cheap & fast; override via GREENLIGHT_LLM_MODEL
LLM_MAX_CLAIMS = 6                  # cap claims/abstract to bound cost + noise
ENRICH_BUDGET_DEFAULT = 25          # max literature-enriched rows per pair (Modal sweep cost cap)
MAX_ITEMS_DEFAULT = 100             # cap evidence volume handed to B/C

# Open Targets datatypeId -> our EvidenceType.
# Source: live datatypeId values returned by the OT v4 evidences query.
OT_TYPE_MAP: dict[str, EvidenceType] = {
    "genetic_association": EvidenceType.genetic,
    "genetic_literature": EvidenceType.genetic,
    "somatic_mutation": EvidenceType.genetic,
    "known_drug": EvidenceType.clinical,
    "clinical": EvidenceType.clinical,
    "affected_pathway": EvidenceType.pathway,
    "animal_model": EvidenceType.in_vivo,
    "rna_expression": EvidenceType.expression,
    "literature": EvidenceType.correlational,
}


# --------------------------------------------------------------------------- #
# Live GraphQL (only used when --mock is NOT set)
# --------------------------------------------------------------------------- #
class TransientAPIError(RuntimeError):
    """Raised for API failures that should be retried."""


def _is_retryable_exception(exc: BaseException) -> bool:
    if isinstance(exc, TransientAPIError):
        return True
    name = type(exc).__name__
    if name in {"TimeoutException", "ConnectError", "NetworkError", "ReadError"}:
        return True
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    return status_code in {408, 429} or (status_code is not None and status_code >= 500)


def _retryable_api_call(fn):
    return retry(
        retry=retry_if_exception(_is_retryable_exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )(fn)


def _summarize_errors(errors: object) -> str:
    text = json.dumps(errors, default=str)
    return text[:500]


@_retryable_api_call
def _gql(query: str, variables: dict) -> dict:
    import httpx  # imported lazily so --mock needs no network/deps
    r = httpx.post(OT_URL, json={"query": query, "variables": variables}, timeout=30)
    r.raise_for_status()
    payload = r.json()
    if "errors" in payload:
        summary = _summarize_errors(payload["errors"])
        if "Internal server error" in summary:
            raise TransientAPIError(summary)
        raise RuntimeError(summary)
    return payload["data"]


_RESOLVE = """
query Resolve($q: String!) {
  search(queryString: $q, entityNames: ["target","disease"]) {
    hits { id name entity }
  }
}
"""

# Validated against the live OT v4 GraphQL browser (2026-05).
# enableIndirect rolls up evidence from descendant diseases in the EFO ontology.
_EVIDENCE = """
query Evidence($efoId: String!, $ensemblId: String!) {
  disease(efoId: $efoId) {
    evidences(ensemblIds: [$ensemblId], enableIndirect: true, size: 200) {
      rows {
        id
        score
        datatypeId
        datasourceId
        publicationYear
        literature
        diseaseFromSource
      }
    }
  }
}
"""


# For diseases, our schemas use EFO/MONDO ids. HP/HPO terms describe
# phenotypic features; using them as canonical disease IDs breaks pair_id joins.
_DISEASE_PREFIX_RANK = {"EFO_": 0, "MONDO_": 1}


def _disease_rank(hit: dict) -> int:
    for p, r in _DISEASE_PREFIX_RANK.items():
        if hit["id"].startswith(p):
            return r
    return 99


def _is_canonical_disease_hit(hit: dict) -> bool:
    return any(hit["id"].startswith(p) for p in _DISEASE_PREFIX_RANK)


def _pick_hit(hits: list[dict], entity: str, query: str) -> str:
    """Pick the best OT search hit: exact name match, then prefix preference, then top hit."""
    typed = [h for h in hits if h.get("entity") == entity]
    if not typed:
        raise LookupError(f"Open Targets returned no {entity} hits for {query!r}")
    if entity == "disease":
        canonical = [h for h in typed if _is_canonical_disease_hit(h)]
        if not canonical:
            available = ", ".join(f"{h.get('id')}:{h.get('name')}" for h in typed[:8])
            raise LookupError(
                f"Open Targets returned no EFO/MONDO disease hits for {query!r}; "
                f"available disease hits: {available}"
            )
        typed = canonical
    q = query.strip().lower()
    exact = [h for h in typed if (h.get("name") or "").strip().lower() == q]
    pool = exact or typed
    if entity == "disease":
        pool = sorted(pool, key=_disease_rank)
    return pool[0]["id"]


def resolve_ids(target: str, disease: str) -> tuple[str, str]:
    ens = _pick_hit(_gql(_RESOLVE, {"q": target})["search"]["hits"], "target", target)
    efo = _pick_hit(_gql(_RESOLVE, {"q": disease})["search"]["hits"], "disease", disease)
    return ens, efo


def fetch_ot_rows(ensembl_id: str, efo_id: str) -> list[dict]:
    data = _gql(_EVIDENCE, {"efoId": efo_id, "ensemblId": ensembl_id})
    return data["disease"]["evidences"]["rows"]


# --------------------------------------------------------------------------- #
# Europe PMC — date-stamped literature; no API key required
# --------------------------------------------------------------------------- #
_HTML_TAG = re.compile(r"<[^>]+>")


def _strip_html(text: Optional[str]) -> str:
    return _HTML_TAG.sub(" ", text or "").strip()


def _parse_epmc_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        return None


@_retryable_api_call
def europepmc_by_pmid(pmid: str) -> Optional[dict]:
    """Fetch one PubMed record from Europe PMC. Returns abstract + first_pub_date."""
    import httpx
    params = {"query": f"EXT_ID:{pmid} AND SRC:MED", "format": "json", "resultType": "core"}
    r = httpx.get(EPMC_URL, params=params, timeout=20)
    r.raise_for_status()
    hits = r.json().get("resultList", {}).get("result", [])
    if not hits:
        return None
    h = hits[0]
    return {
        "pmid": h.get("pmid") or pmid,
        "title": h.get("title", ""),
        "abstract": _strip_html(h.get("abstractText")),
        "first_pub_date": _parse_epmc_date(h.get("firstPublicationDate")),
        "doi": h.get("doi"),
        "url": f"https://europepmc.org/abstract/MED/{pmid}",
    }


@_retryable_api_call
def europepmc_search(query: str, cutoff_date: date, page_size: int = 25) -> list[dict]:
    """Keyword search Europe PMC, hard-bounded to cutoff_date. Step 7 uses this."""
    import httpx
    bounded = f"({query}) AND FIRST_PDATE:[* TO {cutoff_date.isoformat()}]"
    params = {"query": bounded, "format": "json", "resultType": "core", "pageSize": page_size}
    r = httpx.get(EPMC_URL, params=params, timeout=30)
    r.raise_for_status()
    out = []
    for h in r.json().get("resultList", {}).get("result", []):
        pmid = h.get("pmid")
        if not pmid:
            continue
        out.append({
            "pmid": pmid,
            "title": h.get("title", ""),
            "abstract": _strip_html(h.get("abstractText")),
            "first_pub_date": _parse_epmc_date(h.get("firstPublicationDate")),
            "doi": h.get("doi"),
            "url": f"https://europepmc.org/abstract/MED/{pmid}",
        })
    return out


# --------------------------------------------------------------------------- #
# OpenAI claim extraction — verbatim source_text spans for B's NLI
# --------------------------------------------------------------------------- #
_EXTRACT_SYSTEM = (
    "You are a biomedical claim extractor. Given a paper abstract about a "
    "TARGET gene/protein and a DISEASE, extract discrete factual claims that "
    "speak to the target-disease relationship (genetic association, mechanism, "
    "clinical outcome, safety, contradictory findings). For each claim, provide "
    "a source_text span copied VERBATIM from the abstract that justifies it — "
    "no paraphrasing in source_text. Return JSON only with the shape: "
    '{"claims": [{"claim": "...", "source_text": "..."}]}. '
    f"Return at most {LLM_MAX_CLAIMS} claims, most decision-relevant first. "
    'If the abstract is off-topic return {"claims": []}.'
)

_EXTRACT_USER_TMPL = "Target: {target}\nDisease: {disease}\n\nAbstract:\n{abstract}"


def extract_claims_llm(abstract: str, target: str, disease: str,
                       model: Optional[str] = None) -> list[dict]:
    """Call the LLM with JSON mode. Returns [] on any failure (cost-bounded, fail-soft)."""
    if not abstract.strip():
        return []
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()
    model = model or os.getenv("GREENLIGHT_LLM_MODEL", LLM_MODEL_DEFAULT)
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        temperature=0,
        messages=[
            {"role": "system", "content": _EXTRACT_SYSTEM},
            {"role": "user", "content": _EXTRACT_USER_TMPL.format(
                target=target, disease=disease, abstract=abstract[:6000])},
        ],
    )
    try:
        data = json.loads(resp.choices[0].message.content or "{}")
    except json.JSONDecodeError:
        return []
    claims = data.get("claims") or []
    # Keep only well-formed entries with non-empty verbatim spans actually present in the abstract.
    cleaned = []
    for c in claims[:LLM_MAX_CLAIMS]:
        claim = (c.get("claim") or "").strip()
        span = (c.get("source_text") or "").strip()
        if claim and span and span[:80] in abstract:
            cleaned.append({"claim": claim, "source_text": span})
    return cleaned


# --------------------------------------------------------------------------- #
# Row -> EvidenceItem, with the cutoff gate
# --------------------------------------------------------------------------- #
def row_to_item(row: dict, target: str, disease: str) -> EvidenceItem | None:
    etype = OT_TYPE_MAP.get(row.get("datatypeId", ""), EvidenceType.correlational)
    year = row.get("publicationYear")
    # Conservative for year-only metadata: a cutoff of 2020-01-01 must not admit
    # evidence that could have appeared later in 2020.
    pub = date(year, 12, 31) if year else None
    pmids = row.get("literature") or []
    src_id = f"OT:{row.get('id','?')}"
    if pmids:
        src_id = f"PMID:{pmids[0]}"
    claim = (f"{row.get('datasourceId','source')} evidence links {target} to "
             f"{disease} (score={row.get('score'):.2f}).") if row.get("score") is not None \
        else f"{row.get('datasourceId','source')} evidence links {target} to {disease}."
    return EvidenceItem(
        claim=claim,
        evidence_type=etype,
        source_db=SourceDB.open_targets,
        source_id=src_id,
        source_text=row.get("diseaseFromSource") or claim,  # placeholder text; real text added in Step 2
        publication_date=pub,
        retrieval_score=row.get("score"),
        metadata={"datasourceId": row.get("datasourceId"), "pmids": pmids},
    )


def apply_cutoff(items: list[EvidenceItem], cutoff: date,
                 strict: bool = True) -> list[EvidenceItem]:
    """NO FUTURE LEAK. strict=True drops undated items from the retrospective run."""
    kept = []
    for it in items:
        if it.publication_date is None:
            if not strict:
                kept.append(it)
            continue
        if it.publication_date <= cutoff:
            kept.append(it)
    return kept


# --------------------------------------------------------------------------- #
# Step 2: enrich OT rows with Europe PMC abstracts + LLM-extracted claims
# --------------------------------------------------------------------------- #
def _pick_pre_cutoff_paper(pmids: list[str], cutoff: date,
                           errors: list[str]) -> tuple[Optional[dict], int]:
    """Walk a row's PMIDs and return the first paper whose first_pub_date <= cutoff.
    Returns (paper_or_None, papers_fetched). Cheap step that gates the LLM call.
    """
    fetched = 0
    for pmid in pmids:
        if not pmid:
            continue
        try:
            paper = europepmc_by_pmid(pmid)
        except Exception as e:
            errors.append(f"epmc:{pmid}:{type(e).__name__}")
            continue
        fetched += 1
        if not paper or not paper["abstract"]:
            continue
        pub = paper["first_pub_date"]
        if pub is None or pub > cutoff:
            continue
        return paper, fetched
    return None, fetched


def _literature_items_for_row(row: dict, target: str, disease: str,
                              cutoff: date,
                              errors: list[str]) -> tuple[list[EvidenceItem], int]:
    """For an OT row, find the first pre-cutoff linked paper, extract claims from it.
    Returns ([items], papers_processed). Returns ([], n) when no usable paper was
    found or the LLM yielded nothing — caller falls back to OT-generic row item.
    """
    pmids = row.get("literature") or []
    if not pmids:
        return [], 0
    paper, fetched = _pick_pre_cutoff_paper(pmids, cutoff, errors)
    if not paper:
        return [], fetched
    try:
        claims = extract_claims_llm(paper["abstract"], target, disease)
    except Exception as e:
        errors.append(f"llm:{paper['pmid']}:{type(e).__name__}")
        return [], fetched
    if not claims:
        return [], fetched
    etype = OT_TYPE_MAP.get(row.get("datatypeId", ""), EvidenceType.correlational)
    items = [
        EvidenceItem(
            claim=c["claim"],
            evidence_type=etype,
            source_db=SourceDB.europe_pmc,
            source_id=f"PMID:{paper['pmid']}",
            source_text=c["source_text"][:2000],
            source_url=paper["url"],
            publication_date=paper["first_pub_date"],
            retrieval_score=row.get("score"),
            metadata={
                "datasourceId": row.get("datasourceId"),
                "datatypeId": row.get("datatypeId"),
                "ot_id": row.get("id"),
                "title": paper.get("title"),
                "doi": paper.get("doi"),
            },
        )
        for c in claims
    ]
    return items, fetched


# --------------------------------------------------------------------------- #
# The worker: (target, disease, cutoff) -> EvidencePackage   (NO label in/out)
# --------------------------------------------------------------------------- #
def _prefilter_rows(rows: list[dict], cutoff: date) -> list[dict]:
    """Drop OT rows whose publicationYear is strictly after the cutoff year.
    Undated rows pass through — Europe PMC may date them inside the cutoff.
    """
    out = []
    for r in rows:
        y = r.get("publicationYear")
        if y is None or date(y, 12, 31) <= cutoff:
            out.append(r)
    return out


def _validate_canonical_disease_id(disease_id: str) -> None:
    if not any(disease_id.startswith(p) for p in _DISEASE_PREFIX_RANK):
        raise ValueError(f"disease_id must be EFO/MONDO, got {disease_id!r}")


def _empty_error_package(target: str, disease: str, cutoff_date: date,
                         errors: list[str], t0: float,
                         target_id: Optional[str] = None,
                         disease_id: Optional[str] = None,
                         sources_queried: int = 0) -> EvidencePackage:
    resolved_target = target_id or target
    resolved_disease = disease_id or disease
    return EvidencePackage(
        pair_id=make_pair_id(resolved_target, resolved_disease),
        target_symbol=target,
        target_id=resolved_target,
        disease_name=disease,
        disease_id=resolved_disease,
        cutoff_date=cutoff_date,
        evidence_items=[],
        sources_queried=sources_queried,
        papers_processed=0,
        runtime_seconds=round(time.time() - t0, 3),
        errors=errors,
    )


def _sort_items_for_handoff(items: list[EvidenceItem]) -> list[EvidenceItem]:
    return sorted(
        items,
        key=lambda it: (
            it.retrieval_score if it.retrieval_score is not None else -1.0,
            it.publication_date or date.min,
            it.source_id,
        ),
        reverse=True,
    )


def build_evidence_package(target: str, disease: str, cutoff_date: date,
                           mock: bool = False,
                           enrich_budget: int = ENRICH_BUDGET_DEFAULT,
                           max_items: Optional[int] = MAX_ITEMS_DEFAULT,
                           target_id: Optional[str] = None,
                           disease_id: Optional[str] = None) -> EvidencePackage:
    t0 = time.time()
    errors: list[str] = []
    if mock:
        ensembl_id, efo_id = "ENSG00000169174", "EFO_0004911"
        rows = _MOCK_ROWS
    else:
        try:
            if target_id or disease_id:
                if not (target_id and disease_id):
                    raise ValueError("target_id and disease_id must be provided together")
                _validate_canonical_disease_id(disease_id)
                ensembl_id, efo_id = target_id, disease_id
            else:
                ensembl_id, efo_id = resolve_ids(target, disease)
            rows = fetch_ot_rows(ensembl_id, efo_id)
        except Exception as e:
            errors.append(f"resolve_or_ot:{type(e).__name__}:{str(e)[:300]}")
            return _empty_error_package(
                target, disease, cutoff_date, errors, t0,
                target_id=target_id, disease_id=disease_id, sources_queried=1,
            )

    candidate_rows = rows if mock else _prefilter_rows(rows, cutoff_date)
    # Prioritise rows with PMIDs and the highest OT score within the enrichment budget.
    candidate_rows_sorted = sorted(
        candidate_rows,
        key=lambda r: (bool(r.get("literature")), r.get("score") or 0.0),
        reverse=True,
    )

    raw_items: list[EvidenceItem] = []
    papers_processed = 0
    attempts = 0   # bounds total EPMC + LLM cost regardless of success rate
    for r in candidate_rows_sorted:
        attempted_enrichment = (not mock
                                and attempts < enrich_budget
                                and bool(r.get("literature")))
        if attempted_enrichment:
            attempts += 1
            lit_items, np = _literature_items_for_row(
                r, target, disease, cutoff_date, errors)
            papers_processed += np
            if lit_items:
                raw_items.extend(lit_items)
                continue
        # fallback: no PMID, all PMIDs post-cutoff, LLM yielded nothing, or budget spent
        item = row_to_item(r, target, disease)
        if item:
            raw_items.append(item)

    items = _sort_items_for_handoff(apply_cutoff(raw_items, cutoff_date, strict=True))
    if max_items is not None and max_items > 0 and len(items) > max_items:
        errors.append(f"truncated:{len(items)}->{max_items}")
        items = items[:max_items]

    return EvidencePackage(
        pair_id=make_pair_id(ensembl_id, efo_id),
        target_symbol=target, target_id=ensembl_id,
        disease_name=disease, disease_id=efo_id,
        cutoff_date=cutoff_date, evidence_items=items,
        sources_queried=2 if not mock else 1,    # OT + Europe PMC
        papers_processed=papers_processed,
        runtime_seconds=round(time.time() - t0, 3),
        errors=errors,
    )


def search(query: str, cutoff_date: date, limit: int = 10) -> list[EvidenceItem]:
    """Thin wrapper Person B calls for the contradiction hunt (Step 7).
    Returns date-filtered Europe PMC hits as EvidenceItems with abstract as source_text.
    NO FUTURE LEAK: belt-and-suspenders — drops any paper whose
    first_pub_date is missing or > cutoff_date, even if the server returned it.
    """
    items: list[EvidenceItem] = []
    for paper in europepmc_search(query, cutoff_date, page_size=limit):
        if not paper["abstract"]:
            continue
        pub = paper["first_pub_date"]
        if pub is None or pub > cutoff_date:
            continue
        items.append(EvidenceItem(
            claim=paper["title"] or query,
            evidence_type=EvidenceType.correlational,
            source_db=SourceDB.europe_pmc,
            source_id=f"PMID:{paper['pmid']}",
            source_text=paper["abstract"][:2000],
            source_url=paper["url"],
            publication_date=pub,
            metadata={"title": paper.get("title"), "doi": paper.get("doi")},
        ))
    return items


# --------------------------------------------------------------------------- #
# Mock OT rows (lets the pipeline run with no network) — illustrative
# --------------------------------------------------------------------------- #
_MOCK_ROWS = [
    {"id": "e1", "score": 0.92, "datatypeId": "genetic_association",
     "datasourceId": "ot_genetics_portal", "publicationYear": 2006,
     "literature": ["16554528"], "diseaseFromSource": "hypercholesterolemia"},
    {"id": "e2", "score": 0.80, "datatypeId": "affected_pathway",
     "datasourceId": "reactome", "publicationYear": 2007,
     "literature": ["17341576"], "diseaseFromSource": "lipid metabolism"},
    {"id": "e3", "score": 0.70, "datatypeId": "literature",
     "datasourceId": "europepmc", "publicationYear": 2008,
     "literature": ["18438315"], "diseaseFromSource": "hypercholesterolemia"},
    # post-cutoff row -> MUST be dropped by apply_cutoff(strict)
    {"id": "e4", "score": 0.99, "datatypeId": "known_drug",
     "datasourceId": "chembl", "publicationYear": 2015,
     "literature": [], "diseaseFromSource": "hypercholesterolemia"},
]


# --------------------------------------------------------------------------- #
# Step 5/6: Modal fan-out (run with `modal run engine.py::sweep`)
# --------------------------------------------------------------------------- #
import modal  # noqa: E402  (kept at module level so `modal run engine.py` discovers app)

modal_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "httpx>=0.27",
        "openai>=1.40",
        "python-dotenv",
        "pydantic>=2.0",
        "tenacity>=8.0",
    )
    .add_local_python_source("schemas")
)
modal_app = modal.App("greenlight-engine", image=modal_image)


@modal_app.function(
    secrets=[modal.Secret.from_name("openai-secret")],
    timeout=600,
    retries=modal.Retries(max_retries=2, initial_delay=2.0, backoff_coefficient=2.0),
)
def build_evidence_package_remote(target: str, disease: str, cutoff_iso: str,
                                  target_id: Optional[str] = None,
                                  disease_id: Optional[str] = None,
                                  enrich_budget: int = ENRICH_BUDGET_DEFAULT,
                                  max_items: Optional[int] = MAX_ITEMS_DEFAULT) -> dict:
    """Modal worker: one pair -> JSON-serialized EvidencePackage."""
    t0 = time.time()
    cutoff = date.fromisoformat(cutoff_iso)
    try:
        pkg = build_evidence_package(
            target, disease, cutoff,
            mock=False, enrich_budget=enrich_budget, max_items=max_items,
            target_id=target_id, disease_id=disease_id,
        )
        return pkg.model_dump(mode="json")
    except Exception as e:
        pkg = _empty_error_package(
            target, disease, cutoff,
            [f"worker:{type(e).__name__}:{str(e)[:300]}"],
            t0, target_id=target_id, disease_id=disease_id,
        )
        return pkg.model_dump(mode="json")


def _load_pairs(path: str) -> list[tuple[str, str, str, Optional[str], Optional[str]]]:
    """Read TSV rows.

    Supported formats:
      target<TAB>disease<TAB>cutoff_iso
      target<TAB>target_id<TAB>disease<TAB>disease_id<TAB>cutoff_iso
    """
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) == 3:
                target, disease, cutoff_iso = parts
                pairs.append((target, disease, cutoff_iso, None, None))
            elif len(parts) == 5:
                target, target_id, disease, disease_id, cutoff_iso = parts
                _validate_canonical_disease_id(disease_id)
                pairs.append((target, disease, cutoff_iso, target_id, disease_id))
            else:
                raise ValueError(
                    "bad pairs row (need 3 cols target,disease,cutoff or "
                    f"5 cols target,target_id,disease,disease_id,cutoff): {line!r}"
                )
    return pairs


_DEFAULT_PAIRS: list[tuple[str, str, str, Optional[str], Optional[str]]] = [
    ("PCSK9", "hypercholesterolemia", "2020-01-01", None, None),
    ("HMGCR", "hypercholesterolemia", "2020-01-01", None, None),
    ("LDLR",  "hypercholesterolemia", "2020-01-01", None, None),
]


@modal_app.local_entrypoint()
def sweep(pairs_path: str = "", enrich_budget: int = ENRICH_BUDGET_DEFAULT,
          max_items: int = MAX_ITEMS_DEFAULT,
          out_path: str = "sweep_results.jsonl") -> None:
    """Fan out build_evidence_package across pairs and print throughput."""
    triples = _load_pairs(pairs_path) if pairs_path else _DEFAULT_PAIRS
    inputs = [
        (t, d, c, tid, did, enrich_budget, max_items)
        for (t, d, c, tid, did) in triples
    ]
    print(f"sweeping {len(inputs)} pairs "
          f"(enrich_budget={enrich_budget}, max_items={max_items})...")
    t0 = time.time()
    results = list(build_evidence_package_remote.starmap(inputs))
    dt = time.time() - t0
    n_items = sum(len(r["evidence_items"]) for r in results)
    n_papers = sum(r["papers_processed"] for r in results)
    n_errors = sum(len(r.get("errors") or []) for r in results)
    rate = (len(results) / dt) * 60 if dt > 0 else float("inf")
    print(f"\n== sweep done ==")
    print(f"  {len(results)} pairs in {dt:.1f}s  ->  {rate:.1f} pairs/min")
    print(f"  papers_processed={n_papers}  evidence_items={n_items}  errors={n_errors}")
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"  wrote {out_path}")
    for r in results:
        print(f"  - {r['pair_id']:40} items={len(r['evidence_items']):3}  "
              f"papers={r['papers_processed']:3}  t={r['runtime_seconds']:.1f}s")


# --------------------------------------------------------------------------- #
# Local CLI (single pair, no Modal) — kept for fast iteration
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--target", default="PCSK9")
    ap.add_argument("--disease", default="hypercholesterolemia")
    ap.add_argument("--cutoff", default="2010-01-01")
    ap.add_argument("--target-id", default=None,
                    help="optional resolved Ensembl id from Person C benchmark")
    ap.add_argument("--disease-id", default=None,
                    help="optional resolved EFO/MONDO id from Person C benchmark")
    ap.add_argument("--enrich-budget", type=int, default=ENRICH_BUDGET_DEFAULT,
                    help="max literature-enriched rows per pair (bounds LLM + EPMC cost)")
    ap.add_argument("--max-items", type=int, default=MAX_ITEMS_DEFAULT,
                    help="max evidence items after cutoff/sorting; <=0 means no cap")
    a = ap.parse_args()

    pkg = build_evidence_package(
        a.target, a.disease, date.fromisoformat(a.cutoff),
        mock=a.mock, enrich_budget=a.enrich_budget,
        max_items=None if a.max_items <= 0 else a.max_items,
        target_id=a.target_id, disease_id=a.disease_id,
    )
    print(f"pair_id={pkg.pair_id}  cutoff={pkg.cutoff_date}")
    print(f"kept {len(pkg.evidence_items)} items "
          f"({pkg.papers_processed} papers, {pkg.runtime_seconds}s, "
          f"sources_queried={pkg.sources_queried}, errors={len(pkg.errors)})")
    for it in pkg.evidence_items:
        print(f"  [{it.evidence_type.value:13}] {it.publication_date}  "
              f"{it.source_db.value:11} {it.source_id:18} {it.claim[:55]}")
    if pkg.errors:
        print(f"  errors: {pkg.errors[:5]}")
    print("PASS — schema-valid EvidencePackage produced.")
