"""Retrospective validation service for post-2010 target-disease evidence.

Pipeline:
1. Pull post-cutoff evidence from Open Targets using Ensembl + EFO/MONDO IDs.
2. Pull post-cutoff publications from Europe PMC using target + disease terms.
3. Summarize the combined evidence with an OpenAI Chat Completions call.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import httpx
from openai import OpenAI


OT_URL = "https://api.platform.opentargets.org/api/v4/graphql"
ENSEMBL_LOOKUP_ID_URL = "https://rest.ensembl.org/lookup/id/{ensembl_id}"
OLS_SEARCH_URL = "https://www.ebi.ac.uk/ols4/api/search"
EUROPE_PMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

HEADERS_JSON = {"Content-Type": "application/json", "Accept": "application/json"}
TIMEOUT = 20.0


_OT_EVIDENCE_QUERY = """
query Evidence($diseaseId: String!, $ensemblId: String!) {
  disease(efoId: $diseaseId) {
    evidences(ensemblIds: [$ensemblId], size: 500) {
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


@dataclass
class RetrospectiveResult:
    target_symbol: str
    target_id: str
    disease_name: str
    disease_id: str
    cutoff_year: int
    recommendation_2010: str
    overall_verdict: str
    confidence: float
    summary: str
    findings: list[dict[str, str]]
    warnings: list[str]
    sources: list[dict[str, str]]
    used_gpt: bool
    ot_evidence_count: int
    europe_pmc_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_symbol": self.target_symbol,
            "target_id": self.target_id,
            "disease_name": self.disease_name,
            "disease_id": self.disease_id,
            "cutoff_year": self.cutoff_year,
            "recommendation_2010": self.recommendation_2010,
            "overall_verdict": self.overall_verdict,
            "confidence": self.confidence,
            "summary": self.summary,
            "findings": self.findings,
            "warnings": self.warnings,
            "sources": self.sources,
            "used_gpt": self.used_gpt,
            "ot_evidence_count": self.ot_evidence_count,
            "europe_pmc_count": self.europe_pmc_count,
        }


def _post_ot(query: str, variables: dict[str, Any]) -> dict[str, Any]:
    response = httpx.post(
        OT_URL,
        json={"query": query, "variables": variables},
        headers=HEADERS_JSON,
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    if "errors" in payload:
        raise RuntimeError(str(payload["errors"]))

    data = payload.get("data")
    if not isinstance(data, dict):
        return {}
    return data


def fetch_ot_post_cutoff_evidence(
    ensembl_id: str,
    disease_id: str,
    start_year: int,
    max_rows: int = 30,
) -> list[dict[str, Any]]:
    data = _post_ot(
        _OT_EVIDENCE_QUERY,
        {"diseaseId": disease_id, "ensemblId": ensembl_id},
    )
    disease_node = data.get("disease")
    if not isinstance(disease_node, dict):
        return []

    evidences_node = disease_node.get("evidences")
    if not isinstance(evidences_node, dict):
        return []

    rows = evidences_node.get("rows")
    if not isinstance(rows, list):
        return []

    filtered: list[dict[str, Any]] = []
    for row in rows:
        year = row.get("publicationYear")
        if not isinstance(year, int) or year < start_year:
            continue
        filtered.append(row)

    filtered.sort(key=lambda r: (r.get("score") or 0.0), reverse=True)
    return filtered[:max_rows]


def fetch_target_symbol_from_ensembl(ensembl_id: str) -> str | None:
    response = httpx.get(
        ENSEMBL_LOOKUP_ID_URL.format(ensembl_id=ensembl_id),
        headers=HEADERS_JSON,
        timeout=TIMEOUT,
    )
    if response.status_code != 200:
        return None
    data = response.json()
    symbol = data.get("display_name")
    return symbol if isinstance(symbol, str) and symbol.strip() else None


def fetch_disease_label_from_ols(disease_id: str) -> str | None:
    response = httpx.get(
        OLS_SEARCH_URL,
        params={
            "q": disease_id,
            "type": "class",
            "rows": 10,
        },
        headers=HEADERS_JSON,
        timeout=TIMEOUT,
    )
    if response.status_code != 200:
        return None

    docs = response.json().get("response", {}).get("docs", [])
    wanted = disease_id.strip().lower()
    for doc in docs:
        short_form = str(doc.get("short_form", "")).lower()
        if short_form == wanted:
            label = doc.get("label")
            if isinstance(label, str) and label.strip():
                return label

    if docs:
        label = docs[0].get("label")
        if isinstance(label, str) and label.strip():
            return label
    return None


def fetch_europe_pmc_post_cutoff(
    target_symbol: str,
    disease_name: str,
    start_year: int,
    page_size: int = 20,
) -> list[dict[str, Any]]:
    query = f'("{target_symbol}" AND "{disease_name}") AND PUB_YEAR:[{start_year} TO 2100]'
    response = httpx.get(
        EUROPE_PMC_SEARCH_URL,
        params={
            "query": query,
            "format": "json",
            "sort": "CITED desc",
            "pageSize": page_size,
            "resultType": "core",
        },
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    results = response.json().get("resultList", {}).get("result", []) or []

    publications: list[dict[str, Any]] = []
    for item in results:
        year_raw = item.get("pubYear")
        try:
            year = int(year_raw)
        except (TypeError, ValueError):
            continue
        if year < start_year:
            continue

        pmid = item.get("pmid")
        doi = item.get("doi")
        source_id = f"PMID:{pmid}" if pmid else (f"DOI:{doi}" if doi else "n/a")

        publications.append(
            {
                "title": item.get("title") or "Untitled",
                "year": str(year),
                "source_id": source_id,
                "journal": item.get("journalTitle") or "Unknown journal",
                "url": f"https://europepmc.org/article/MED/{pmid}" if pmid else "",
                "abstract": item.get("abstractText") or "",
            }
        )
    return publications


def fetch_europe_pmc_abstract_by_pmid(pmid: str) -> str:
    response = httpx.get(
        EUROPE_PMC_SEARCH_URL,
        params={
            "query": f"EXT_ID:{pmid} AND SRC:MED",
            "format": "json",
            "resultType": "core",
            "pageSize": 1,
        },
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    results = response.json().get("resultList", {}).get("result", []) or []
    if not results:
        return ""
    return str(results[0].get("abstractText") or "")


def enrich_ot_rows_with_abstracts(ot_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach abstracts to Open Targets rows where PMID is available."""
    enriched_rows: list[dict[str, Any]] = []
    for row in ot_rows:
        row_copy = dict(row)
        pmids = row_copy.get("literature") or []
        pmid = str(pmids[0]).strip() if pmids else ""
        abstract_text = ""
        if pmid:
            try:
                abstract_text = fetch_europe_pmc_abstract_by_pmid(pmid)
            except Exception:
                abstract_text = ""
        row_copy["abstract"] = abstract_text
        enriched_rows.append(row_copy)
    return enriched_rows


def _to_evidence_lines(
    ot_rows: list[dict[str, Any]],
    epmc_rows: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, str]]]:
    lines: list[str] = []
    sources: list[dict[str, str]] = []

    for row in ot_rows[:12]:
        pmids = row.get("literature") or []
        pmid = str(pmids[0]) if pmids else ""
        source_id = f"PMID:{pmid}" if pmid else f"OT:{row.get('id', 'unknown')}"
        year = row.get("publicationYear")
        score = row.get("score")
        datatype = row.get("datatypeId") or "unknown"
        datasource = row.get("datasourceId") or "unknown"
        disease_from_source = row.get("diseaseFromSource") or ""
        abstract_text = str(row.get("abstract") or "")
        abstract_snippet = " ".join(abstract_text.split())[:900]

        line = (
            f"OpenTargets | year={year} | datatype={datatype} | datasource={datasource} "
            f"| score={score} | source={source_id} | text={disease_from_source} "
            f"| abstract={abstract_snippet}"
        )
        lines.append(line)
        sources.append(
            {
                "source": "Open Targets",
                "source_id": source_id,
                "year": str(year),
                "detail": f"{datatype} via {datasource}",
                "abstract": abstract_snippet,
            }
        )

    for item in epmc_rows[:12]:
        abstract_text = str(item.get("abstract") or "")
        abstract_snippet = " ".join(abstract_text.split())[:900]
        line = (
            f"EuropePMC | year={item['year']} | source={item['source_id']} "
            f"| journal={item['journal']} | title={item['title']} | abstract={abstract_snippet}"
        )
        lines.append(line)
        sources.append(
            {
                "source": "Europe PMC",
                "source_id": item["source_id"],
                "year": item["year"],
                "detail": item["title"],
                "url": item.get("url", ""),
                "abstract": abstract_snippet,
            }
        )

    return lines, sources


def _heuristic_fallback(
    recommendation_2010: str,
    ot_rows: list[dict[str, Any]],
    epmc_rows: list[dict[str, Any]],
) -> tuple[str, float, str, list[dict[str, str]]]:
    known_drug_count = sum(1 for row in ot_rows if row.get("datatypeId") == "known_drug")
    high_score_count = sum(1 for row in ot_rows if (row.get("score") or 0.0) >= 0.7)
    paper_count = len(epmc_rows)

    if known_drug_count > 0 or high_score_count >= 3 or paper_count >= 12:
        trend = "validated"
        confidence = 0.67
        finding = "Post-2010 evidence volume and/or known-drug signals suggest the pair validated."
    elif paper_count <= 2 and high_score_count == 0:
        trend = "not_validated"
        confidence = 0.58
        finding = "Limited post-2010 signal was found, suggesting weak validation support."
    else:
        trend = "inconclusive"
        confidence = 0.45
        finding = "Signals are mixed without enough strength to make a confident retrospective call."

    recommendation_is_pursue = recommendation_2010.strip().lower() == "pursue"
    if trend == "inconclusive":
        overall_verdict = "Mixed / Inconclusive"
    elif (trend == "validated" and recommendation_is_pursue) or (
        trend == "not_validated" and not recommendation_is_pursue
    ):
        overall_verdict = "Likely Correct"
    else:
        overall_verdict = "Likely Incorrect"

    findings = [
        {
            "signal": "Open Targets post-cutoff evidence",
            "finding": f"High-score items: {high_score_count}; known_drug items: {known_drug_count}.",
            "result": "Supports" if trend == "validated" else ("Refutes" if trend == "not_validated" else "Mixed"),
        },
        {
            "signal": "Europe PMC publication volume",
            "finding": f"Post-cutoff matched publications found: {paper_count}.",
            "result": "Supports" if paper_count >= 12 else ("Refutes" if paper_count <= 2 else "Mixed"),
        },
    ]

    return overall_verdict, confidence, finding, findings


def _gpt_summarize(
    target_symbol: str,
    disease_name: str,
    recommendation_2010: str,
    evidence_lines: list[str],
) -> tuple[str, float, str, list[dict[str, str]]]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    client = OpenAI(api_key=api_key)

    system_prompt = (
        "You are a biomedical retrospective analyst. "
        "Given post-cutoff evidence for one target-disease pair, evaluate whether a 2010 recommendation held up. "
        "Return strict JSON only."
    )

    user_payload = {
        "task": "Summarize post-2010 evidence and determine if the 2010 recommendation was correct.",
        "target_symbol": target_symbol,
        "disease_name": disease_name,
        "recommendation_2010": recommendation_2010,
        "required_output_schema": {
            "overall_verdict": "one of: Likely Correct | Likely Incorrect | Mixed / Inconclusive",
            "confidence": "float between 0 and 1",
            "summary": "2-4 sentence high-signal summary",
            "findings": [
                {
                    "signal": "short label",
                    "finding": "short evidence-backed statement",
                    "result": "one of: Supports | Refutes | Mixed",
                }
            ],
        },
        "evidence_lines": evidence_lines[:24],
    }

    completion = client.chat.completions.create(
        model=model,
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload)},
        ],
    )

    content = completion.choices[0].message.content or "{}"
    parsed = json.loads(content)

    overall_verdict = str(parsed.get("overall_verdict", "Mixed / Inconclusive"))
    confidence = float(parsed.get("confidence", 0.5))
    summary = str(parsed.get("summary", "No summary returned."))

    findings_raw = parsed.get("findings", [])
    findings: list[dict[str, str]] = []
    if isinstance(findings_raw, list):
        for item in findings_raw[:6]:
            if isinstance(item, dict):
                findings.append(
                    {
                        "signal": str(item.get("signal", "Signal")),
                        "finding": str(item.get("finding", "")),
                        "result": str(item.get("result", "Mixed")),
                    }
                )

    if not findings:
        findings.append(
            {
                "signal": "Model output",
                "finding": "The model returned an empty findings list.",
                "result": "Mixed",
            }
        )

    return overall_verdict, max(0.0, min(1.0, confidence)), summary, findings


def run_retrospective_validation(
    target_symbol: str,
    target_id: str,
    disease_name: str,
    disease_id: str,
    recommendation_2010: str,
    cutoff_year: int = 2010,
) -> dict[str, Any]:
    warnings: list[str] = []
    start_year = cutoff_year + 1

    symbol_from_id = None
    disease_from_id = None

    try:
        symbol_from_id = fetch_target_symbol_from_ensembl(target_id)
    except Exception as exc:
        warnings.append(f"Could not resolve target label from Ensembl: {exc}")

    try:
        disease_from_id = fetch_disease_label_from_ols(disease_id)
    except Exception as exc:
        warnings.append(f"Could not resolve disease label from OLS: {exc}")

    query_target = symbol_from_id or target_symbol
    query_disease = disease_from_id or disease_name

    ot_rows: list[dict[str, Any]] = []
    epmc_rows: list[dict[str, Any]] = []

    try:
        ot_rows = fetch_ot_post_cutoff_evidence(target_id, disease_id, start_year=start_year)
    except Exception as exc:
        warnings.append(f"Open Targets query failed: {exc}")

    if ot_rows:
        ot_rows = enrich_ot_rows_with_abstracts(ot_rows)

    try:
        epmc_rows = fetch_europe_pmc_post_cutoff(query_target, query_disease, start_year=start_year)
    except Exception as exc:
        warnings.append(f"Europe PMC query failed: {exc}")

    evidence_lines, sources = _to_evidence_lines(ot_rows, epmc_rows)
    used_gpt = False

    try:
        overall_verdict, confidence, summary, findings = _gpt_summarize(
            target_symbol=query_target,
            disease_name=query_disease,
            recommendation_2010=recommendation_2010,
            evidence_lines=evidence_lines,
        )
        used_gpt = True
    except Exception as exc:
        warnings.append(f"GPT summarization unavailable, used heuristic fallback: {exc}")
        overall_verdict, confidence, summary, findings = _heuristic_fallback(
            recommendation_2010=recommendation_2010,
            ot_rows=ot_rows,
            epmc_rows=epmc_rows,
        )

    result = RetrospectiveResult(
        target_symbol=query_target,
        target_id=target_id,
        disease_name=query_disease,
        disease_id=disease_id,
        cutoff_year=cutoff_year,
        recommendation_2010=recommendation_2010,
        overall_verdict=overall_verdict,
        confidence=confidence,
        summary=summary,
        findings=findings,
        warnings=warnings,
        sources=sources[:20],
        used_gpt=used_gpt,
        ot_evidence_count=len(ot_rows),
        europe_pmc_count=len(epmc_rows),
    )
    return result.to_dict()
