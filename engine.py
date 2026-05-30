"""
engine.py — Person A, STEP 1: Open Targets -> first EvidencePackage.

What this gives you:
  - resolve a gene symbol -> Ensembl id, and a disease name -> EFO id
  - pull typed target-disease evidence from Open Targets
  - map it onto our EvidenceType enum
  - apply the cutoff filter (NO FUTURE LEAK)
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
import time
from datetime import date

from schemas import (
    EvidenceItem, EvidencePackage, EvidenceType, SourceDB, make_pair_id,
)

OT_URL = "https://api.platform.opentargets.org/api/v4/graphql"

# Open Targets datatypeId -> our EvidenceType
OT_TYPE_MAP: dict[str, EvidenceType] = {
    "genetic_association": EvidenceType.genetic,
    "somatic_mutation": EvidenceType.genetic,
    "known_drug": EvidenceType.clinical,
    "affected_pathway": EvidenceType.pathway,
    "animal_model": EvidenceType.in_vivo,
    "rna_expression": EvidenceType.expression,
    "literature": EvidenceType.correlational,
}


# --------------------------------------------------------------------------- #
# Live GraphQL (only used when --mock is NOT set)
# --------------------------------------------------------------------------- #
def _gql(query: str, variables: dict) -> dict:
    import httpx  # imported lazily so --mock needs no network/deps
    r = httpx.post(OT_URL, json={"query": query, "variables": variables}, timeout=30)
    r.raise_for_status()
    payload = r.json()
    if "errors" in payload:
        raise RuntimeError(payload["errors"])
    return payload["data"]


_RESOLVE = """
query Resolve($q: String!) {
  search(queryString: $q, entityNames: ["target","disease"]) {
    hits { id name entity }
  }
}
"""

# VALIDATE field names live before relying on this.
_EVIDENCE = """
query Evidence($efoId: String!, $ensemblId: String!) {
  disease(efoId: $efoId) {
    evidences(ensemblIds: [$ensemblId], size: 200) {
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


def resolve_ids(target: str, disease: str) -> tuple[str, str]:
    data = _gql(_RESOLVE, {"q": target})
    ens = next(h["id"] for h in data["search"]["hits"] if h["entity"] == "target")
    data = _gql(_RESOLVE, {"q": disease})
    efo = next(h["id"] for h in data["search"]["hits"] if h["entity"] == "disease")
    return ens, efo


def fetch_ot_rows(ensembl_id: str, efo_id: str) -> list[dict]:
    data = _gql(_EVIDENCE, {"efoId": efo_id, "ensemblId": ensembl_id})
    return data["disease"]["evidences"]["rows"]


# --------------------------------------------------------------------------- #
# Row -> EvidenceItem, with the cutoff gate
# --------------------------------------------------------------------------- #
def row_to_item(row: dict, target: str, disease: str) -> EvidenceItem | None:
    etype = OT_TYPE_MAP.get(row.get("datatypeId", ""), EvidenceType.correlational)
    year = row.get("publicationYear")
    pub = date(year, 1, 1) if year else None
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
# The worker: (target, disease, cutoff) -> EvidencePackage   (NO label in/out)
# --------------------------------------------------------------------------- #
def build_evidence_package(target: str, disease: str, cutoff_date: date,
                           mock: bool = False) -> EvidencePackage:
    t0 = time.time()
    if mock:
        ensembl_id, efo_id = "ENSG00000169174", "EFO_0004911"
        rows = _MOCK_ROWS
    else:
        ensembl_id, efo_id = resolve_ids(target, disease)
        rows = fetch_ot_rows(ensembl_id, efo_id)

    raw_items = [it for r in rows if (it := row_to_item(r, target, disease))]
    items = apply_cutoff(raw_items, cutoff_date, strict=True)

    return EvidencePackage(
        pair_id=make_pair_id(ensembl_id, efo_id),
        target_symbol=target, target_id=ensembl_id,
        disease_name=disease, disease_id=efo_id,
        cutoff_date=cutoff_date, evidence_items=items,
        sources_queried=1,                       # just Open Targets in Step 1
        papers_processed=len({p for r in rows for p in (r.get("literature") or [])}),
        runtime_seconds=round(time.time() - t0, 3),
    )


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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--target", default="PCSK9")
    ap.add_argument("--disease", default="hypercholesterolemia")
    ap.add_argument("--cutoff", default="2010-01-01")
    a = ap.parse_args()

    pkg = build_evidence_package(
        a.target, a.disease, date.fromisoformat(a.cutoff), mock=a.mock,
    )
    print(f"pair_id={pkg.pair_id}  cutoff={pkg.cutoff_date}")
    print(f"kept {len(pkg.evidence_items)} items "
          f"({pkg.papers_processed} papers, {pkg.runtime_seconds}s)")
    for it in pkg.evidence_items:
        print(f"  [{it.evidence_type.value:13}] {it.publication_date}  {it.claim[:55]}")
    print("PASS — schema-valid EvidencePackage produced.")
