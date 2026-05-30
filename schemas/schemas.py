"""
GreenLight — shared interface schemas (the three team contracts).

Data flow:
    C builds BenchmarkPair[]  ->  A runs the engine per pair  ->  EvidencePackage
    EvidencePackage           ->  B verifies + synthesizes    ->  Dossier
    Dossier[] + BenchmarkPair[] ->  C scores + renders demo    ->  EvalResult

Everyone imports from this file. Freeze it FIRST, then build against mocks.
Pydantic v2. To send across Modal / processes, use `.model_dump(mode="json")`.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------------- #
# Controlled vocabularies (enums stop A/B/C from drifting on string values)
# --------------------------------------------------------------------------- #
class SourceDB(str, Enum):
    open_targets = "open_targets"
    pubmed = "pubmed"
    europe_pmc = "europe_pmc"
    chembl = "chembl"
    clinicaltrials = "clinicaltrials"
    ctd = "ctd"
    disgenet = "disgenet"
    semantic_scholar = "semantic_scholar"


class EvidenceType(str, Enum):
    """Drives B's tiering. Ordered strongest -> weakest by convention."""
    genetic = "genetic"            # GWAS, Mendelian, genetic association
    clinical = "clinical"          # trial / human outcome data
    in_vivo = "in_vivo"            # animal model
    in_vitro = "in_vitro"          # cell / assay
    pathway = "pathway"            # mechanistic / pathway membership
    expression = "expression"      # differential expression
    correlational = "correlational"  # association / co-mention only


class Verdict(str, Enum):
    supported = "supported"        # source entails the claim
    contradicted = "contradicted"  # source refutes the claim
    unsupported = "unsupported"    # source does not back the claim (hallucination)
    uncertain = "uncertain"        # ambiguous / insufficient


class Recommendation(str, Enum):
    pursue = "pursue"
    investigate_further = "investigate_further"
    deprioritize = "deprioritize"


class Split(str, Enum):
    train = "train"          # the self-improvement loop may learn from these
    held_out = "held_out"    # the TRUSTED number; never seen by the loop


# Starting tier weights for B (tune freely). Genetic evidence dominates.
TIER_WEIGHTS: dict[EvidenceType, float] = {
    EvidenceType.genetic: 1.0,
    EvidenceType.clinical: 0.9,
    EvidenceType.in_vivo: 0.6,
    EvidenceType.in_vitro: 0.4,
    EvidenceType.pathway: 0.3,
    EvidenceType.expression: 0.3,
    EvidenceType.correlational: 0.15,
}


def make_pair_id(target_id: str, disease_id: str) -> str:
    """Deterministic join key shared by all three stages."""
    return f"{target_id}::{disease_id}"


# --------------------------------------------------------------------------- #
# CONTRACT 1 — A -> B   (evidence engine output)
# --------------------------------------------------------------------------- #
class EvidenceItem(BaseModel):
    """One extracted, citable factual claim. B verifies these one by one."""
    model_config = ConfigDict(extra="forbid")

    claim: str                                   # the extracted assertion
    evidence_type: EvidenceType
    source_db: SourceDB
    source_id: str                               # PMID / DOI / NCT / ChEMBL / OT id
    source_text: str                             # REQUIRED: passage B runs entailment on
    source_url: Optional[str] = None
    publication_date: Optional[date] = None      # used to enforce the cutoff
    retrieval_score: Optional[float] = None       # relevance from A's retriever
    metadata: dict = Field(default_factory=dict)  # escape hatch (organism, n, assay...)


class EvidencePackage(BaseModel):
    """Everything A's worker found for ONE (target, disease) pair."""
    model_config = ConfigDict(extra="forbid")

    pair_id: str                                  # make_pair_id(target_id, disease_id)
    target_symbol: str                            # e.g. "PCSK9"
    target_id: str                                # Ensembl gene id, e.g. "ENSG00000169174"
    disease_name: str                             # e.g. "hypercholesterolemia"
    disease_id: str                               # EFO / MONDO id, e.g. "EFO_0004911"
    cutoff_date: date                             # only sources on/before this were used
    evidence_items: list[EvidenceItem] = Field(default_factory=list)

    # observability / feeds the throughput number
    sources_queried: int = 0
    papers_processed: int = 0
    runtime_seconds: float = 0.0
    errors: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# CONTRACT 2 — B -> C   (verifier + synthesizer output)
# --------------------------------------------------------------------------- #
class VerifiedClaim(BaseModel):
    """An EvidenceItem after B checks it against its source."""
    model_config = ConfigDict(extra="forbid")

    claim: str
    evidence_type: EvidenceType
    verdict: Verdict
    verdict_confidence: float = Field(ge=0.0, le=1.0)  # from NLI / judge
    tier_weight: float = Field(ge=0.0, le=1.0)         # from TIER_WEIGHTS (tunable)
    is_negative_evidence: bool = False                 # found by contradiction hunt

    # carried through for the dossier's citations
    source_db: SourceDB
    source_id: str
    source_url: Optional[str] = None
    source_text: str
    publication_date: Optional[date] = None


class Dossier(BaseModel):
    """B's final per-pair output. viability_score is the headline number."""
    model_config = ConfigDict(extra="forbid")

    pair_id: str
    target_symbol: str
    target_id: str
    disease_name: str
    disease_id: str
    cutoff_date: date

    verified_claims: list[VerifiedClaim] = Field(default_factory=list)

    # THE outputs
    viability_score: float = Field(ge=0.0, le=1.0)  # probability-like -> C can calibrate
    recommendation: Recommendation
    rationale: str                                   # short synthesized explanation

    # quick stats for the demo UI
    supported_count: int = 0
    contradicted_count: int = 0
    unsupported_count: int = 0
    tier_breakdown: dict[str, int] = Field(default_factory=dict)  # evidence_type -> count
    flags: list[str] = Field(default_factory=list)   # e.g. "hallucination_caught"

    # observability
    model_used: Optional[str] = None
    runtime_seconds: float = 0.0


# --------------------------------------------------------------------------- #
# CONTRACT 3 — C -> A   (benchmark the sweep runs over)
# --------------------------------------------------------------------------- #
class BenchmarkPair(BaseModel):
    """One labelled target-disease pair for the retrospective test."""
    model_config = ConfigDict(extra="forbid")

    pair_id: str
    target_symbol: str
    target_id: str                                # Ensembl
    disease_name: str
    disease_id: str                               # EFO / MONDO
    cutoff_date: date                             # the "hide the future" line for this pair

    validated_after_cutoff: bool                  # GROUND TRUTH label
    validation_evidence: Optional[str] = None     # what validated it (OT score / NCT / approval)
    is_negative_control: bool = False             # known dud, needed for AUPRC negatives
    split: Split = Split.held_out
    notes: Optional[str] = None


# --------------------------------------------------------------------------- #
# (Bonus) C-internal — scoring output for the dashboard
# --------------------------------------------------------------------------- #
class PairResult(BaseModel):
    pair_id: str
    predicted_score: float
    label: bool                                   # validated_after_cutoff
    rank: Optional[int] = None


class EvalResult(BaseModel):
    """Aggregate metrics C renders in the demo."""
    model_config = ConfigDict(extra="forbid")

    split: Split
    n_pairs: int
    top_k: int
    top_k_hit_rate: float                          # headline
    auprc: Optional[float] = None
    brier_score: Optional[float] = None            # calibration
    calibration_bins: list[dict] = Field(default_factory=list)  # for reliability curve
    citation_validity_rate: Optional[float] = None  # supported / total verified claims
    # scale story
    pairs_per_minute: Optional[float] = None
    total_papers_processed: Optional[int] = None
    per_pair: list[PairResult] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Smoke test / usage example
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    pid = make_pair_id("ENSG00000169174", "EFO_0004911")

    item = EvidenceItem(
        claim="PCSK9 loss-of-function variants lower LDL cholesterol.",
        evidence_type=EvidenceType.genetic,
        source_db=SourceDB.pubmed,
        source_id="16554528",
        source_text="Nonsense mutations in PCSK9 were associated with reduced LDL...",
        publication_date=date(2006, 3, 15),
    )
    pkg = EvidencePackage(
        pair_id=pid, target_symbol="PCSK9", target_id="ENSG00000169174",
        disease_name="hypercholesterolemia", disease_id="EFO_0004911",
        cutoff_date=date(2010, 1, 1), evidence_items=[item],
        sources_queried=5, papers_processed=42, runtime_seconds=3.1,
    )

    vclaim = VerifiedClaim(
        claim=item.claim, evidence_type=item.evidence_type,
        verdict=Verdict.supported, verdict_confidence=0.94,
        tier_weight=TIER_WEIGHTS[item.evidence_type],
        source_db=item.source_db, source_id=item.source_id,
        source_text=item.source_text, publication_date=item.publication_date,
    )
    dossier = Dossier(
        pair_id=pid, target_symbol="PCSK9", target_id="ENSG00000169174",
        disease_name="hypercholesterolemia", disease_id="EFO_0004911",
        cutoff_date=date(2010, 1, 1), verified_claims=[vclaim],
        viability_score=0.87, recommendation=Recommendation.pursue,
        rationale="Strong human genetic support; LoF lowers LDL.",
        supported_count=1, tier_breakdown={"genetic": 1},
    )
    bench = BenchmarkPair(
        pair_id=pid, target_symbol="PCSK9", target_id="ENSG00000169174",
        disease_name="hypercholesterolemia", disease_id="EFO_0004911",
        cutoff_date=date(2010, 1, 1), validated_after_cutoff=True,
        validation_evidence="PCSK9 inhibitors approved (evolocumab 2015)",
        split=Split.held_out,
    )

    for m in (pkg, dossier, bench):
        # round-trips to JSON cleanly -> safe across Modal / processes
        m.model_dump(mode="json")
    print("OK — all schemas construct and serialize. pair_id =", pid)
