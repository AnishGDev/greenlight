"""
GreenLight — end-to-end workflow trace for ONE pair.

Walks a single (target, disease) through all three interfaces:
    C.build_benchmark()         -> BenchmarkPair        (the input + hidden label)
    A.run_engine(pair)          -> EvidencePackage      (gather evidence)
    B.verify_and_synthesize(pkg)-> Dossier              (check + score)
    C.score(dossier, pair)      -> PairResult / EvalResult (grade vs label)

The stage functions are MOCKED (hardcoded outputs) so the team can build and
test against realistic data immediately. Replace the bodies with real logic.

Sample data is ILLUSTRATIVE (PCSK9/LDL genetics are real and well established;
IDs and source text here are stand-ins for a demo, not literature citations).
"""

from __future__ import annotations

from datetime import date

from schemas import (
    BenchmarkPair, Dossier, EvidenceItem, EvidencePackage, EvalResult,
    EvidenceType, PairResult, Recommendation, SourceDB, Split,
    TIER_WEIGHTS, Verdict, VerifiedClaim, make_pair_id,
)

TARGET_SYMBOL = "PCSK9"
TARGET_ID = "ENSG00000169174"          # Ensembl gene id
DISEASE_NAME = "hypercholesterolemia"
DISEASE_ID = "EFO_0004911"             # EFO id
CUTOFF = date(2010, 1, 1)              # "hide everything after 2010"
PAIR_ID = make_pair_id(TARGET_ID, DISEASE_ID)


# =========================================================================== #
# STAGE 0 — Person C: build the benchmark (owns the ground-truth label)
# =========================================================================== #
def build_benchmark() -> list[BenchmarkPair]:
    return [
        BenchmarkPair(
            pair_id=PAIR_ID,
            target_symbol=TARGET_SYMBOL, target_id=TARGET_ID,
            disease_name=DISEASE_NAME, disease_id=DISEASE_ID,
            cutoff_date=CUTOFF,
            validated_after_cutoff=True,   # <-- LABEL. Never sent to A or B.
            validation_evidence="PCSK9 inhibitors (evolocumab/alirocumab) approved 2015",
            is_negative_control=False,
            split=Split.held_out,
        )
    ]


# =========================================================================== #
# STAGE 1 — Person A: run the evidence engine (gets identity + cutoff ONLY)
# =========================================================================== #
def run_engine(target_id: str, disease_id: str, cutoff_date: date) -> EvidencePackage:
    # NOTE: A receives no label. It only queries sources <= cutoff_date.
    items = [
        EvidenceItem(
            claim="PCSK9 loss-of-function variants are associated with lower LDL-C.",
            evidence_type=EvidenceType.genetic,
            source_db=SourceDB.pubmed, source_id="PMID:16554528",
            source_text=("Nonsense mutations in PCSK9 were associated with "
                         "28% lower LDL-C and reduced coronary events."),
            publication_date=date(2006, 3, 23),
        ),
        EvidenceItem(
            claim="PCSK9 LoF carriers show reduced incidence of coronary events.",
            evidence_type=EvidenceType.genetic,
            source_db=SourceDB.open_targets, source_id="OT:evidence:9921",
            source_text=("Cohort analysis: carriers had a markedly lower rate "
                         "of coronary heart disease over 15 years."),
            publication_date=date(2006, 3, 23),
        ),
        EvidenceItem(
            claim="PCSK9 binds the LDL receptor and promotes its degradation.",
            evidence_type=EvidenceType.in_vitro,
            source_db=SourceDB.pubmed, source_id="PMID:17341576",
            source_text=("In hepatocytes, secreted PCSK9 bound the LDLR and "
                         "routed it to lysosomal degradation."),
            publication_date=date(2007, 5, 1),
        ),
        # An LLM OVER-EXTRACTION: the source text does NOT support this claim.
        # B's Verifier should catch it -> the on-stage demo moment.
        EvidenceItem(
            claim="PCSK9 inhibition cures familial hypercholesterolemia in all patients.",
            evidence_type=EvidenceType.correlational,
            source_db=SourceDB.pubmed, source_id="PMID:18438315",
            source_text=("In a small cohort, an anti-PCSK9 antibody reduced "
                         "LDL-C levels over 12 weeks."),
            publication_date=date(2008, 9, 10),
        ),
    ]
    return EvidencePackage(
        pair_id=PAIR_ID,
        target_symbol=TARGET_SYMBOL, target_id=target_id,
        disease_name=DISEASE_NAME, disease_id=disease_id,
        cutoff_date=cutoff_date, evidence_items=items,
        sources_queried=6, papers_processed=58, runtime_seconds=4.2,
    )


# A also exposes a search() that B calls during its contradiction hunt.
def search(query: str, cutoff_date: date) -> list[EvidenceItem]:
    # MOCK: returns one illustrative caution item for the negative-evidence hunt.
    return [
        EvidenceItem(
            claim="Long-term safety of sustained very low LDL-C was not yet established.",
            evidence_type=EvidenceType.clinical,
            source_db=SourceDB.pubmed, source_id="PMID:19otc000",
            source_text=("As of this review, long-term outcomes of profound "
                         "LDL-C lowering remained under investigation."),
            publication_date=date(2009, 11, 2),
        )
    ]


# =========================================================================== #
# STAGE 2 — Person B: verify each claim, then synthesize the dossier
# =========================================================================== #
def _verify_one(item: EvidenceItem, verdict: Verdict, conf: float,
                negative: bool = False) -> VerifiedClaim:
    return VerifiedClaim(
        claim=item.claim, evidence_type=item.evidence_type,
        verdict=verdict, verdict_confidence=conf,
        tier_weight=TIER_WEIGHTS[item.evidence_type],
        is_negative_evidence=negative,
        source_db=item.source_db, source_id=item.source_id,
        source_url=item.source_url, source_text=item.source_text,
        publication_date=item.publication_date,
    )


def verify_and_synthesize(pkg: EvidencePackage) -> Dossier:
    its = pkg.evidence_items
    verified = [
        _verify_one(its[0], Verdict.supported, 0.95),     # genetic, strong
        _verify_one(its[1], Verdict.supported, 0.90),     # genetic, strong
        _verify_one(its[2], Verdict.supported, 0.88),     # in_vitro, mechanism
        _verify_one(its[3], Verdict.unsupported, 0.82),   # CAUGHT hallucination
    ]
    # contradiction hunt -> negative evidence via A.search()
    for neg in search("PCSK9 inhibition safety concerns", pkg.cutoff_date):
        verified.append(_verify_one(neg, Verdict.uncertain, 0.6, negative=True))

    supported = [v for v in verified if v.verdict == Verdict.supported]
    unsupported = [v for v in verified if v.verdict == Verdict.unsupported]

    # Tier-weighted score from SUPPORTED claims only; hallucinations excluded.
    raw = sum(v.tier_weight * v.verdict_confidence for v in supported)
    score = min(0.99, raw / 3.0)   # toy squashing -> probability-like [0,1]

    flags = []
    if unsupported:
        flags.append("hallucination_caught")
    if any(v.is_negative_evidence for v in verified):
        flags.append("open_safety_question")

    tier_breakdown: dict[str, int] = {}
    for v in verified:
        tier_breakdown[v.evidence_type.value] = tier_breakdown.get(v.evidence_type.value, 0) + 1

    return Dossier(
        pair_id=pkg.pair_id,
        target_symbol=pkg.target_symbol, target_id=pkg.target_id,
        disease_name=pkg.disease_name, disease_id=pkg.disease_id,
        cutoff_date=pkg.cutoff_date, verified_claims=verified,
        viability_score=round(score, 3),
        recommendation=Recommendation.pursue,
        rationale=("Strong human genetic support (LoF -> lower LDL-C and fewer "
                   "coronary events) plus a clear LDLR-degradation mechanism. "
                   "One over-stated claim was rejected; one open safety question noted."),
        supported_count=len(supported),
        contradicted_count=sum(1 for v in verified if v.verdict == Verdict.contradicted),
        unsupported_count=len(unsupported),
        tier_breakdown=tier_breakdown,
        flags=flags, model_used="gpt-mock", runtime_seconds=2.7,
    )


# =========================================================================== #
# STAGE 3 — Person C: grade the dossier against the hidden label
# =========================================================================== #
def score(dossiers: list[Dossier], benchmark: list[BenchmarkPair],
          top_k: int = 10) -> EvalResult:
    labels = {b.pair_id: b.validated_after_cutoff for b in benchmark}
    ranked = sorted(dossiers, key=lambda d: d.viability_score, reverse=True)
    per_pair = [
        PairResult(pair_id=d.pair_id, predicted_score=d.viability_score,
                   label=labels[d.pair_id], rank=i + 1)
        for i, d in enumerate(ranked)
    ]
    # citation validity = supported / (supported + unsupported) across dossiers
    sup = sum(d.supported_count for d in dossiers)
    uns = sum(d.unsupported_count for d in dossiers)
    cvr = sup / (sup + uns) if (sup + uns) else None

    hits = sum(1 for p in per_pair[:top_k] if p.label)
    pos = sum(1 for p in per_pair if p.label) or 1
    return EvalResult(
        split=Split.held_out, n_pairs=len(per_pair), top_k=top_k,
        top_k_hit_rate=hits / min(top_k, pos),
        citation_validity_rate=cvr,
        pairs_per_minute=58.0, total_papers_processed=58,
        per_pair=per_pair,
    )


# =========================================================================== #
# Run the trace
# =========================================================================== #
if __name__ == "__main__":
    line = "-" * 70

    print(line, "\nSTAGE 0  C -> builds benchmark (label kept private)")
    bench = build_benchmark()
    b = bench[0]
    print(f"  pair_id={b.pair_id}\n  cutoff={b.cutoff_date}  "
          f"validated_after_cutoff={b.validated_after_cutoff} (HIDDEN from A/B)")

    print(line, "\nSTAGE 1  A -> run_engine (sees identity + cutoff only)")
    pkg = run_engine(b.target_id, b.disease_id, b.cutoff_date)
    print(f"  {len(pkg.evidence_items)} evidence items, "
          f"{pkg.papers_processed} papers, {pkg.runtime_seconds}s")
    for it in pkg.evidence_items:
        print(f"    - [{it.evidence_type.value:13}] {it.claim[:60]}")

    print(line, "\nSTAGE 2  B -> verify_and_synthesize")
    dossier = verify_and_synthesize(pkg)
    for v in dossier.verified_claims:
        tag = " (NEGATIVE)" if v.is_negative_evidence else ""
        print(f"    - {v.verdict.value:12} conf={v.verdict_confidence:.2f} "
              f"tier={v.tier_weight:.2f}{tag}  {v.claim[:45]}")
    print(f"  viability_score={dossier.viability_score}  "
          f"recommendation={dossier.recommendation.value}")
    print(f"  flags={dossier.flags}")

    print(line, "\nSTAGE 3  C -> score vs hidden label")
    result = score([dossier], bench)
    pr = result.per_pair[0]
    print(f"  predicted={pr.predicted_score}  label={pr.label}  rank={pr.rank}")
    print(f"  top_{result.top_k}_hit_rate={result.top_k_hit_rate}  "
          f"citation_validity_rate={result.citation_validity_rate}")
    print(line, "\nOK — one pair flowed cleanly through all three contracts.")
