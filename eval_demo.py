"""
eval_demo.py — Person C: Benchmark + Metrics (+ Streamlit demo in app.py).

Contracts:
  build_benchmark() -> list[BenchmarkPair]   (C -> A: the sweep runs over these)
  score(dossiers, benchmark) -> EvalResult    (the graded result for the demo)

Build against mocks first: workflow_example.build_benchmark and .score are
worked reference implementations you can copy and extend.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from schemas import BenchmarkPair, Dossier, EvalResult, PairResult, Split, make_pair_id


CURATED_LABELS: dict[tuple[str, str], tuple[bool, str, bool]] = {
    ("PCSK9", "hypercholesterolemia"): (
        True,
        "PCSK9 inhibitors approved and outcomes data supported LDL/CV benefit after the cutoff.",
        False,
    ),
    ("HMGCR", "hypercholesterolemia"): (
        True,
        "Statin target remained clinically validated by long-running LDL/CV outcomes evidence.",
        False,
    ),
    ("LDLR", "hypercholesterolemia"): (
        True,
        "LDLR biology remained central to familial hypercholesterolemia validation.",
        False,
    ),
    ("ANGPTL3", "hypercholesterolemia"): (
        True,
        "ANGPTL3 inhibition advanced clinically for lipid disorders after the cutoff.",
        False,
    ),
    ("IL17A", "psoriasis"): (
        True,
        "IL-17A blockade became a validated psoriasis therapeutic strategy.",
        False,
    ),
    ("TNF", "rheumatoid arthritis"): (
        True,
        "TNF blockade was already and remained clinically validated for rheumatoid arthritis.",
        False,
    ),
    ("GAPDH", "hypercholesterolemia"): (
        False,
        "Housekeeping-gene negative control; no target-disease validation expected.",
        True,
    ),
    ("ACTB", "type 2 diabetes mellitus"): (
        False,
        "Housekeeping-gene negative control; no target-disease validation expected.",
        True,
    ),
}


def _read_pairs_tsv(path: Path) -> list[tuple[str, str, date]]:
    rows: list[tuple[str, str, date]] = []
    if not path.exists():
        return rows
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) == 3:
            target, disease, cutoff_iso = parts
        elif len(parts) == 5:
            target, _target_id, disease, _disease_id, cutoff_iso = parts
        else:
            continue
        try:
            cutoff = date.fromisoformat(cutoff_iso)
        except ValueError:
            continue
        rows.append((target, disease, cutoff))
    return rows


def _known_ids(target: str, disease: str) -> tuple[str, str]:
    known: dict[tuple[str, str], tuple[str, str]] = {
        ("PCSK9", "hypercholesterolemia"): ("ENSG00000169174", "EFO_0004911"),
        ("HMGCR", "hypercholesterolemia"): ("ENSG00000113161", "EFO_0004911"),
        ("LDLR", "hypercholesterolemia"): ("ENSG00000130164", "EFO_0004911"),
        ("ANGPTL3", "hypercholesterolemia"): ("ENSG00000132855", "EFO_0004911"),
        ("IL17A", "psoriasis"): ("ENSG00000112115", "EFO_0000676"),
        ("TNF", "rheumatoid arthritis"): ("ENSG00000232810", "EFO_0000685"),
        ("GAPDH", "hypercholesterolemia"): ("ENSG00000111640", "EFO_0004911"),
        ("ACTB", "type 2 diabetes mellitus"): ("ENSG00000075624", "EFO_0001360"),
    }
    return known.get((target, disease), (target, disease.replace(" ", "_")))


def build_benchmark() -> list[BenchmarkPair]:
    """Build a labelled retrospective benchmark from the TSV plus curated labels."""
    rows = _read_pairs_tsv(Path("pairs_full.tsv"))
    if not rows:
        rows = [(target, disease, date(2018, 1, 1)) for target, disease in CURATED_LABELS]

    benchmark: list[BenchmarkPair] = []
    for target, disease, cutoff in rows:
        label = CURATED_LABELS.get((target, disease))
        if label is None:
            continue
        validated, evidence, is_negative = label
        target_id, disease_id = _known_ids(target, disease)
        benchmark.append(
            BenchmarkPair(
                pair_id=make_pair_id(target_id, disease_id),
                target_symbol=target,
                target_id=target_id,
                disease_name=disease,
                disease_id=disease_id,
                cutoff_date=cutoff,
                validated_after_cutoff=validated,
                validation_evidence=evidence,
                is_negative_control=is_negative,
                split=Split.held_out,
            )
        )
    return benchmark


def _average_precision(labels: list[bool], scores: list[float]) -> float | None:
    positives = sum(1 for label in labels if label)
    if positives == 0:
        return None
    ranked = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    hits = 0
    precision_sum = 0.0
    for idx, (_score, label) in enumerate(ranked, start=1):
        if label:
            hits += 1
            precision_sum += hits / idx
    return precision_sum / positives


def _brier(labels: list[bool], scores: list[float]) -> float | None:
    if not labels:
        return None
    return sum((score - float(label)) ** 2 for label, score in zip(labels, scores)) / len(labels)


def _calibration_bins(labels: list[bool], scores: list[float], n_bins: int = 5) -> list[dict]:
    bins: list[dict] = []
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        if i == n_bins - 1:
            members = [(label, score) for label, score in zip(labels, scores) if lo <= score <= hi]
        else:
            members = [(label, score) for label, score in zip(labels, scores) if lo <= score < hi]
        if not members:
            continue
        bin_labels, bin_scores = zip(*members)
        bins.append(
            {
                "bin_start": lo,
                "bin_end": hi,
                "n": len(members),
                "mean_score": sum(bin_scores) / len(bin_scores),
                "empirical_rate": sum(1 for label in bin_labels if label) / len(bin_labels),
            }
        )
    return bins


def score(dossiers: list[Dossier], benchmark: list[BenchmarkPair]) -> EvalResult:
    """Score labelled dossiers against the hidden retrospective labels."""
    labels_by_pair = {
        b.pair_id: b.validated_after_cutoff
        for b in benchmark
        if b.validated_after_cutoff is not None
    }
    labelled = [d for d in dossiers if d.pair_id in labels_by_pair]
    ranked = sorted(labelled, key=lambda d: d.viability_score, reverse=True)
    per_pair = [
        PairResult(
            pair_id=d.pair_id,
            predicted_score=d.viability_score,
            label=bool(labels_by_pair[d.pair_id]),
            rank=i + 1,
        )
        for i, d in enumerate(ranked)
    ]

    labels = [p.label for p in per_pair]
    scores = [p.predicted_score for p in per_pair]
    top_k = min(10, len(per_pair)) or 0
    positives = sum(1 for label in labels if label)
    hits = sum(1 for p in per_pair[:top_k] if p.label)
    denom = min(top_k, positives) if positives and top_k else 1

    supported = sum(1 for d in labelled for v in d.verified_claims if v.verdict.value == "supported")
    total_verified = sum(len(d.verified_claims) for d in labelled)
    citation_validity = supported / total_verified if total_verified else None
    runtime_total = sum(d.runtime_seconds for d in labelled)
    pairs_per_minute = (len(labelled) / runtime_total) * 60 if runtime_total > 0 else None

    split = next((b.split for b in benchmark if b.split is not None), Split.held_out)
    return EvalResult(
        split=split,
        n_pairs=len(per_pair),
        top_k=top_k,
        top_k_hit_rate=hits / denom,
        auprc=_average_precision(labels, scores),
        brier_score=_brier(labels, scores),
        calibration_bins=_calibration_bins(labels, scores),
        citation_validity_rate=citation_validity,
        pairs_per_minute=pairs_per_minute,
        total_papers_processed=None,
        per_pair=per_pair,
    )


# The demo UI lives in app.py and is run with:  streamlit run app.py
# It should call engine.build_evidence_package -> verifier.verify_and_synthesize
# and render: the run-a-dossier view, the verifier catch, and the metrics dashboard.
