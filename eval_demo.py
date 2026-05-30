"""
eval_demo.py — Person C: Benchmark + Metrics (+ Streamlit demo in app.py).

Contracts:
  build_benchmark() -> list[BenchmarkPair]   (C -> A: the sweep runs over these)
  score(dossiers, benchmark) -> EvalResult    (the graded result for the demo)

Build against mocks first: workflow_example.build_benchmark and .score are
worked reference implementations you can copy and extend.
"""

from __future__ import annotations

from schemas import BenchmarkPair, Dossier, EvalResult  # noqa: F401


def build_benchmark() -> list[BenchmarkPair]:
    """
    TODO (C's tasks):
      - curate ~40 target-disease pairs with known later outcomes
      - label validated_after_cutoff using Open Targets / ClinicalTrials.gov
      - include negative controls; set split=held_out for the trusted set
    """
    raise NotImplementedError("C: implement build_benchmark")


def score(dossiers: list[Dossier], benchmark: list[BenchmarkPair]) -> EvalResult:
    """
    TODO (C's tasks):
      - top_k hit rate, AUPRC, Brier/calibration, citation_validity_rate
      - join dossiers to benchmark on pair_id
    """
    raise NotImplementedError("C: implement score")


# The demo UI lives in app.py and is run with:  streamlit run app.py
# It should call engine.build_evidence_package -> verifier.verify_and_synthesize
# and render: the run-a-dossier view, the verifier catch, and the metrics dashboard.
