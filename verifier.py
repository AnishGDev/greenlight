"""
verifier.py — Person B: Verifier + Synthesizer.

Contract: EvidencePackage (from A) -> Dossier (to C).
Build against mocks first: see workflow_example.verify_and_synthesize for a
fully worked reference implementation you can copy and replace piece by piece.
"""

from __future__ import annotations

from schemas import Dossier, EvidencePackage  # noqa: F401


def verify_and_synthesize(pkg: EvidencePackage) -> Dossier:
    """
    TODO (B's tasks):
      1. claim extractor  -> discrete checkable claims + cited source
      2. entailment Verifier (OpenAI strict prompt first; local NLI later)
      3. contradiction / negative-evidence hunt  (call A's engine.search())
      4. evidence tiering (use schemas.TIER_WEIGHTS as the starting point)
      5. synthesize -> Dossier with calibratable viability_score in [0, 1]
      6. set flags=['hallucination_caught'] when a claim is 'unsupported'
    """
    raise NotImplementedError("B: implement verify_and_synthesize")


if __name__ == "__main__":
    # Smoke test against A's mock package:
    from workflow_example import run_engine
    from datetime import date
    pkg = run_engine("ENSG00000169174", "EFO_0004911", date(2010, 1, 1))
    print(verify_and_synthesize(pkg))
