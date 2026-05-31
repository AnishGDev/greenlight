"""Schema package exports for the GreenLight app.

This lets top-level code import from `schemas` instead of `schemas.schemas`.
"""

from .schemas import BenchmarkPair, Dossier, EvalResult, EvidenceItem, EvidencePackage, PairResult, VerifiedClaim
from .schemas import Recommendation, Split, SourceDB, Verdict, EvidenceType
from .schemas import TIER_WEIGHTS, make_pair_id

__all__ = [
    "BenchmarkPair",
    "Dossier",
    "EvalResult",
    "EvidenceItem",
    "EvidencePackage",
    "PairResult",
    "VerifiedClaim",
    "Recommendation",
    "Split",
    "SourceDB",
    "EvidenceType",
    "TIER_WEIGHTS",
    "make_pair_id",
]
