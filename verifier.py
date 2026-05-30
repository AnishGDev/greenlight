"""
verifier.py — Person B: Verifier + Synthesizer (CPU-only v1).

Contract: EvidencePackage (from A) -> Dossier (to C).

Implementation notes (v1):
- Per-claim entailment is executed in parallel on Modal CPU workers defined in
  modal_verifier.verify_claim_cpu. Local mode runs OpenAI calls directly.
- Outputs a probability-like, uncalibrated viability_score in [0,1]. Stage C
  can calibrate this on the retrospective benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Callable, Iterable

from dotenv import load_dotenv

from schemas import (
  Dossier,
  EvidenceItem,
  EvidencePackage,
  TIER_WEIGHTS,
  Recommendation,
  VerifiedClaim,
  Verdict,
)


# Sidecar audit trail. We keep this out of schemas.py because the contracts are frozen.
_AUDIT_RECORDS: list[dict[str, Any]] = []
_ACTIVE_PAIR_ID: str | None = None


def _reset_audit(pair_id: str | None = None) -> None:
  global _AUDIT_RECORDS, _ACTIVE_PAIR_ID
  _AUDIT_RECORDS = []
  _ACTIVE_PAIR_ID = pair_id


def get_audit_records() -> list[dict[str, Any]]:
  """Return audit records for the most recent verify_and_synthesize() call."""
  return list(_AUDIT_RECORDS)


def _append_audit_record(
  claim_index: int,
  v: VerifiedClaim,
  model_used: str | None,
  reason: str = "",
  supporting_span: str = "",
) -> None:
  rec = {
    "pair_id": _ACTIVE_PAIR_ID,
    "claim_index": claim_index,
    "claim": v.claim,
    "evidence_type": v.evidence_type.value,
    "source_id": v.source_id,
    "source_url": v.source_url,
    "verdict": v.verdict.value,
    "confidence": v.verdict_confidence,
    "reason": reason or _fallback_reason(v),
    "supporting_span": supporting_span or "",
    "source_text_excerpt": (v.source_text or "")[:1200],
    "model_used": model_used,
    "is_negative_evidence": v.is_negative_evidence,
  }
  _AUDIT_RECORDS.append(rec)
  if v.verdict in {Verdict.unsupported, Verdict.contradicted}:
    print(
      f"[audit] rejected claim {claim_index}: verdict={v.verdict.value} "
      f"conf={v.verdict_confidence:.2f} reason={rec['reason'][:180]}",
      flush=True,
    )


def _mark_audit_negative(v: VerifiedClaim) -> None:
  for rec in reversed(_AUDIT_RECORDS):
    if rec.get("claim") == v.claim and rec.get("source_id") == v.source_id:
      rec["is_negative_evidence"] = True
      if "negative" not in str(rec.get("reason", "")).lower():
        rec["reason"] = f"Negative-evidence check: {rec.get('reason', '')}".strip()
      return


def _fallback_reason(v: VerifiedClaim) -> str:
  if v.verdict == Verdict.supported:
    return "The cited passage was judged to clearly support the claim."
  if v.verdict == Verdict.contradicted:
    return "The cited passage was judged to contradict the claim."
  if v.verdict == Verdict.unsupported:
    return "The cited passage was judged not to clearly entail the claim."
  return "The cited passage was judged ambiguous or insufficient."


def audit_records_from_dossier(dossier: Dossier) -> list[dict[str, Any]]:
  """Post-hoc audit reconstruction when a run did not produce --audit-jsonl."""
  out: list[dict[str, Any]] = []
  for i, v in enumerate(dossier.verified_claims, start=1):
    out.append({
      "pair_id": dossier.pair_id,
      "claim_index": i,
      "claim": v.claim,
      "evidence_type": v.evidence_type.value,
      "source_id": v.source_id,
      "source_url": v.source_url,
      "verdict": v.verdict.value,
      "confidence": v.verdict_confidence,
      "reason": _fallback_reason(v),
      "supporting_span": "",
      "source_text_excerpt": (v.source_text or "")[:1200],
      "model_used": dossier.model_used,
      "is_negative_evidence": v.is_negative_evidence,
    })
  return out


# ----------------------------- Config helpers ----------------------------- #
def _get_thresholds() -> tuple[float, float]:
  raw = os.environ.get("RECOMMEND_THRESHOLDS", "0.7,0.4")
  parts = raw.split(",")
  try:
    hi = float(parts[0])
    lo = float(parts[1]) if len(parts) > 1 else 0.4
  except Exception:
    hi, lo = 0.7, 0.4
  return max(0.0, min(1.0, hi)), max(0.0, min(1.0, lo))


def _map_recommendation(score: float) -> Recommendation:
  hi, lo = _get_thresholds()
  if score >= hi:
    return Recommendation.pursue
  if score >= lo:
    return Recommendation.investigate_further
  return Recommendation.deprioritize


def _claim_key(text: str) -> str:
  return " ".join((text or "").lower().split())


def _tokens(text: str) -> set[str]:
  stop = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "with", "by", "on",
    "is", "are", "was", "were", "be", "been", "this", "that", "these", "those",
    "as", "at", "from", "it", "its", "their", "has", "have", "had", "into", "than",
  }
  raw = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
  return {t for t in raw.split() if len(t) > 2 and t not in stop}


def _similarity(a: str, b: str) -> float:
  ta, tb = _tokens(a), _tokens(b)
  if not ta or not tb:
    return 0.0
  return len(ta & tb) / len(ta | tb)


def _polarity(text: str) -> int:
  """Crude polarity for internal conflict detection. +1 positive, -1 negative/caution, 0 neutral."""
  t = f" {text.lower()} "
  negative_markers = [
    " no evidence ", " not associated ", " no association ", " does not ",
    " did not ", " failed ", " failure ", " lack of ", " reduced efficacy ",
    " adverse ", " toxicity ", " risk ", " concern ", " contraindicat",
  ]
  positive_markers = [
    " associated ", " cause ", " causes ", " linked ", " supports ", " reduced ",
    " decreases ", " inhibits ", " improves ", " effective ", " significant ",
  ]
  if any(m in t for m in negative_markers):
    return -1
  if any(m in t for m in positive_markers):
    return 1
  return 0


def _dedupe_verified_for_scoring(verified: list[VerifiedClaim]) -> list[VerifiedClaim]:
  """Avoid duplicated claim/source/verdict rows dominating the score."""
  out: list[VerifiedClaim] = []
  seen: set[tuple[str, str, str, bool]] = set()
  for v in verified:
    key = (_claim_key(v.claim), v.source_id, v.verdict.value, v.is_negative_evidence)
    if key in seen:
      continue
    seen.add(key)
    out.append(v)
  return out


def _score_viability(verified: list[VerifiedClaim]) -> float:
  """Raw viability score using supported evidence minus bounded quality/negative penalties."""
  scored = _dedupe_verified_for_scoring(verified)
  positives = [
    v for v in scored
    if v.verdict == Verdict.supported and not v.is_negative_evidence
  ]
  if not positives:
    base = 0.0
  else:
    num = sum(v.tier_weight * max(0.0, min(1.0, v.verdict_confidence)) for v in positives)
    den = sum(v.tier_weight for v in positives) or 1.0
    base = num / den

  unsupported = [v for v in scored if v.verdict == Verdict.unsupported and not v.is_negative_evidence]
  contradicted = [v for v in scored if v.verdict == Verdict.contradicted and not v.is_negative_evidence]
  negatives = [
    v for v in scored
    if v.is_negative_evidence and v.verdict in {Verdict.supported, Verdict.contradicted}
  ]

  total = max(1, len(scored))
  unsupported_penalty = min(0.12, 0.20 * (len(unsupported) / total))
  contradiction_penalty = min(
    0.20,
    sum(0.06 * v.tier_weight * max(0.0, min(1.0, v.verdict_confidence)) for v in contradicted),
  )
  negative_penalty = min(
    0.30,
    sum(0.08 * max(0.5, v.tier_weight) * max(0.0, min(1.0, v.verdict_confidence)) for v in negatives),
  )
  score = base - unsupported_penalty - contradiction_penalty - negative_penalty
  return round(max(0.0, min(1.0, score)), 3)


def _synthesize_rationale(verified: list[VerifiedClaim]) -> str:
  scored = _dedupe_verified_for_scoring(verified)
  parts: list[str] = []
  top_supported = sorted(
    [v for v in scored if v.verdict == Verdict.supported and not v.is_negative_evidence],
    key=lambda v: (v.tier_weight, v.verdict_confidence),
    reverse=True,
  )[:3]
  if top_supported:
    kinds = ", ".join(sorted({v.evidence_type.value.replace("_", " ") for v in top_supported}))
    examples = "; ".join(f"{v.source_id}: {v.claim[:90]}" for v in top_supported[:2])
    parts.append(f"Verified {kinds} support is present ({examples}).")
  cross_source = [
    v for v in scored
    if v.verdict == Verdict.supported and not v.is_negative_evidence and v.tier_weight <= 0.3
  ]
  if len(cross_source) >= 2:
    parts.append(f"Independent literature added {len(cross_source)} additional corroborating checks.")
  negatives = [v for v in scored if v.is_negative_evidence and v.verdict in {Verdict.supported, Verdict.contradicted}]
  contradicted = [v for v in scored if v.verdict == Verdict.contradicted]
  unsupported = [v for v in scored if v.verdict == Verdict.unsupported]
  if negatives:
    parts.append(f"{len(negatives)} negative or cautionary finding(s) were found and penalized.")
  if contradicted:
    parts.append(f"{len(contradicted)} claim(s) were contradicted by checked sources.")
  if unsupported:
    parts.append(f"{len(unsupported)} unsupported claim(s) were excluded from positive scoring.")
  if not parts:
    parts.append("Limited verifiable evidence within cutoff; recommend further investigation.")
  return " ".join(parts)


def _finalize_stats(verified: list[VerifiedClaim]) -> tuple[int, int, int, dict[str, int], list[str]]:
  supported_count = sum(1 for v in verified if v.verdict == Verdict.supported)
  contradicted_count = sum(1 for v in verified if v.verdict == Verdict.contradicted)
  unsupported_count = sum(1 for v in verified if v.verdict == Verdict.unsupported)
  tb: dict[str, int] = {}
  for v in verified:
    tb[v.evidence_type.value] = tb.get(v.evidence_type.value, 0) + 1
  flags: list[str] = []
  if unsupported_count:
    flags.append("hallucination_caught")
  if any(v.is_negative_evidence for v in verified):
    flags.append("open_safety_question")
  if contradicted_count:
    flags.append("contradiction_found")
  if supported_count:
    flags.append("verified_support_present")
  return supported_count, contradicted_count, unsupported_count, tb, flags


def _detect_internal_consistency(verified: list[VerifiedClaim]) -> list[str]:
  """Approximate pair-level conflict detector among supported claims."""
  supported = [v for v in verified if v.verdict == Verdict.supported]
  for i, a in enumerate(supported):
    pa = _polarity(a.claim)
    if pa == 0:
      continue
    for b in supported[i + 1:]:
      pb = _polarity(b.claim)
      if pb == 0 or pa == pb:
        continue
      if _similarity(a.claim, b.claim) >= 0.35:
        return ["possible_internal_conflict"]
  return []


# ------------------------- Local OpenAI judge (CPU) ----------------------- #
def _judge_entailment_local(source_text: str, claim: str) -> tuple[Verdict, float, str, str, str]:
  from openai import OpenAI
  from tenacity import retry, wait_exponential, stop_after_attempt

  model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
  # Fallback when no API key is present (offline dev): return uncertain, 0.0
  if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_BASE")):
    return Verdict.uncertain, 0.0, f"openai:{model}", "No OpenAI API key configured; skipped live entailment.", ""
  client = OpenAI()

  system = (
    "You are a strict entailment judge. Given a claim and a cited source "
    "passage, decide if the passage SUPPORTS the claim, CONTRADICTS it, "
    "or is UNSUPPORTED; return UNCERTAIN if ambiguous. Be conservative and "
    "do not use outside knowledge beyond the passage."
  )
  user = (
    "Return ONLY a JSON object with keys: "
    "verdict (supported|contradicted|unsupported|uncertain), confidence (0..1), "
    "reason (one concise sentence), and supporting_span (exact quote from the source if supported, else empty string).\n\n"
    f"Claim: {claim}\n\nSource Passage:\n{source_text}"
  )

  @retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(5))
  def _call() -> dict:
    resp = client.responses.create(
      model=model,
      input=[
        {"role": "system", "content": system},
        {"role": "user", "content": user},
      ],
      temperature=float(os.environ.get("OPENAI_TEMPERATURE", 0)),
    )
    txt = resp.output_text or "{}"
    try:
      data = json.loads(txt)
    except json.JSONDecodeError:
      start = txt.find("{")
      end = txt.rfind("}")
      if start != -1 and end != -1 and end > start:
        data = json.loads(txt[start : end + 1])
      else:
        raise
    return data

  data = _call()
  vmap = {
    "supported": Verdict.supported,
    "contradicted": Verdict.contradicted,
    "unsupported": Verdict.unsupported,
    "uncertain": Verdict.uncertain,
  }
  verdict = vmap.get(str(data.get("verdict", "uncertain")).lower(), Verdict.uncertain)
  try:
    conf = float(data.get("confidence", 0.0))
  except Exception:
    conf = 0.0
  conf = max(0.0, min(1.0, conf))
  reason = str(data.get("reason", "")).strip()
  supporting_span = str(data.get("supporting_span", "")).strip()
  return verdict, conf, f"openai:{model}", reason, supporting_span


def _verify_claim_local(item: EvidenceItem) -> tuple[VerifiedClaim, str, str, str]:
  if not item.source_text:
    verdict, conf = Verdict.unsupported, 0.0
    model_used = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    reason = "No source_text was provided, so the claim cannot be verified."
    supporting_span = ""
  else:
    judged = _judge_entailment_local(item.source_text, item.claim)
    verdict, conf, model_used = judged[:3]
    reason = judged[3] if len(judged) > 3 else ""
    supporting_span = judged[4] if len(judged) > 4 else ""
  v = VerifiedClaim(
    claim=item.claim,
    evidence_type=item.evidence_type,
    verdict=verdict,
    verdict_confidence=conf,
    tier_weight=TIER_WEIGHTS[item.evidence_type],
    is_negative_evidence=False,
    source_db=item.source_db,
    source_id=item.source_id,
    source_url=item.source_url,
    source_text=item.source_text or "",
    publication_date=item.publication_date,
  )
  return v, model_used, reason, supporting_span


# ------------------------ Modal per-claim orchestration ------------------- #
def _verify_claims_modal(items: Iterable[EvidenceItem]) -> tuple[list[VerifiedClaim], str]:
  # Late import to avoid mandatory Modal dependency for local runs
  from modal_verifier import app as modal_app, verify_claim_cpu  # type: ignore

  items = list(items)
  print(f"[modal] spawning {len(items)} claim jobs...", flush=True)
  verified: list[VerifiedClaim] = []
  model_used = None
  with modal_app.run():
    jobs = [verify_claim_cpu.spawn(it.model_dump(mode="json")) for it in items]
    for i, job in enumerate(jobs, start=1):
      print(f"[modal] waiting for job {i}/{len(jobs)}...", flush=True)
      out = job.get()
      mdl = out.pop("_model_used", None)
      reason = out.pop("_audit_reason", "")
      supporting_span = out.pop("_audit_supporting_span", "")
      if mdl and not model_used:
        model_used = mdl
      v = VerifiedClaim.model_validate(out)
      print(f"[modal] result {i}/{len(jobs)}: {v.verdict.value} conf={v.verdict_confidence:.2f} tier={v.tier_weight}", flush=True)
      _append_audit_record(i, v, mdl or model_used, reason, supporting_span)
      verified.append(v)
  print(f"[modal] collected {len(verified)} verified claims", flush=True)
  return verified, (model_used or "openai:unknown")


def _verify_claims_local(items: Iterable[EvidenceItem]) -> tuple[list[VerifiedClaim], str]:
  items = list(items)
  verified: list[VerifiedClaim] = []
  model_used = None
  for i, it in enumerate(items, start=1):
    print(f"[local] verifying claim {i}/{len(items)}: {it.claim[:120]}", flush=True)
    v, mdl, reason, supporting_span = _verify_claim_local(it)
    print(f"[local] -> {v.verdict.value} conf={v.verdict_confidence:.2f} tier={v.tier_weight}", flush=True)
    if mdl and not model_used:
      model_used = mdl
    _append_audit_record(i, v, mdl, reason, supporting_span)
    verified.append(v)
  print(f"[local] verified {len(verified)} claims", flush=True)
  return verified, (model_used or "openai:unknown")


def _call_search(search: Callable[[str, object], list[EvidenceItem]], query: str,
                 cutoff_date: object, limit: int) -> list[EvidenceItem]:
  """Call A.search while tolerating either 2-arg or 3-arg signatures."""
  try:
    return search(query, cutoff_date, limit=limit) or []  # type: ignore[misc]
  except TypeError:
    return search(query, cutoff_date) or []


def _verify_items_by_mode(items: list[EvidenceItem], mode: str) -> list[VerifiedClaim]:
  if not items:
    return []
  if mode == "modal":
    out, _ = _verify_claims_modal(items)
  else:
    out, _ = _verify_claims_local(items)
  return out


def _claim_terms(claim: str, max_terms: int = 8) -> str:
  toks = sorted(_tokens(claim), key=len, reverse=True)
  return " ".join(toks[:max_terms])


def _disease_aliases(disease_name: str) -> set[str]:
  d = (disease_name or "").lower().strip()
  aliases = {d}
  if "emia" in d:
    aliases.add(d.replace("emia", "aemia"))
  if "aemia" in d:
    aliases.add(d.replace("aemia", "emia"))
  if d == "hypercholesterolemia":
    aliases.update({"hypercholesterolaemia", "familial hypercholesterolemia", "familial hypercholesterolaemia", "adh"})
  return {a for a in aliases if a}


def _mentions_current_pair(text: str, target_symbol: str, disease_name: str) -> bool:
  t = (text or "").lower()
  target_ok = (target_symbol or "").lower() in t
  disease_ok = any(alias in t for alias in _disease_aliases(disease_name))
  return target_ok and disease_ok


def _make_claim_against_source(original: VerifiedClaim, source_item: EvidenceItem) -> EvidenceItem:
  """Check the original claim against an independent source passage."""
  return EvidenceItem(
    claim=original.claim,
    evidence_type=source_item.evidence_type,
    source_db=source_item.source_db,
    source_id=source_item.source_id,
    source_text=source_item.source_text,
    source_url=source_item.source_url,
    publication_date=source_item.publication_date,
    retrieval_score=source_item.retrieval_score,
    metadata=source_item.metadata,
  )


def _cross_source_corroboration(
  pkg: EvidencePackage,
  verified: list[VerifiedClaim],
  mode: str,
  search: Callable[[str, object], list[EvidenceItem]] | None,
) -> list[VerifiedClaim]:
  """B3: verify high-tier supported claims against independent literature passages."""
  if search is None:
    return []

  max_claims = int(os.environ.get("CORROBORATION_MAX_CLAIMS", "5"))
  per_claim_limit = int(os.environ.get("CORROBORATION_SOURCES_PER_CLAIM", "3"))
  min_tier = float(os.environ.get("CORROBORATION_MIN_TIER", "0.6"))

  candidates = [
    v for v in verified
    if v.verdict == Verdict.supported
    and not v.is_negative_evidence
    and v.tier_weight >= min_tier
  ]
  candidates = sorted(candidates, key=lambda v: (v.tier_weight, v.verdict_confidence), reverse=True)[:max_claims]

  all_checks: list[EvidenceItem] = []
  seen_source_claims: set[tuple[str, str]] = set()

  print(f"[corroboration] checking {len(candidates)} high-tier supported claims", flush=True)
  for v in candidates:
    q = f"{pkg.target_symbol} {pkg.disease_name} {_claim_terms(v.claim)}"
    print(f"[corroboration] query={q!r}", flush=True)
    hits = _call_search(search, q, pkg.cutoff_date, limit=per_claim_limit + 2)
    independent = [
      h for h in hits
      if h.source_id != v.source_id
      and h.source_text
      and _mentions_current_pair(h.source_text, pkg.target_symbol, pkg.disease_name)
      and (h.publication_date is None or h.publication_date <= pkg.cutoff_date)
    ][:per_claim_limit]
    print(f"[corroboration] found {len(independent)} independent candidates", flush=True)

    for h in independent:
      key = (_claim_key(v.claim), h.source_id)
      if key in seen_source_claims:
        continue
      seen_source_claims.add(key)
      all_checks.append(_make_claim_against_source(v, h))

  checked = _verify_items_by_mode(all_checks, mode)
  for v in checked:
    if v.verdict == Verdict.contradicted:
      v.is_negative_evidence = True
      _mark_audit_negative(v)
  print(f"[corroboration] added {len(checked)} independent verification checks", flush=True)
  return checked


def _negative_claim(pkg: EvidencePackage, source_item: EvidenceItem) -> EvidenceItem:
  """Turn a retrieved candidate passage into a checkable negative-evidence proposition."""
  claim = (
    f"This passage provides cautionary, contradictory, failed, safety, or negative evidence "
    f"against pursuing {pkg.target_symbol} for {pkg.disease_name}."
  )
  return EvidenceItem(
    claim=claim,
    evidence_type=source_item.evidence_type,
    source_db=source_item.source_db,
    source_id=source_item.source_id,
    source_text=source_item.source_text,
    source_url=source_item.source_url,
    publication_date=source_item.publication_date,
    retrieval_score=source_item.retrieval_score,
    metadata=source_item.metadata,
  )


def _contradiction_hunt(pkg: EvidencePackage, mode: str, search: Callable[[str, object], list[EvidenceItem]] | None) -> list[VerifiedClaim]:
  if search is None:
    return []

  per_query_limit = int(os.environ.get("NEGATIVE_SEARCH_LIMIT", "3"))
  queries = [
    f"{pkg.target_symbol} {pkg.disease_name} failure",
    f"{pkg.target_symbol} {pkg.disease_name} adverse effects",
    f"{pkg.target_symbol} {pkg.disease_name} safety concern",
    f"{pkg.target_symbol} {pkg.disease_name} failed trial",
    f"{pkg.target_symbol} {pkg.disease_name} failed to replicate",
    f"{pkg.target_symbol} knockout phenotype {pkg.disease_name}",
  ]

  candidates: list[EvidenceItem] = []
  seen_sources: set[str] = set()
  for q in queries:
    print(f"[contradiction_hunt] query={q!r} cutoff={pkg.cutoff_date}", flush=True)
    hits = _call_search(search, q, pkg.cutoff_date, limit=per_query_limit)
    for h in hits:
      if not h.source_text or h.source_id in seen_sources:
        continue
      if not _mentions_current_pair(h.source_text, pkg.target_symbol, pkg.disease_name):
        continue
      if h.publication_date is not None and h.publication_date > pkg.cutoff_date:
        continue
      seen_sources.add(h.source_id)
      candidates.append(_negative_claim(pkg, h))

  print(f"[contradiction_hunt] classifying {len(candidates)} candidate negative passages", flush=True)
  checked = _verify_items_by_mode(candidates, mode)

  negs: list[VerifiedClaim] = []
  for v in checked:
    if v.verdict != Verdict.supported:
      continue
    v.is_negative_evidence = True
    _mark_audit_negative(v)
    negs.append(v)
  print(f"[contradiction_hunt] retained {len(negs)} supported negative-evidence claims", flush=True)
  return negs


def verify_and_synthesize(pkg: EvidencePackage, *, mode: str = "modal",
              search: Callable[[str, object], list[EvidenceItem]] | None = None) -> Dossier:
  """
  Verifies each claim in the EvidencePackage (OpenAI entailment) and returns a Dossier.
  mode: 'modal' (default) spawns per-claim workers on Modal; 'local' runs in-process.
  search: optional A.search(query, cutoff_date) for contradiction/negative-evidence hunt.
  """
  t0 = time.time()
  _reset_audit(pkg.pair_id)

  items = list(pkg.evidence_items)
  print(f"[verify] starting verification for pair={pkg.pair_id} target={pkg.target_symbol} disease={pkg.disease_name} items={len(items)} mode={mode}", flush=True)
  if not items:
    # Empty dossier skeleton
    empty = Dossier(
      pair_id=pkg.pair_id,
      target_symbol=pkg.target_symbol,
      target_id=pkg.target_id,
      disease_name=pkg.disease_name,
      disease_id=pkg.disease_id,
      cutoff_date=pkg.cutoff_date,
      verified_claims=[],
      viability_score=0.0,
      recommendation=_map_recommendation(0.0),
      rationale="No verifiable evidence within cutoff.",
    )
    empty.supported_count = 0
    empty.contradicted_count = 0
    empty.unsupported_count = 0
    empty.tier_breakdown = {}
    empty.flags = ["insufficient_evidence"]
    empty.model_used = None
    empty.runtime_seconds = round(time.time() - t0, 3)
    return empty

  # Load env for local secret access
  load_dotenv()

  if mode == "modal":
    verified, model_used = _verify_claims_modal(items)
  elif mode == "local":
    verified, model_used = _verify_claims_local(items)
  else:
    raise ValueError("mode must be 'modal' or 'local'")

  # B3: add independent literature checks for high-tier supported claims if A.search is available.
  if search is not None:
    verified.extend(_cross_source_corroboration(pkg, verified, mode, search))

  # B4: add active negative/cautionary evidence if A.search is available.
  verified.extend(_contradiction_hunt(pkg, mode, search))

  score = _score_viability(verified)
  rec = _map_recommendation(score)
  rationale = _synthesize_rationale(verified)
  scount, ccount, ucount, tbreak, flags = _finalize_stats(verified)
  flags = sorted(set(flags + _detect_internal_consistency(verified)))

  dossier = Dossier(
    pair_id=pkg.pair_id,
    target_symbol=pkg.target_symbol,
    target_id=pkg.target_id,
    disease_name=pkg.disease_name,
    disease_id=pkg.disease_id,
    cutoff_date=pkg.cutoff_date,
    verified_claims=verified,
    viability_score=score,
    recommendation=rec,
    rationale=rationale,
    supported_count=scount,
    contradicted_count=ccount,
    unsupported_count=ucount,
    tier_breakdown=tbreak,
    flags=flags,
    model_used=model_used,
    runtime_seconds=round(time.time() - t0, 3),
  )
  return dossier


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--in", dest="in_path", help="Path to EvidencePackage JSON")
  ap.add_argument("--out", dest="out_path", help="Path to write Dossier JSON", default=None)
  ap.add_argument("--mode", choices=["local", "modal"], default="modal")
  ap.add_argument("--no-contradictions", action="store_true", help="Disable contradiction hunt")
  # Batch JSONL options
  ap.add_argument("--jsonl", dest="jsonl_path", help="Path to JSONL of EvidencePackage objects")
  ap.add_argument("--disease-name", dest="disease_name", help="Filter: disease_name contains this substring (case-insensitive)")
  ap.add_argument("--disease-id", dest="disease_id", help="Filter: exact disease_id (e.g., EFO_0004911)")
  ap.add_argument("--limit", type=int, default=5, help="Max number of packages to process from JSONL")
  ap.add_argument("--out-jsonl", dest="out_jsonl", help="Write Dossier results JSONL to this path")
  ap.add_argument("--audit-jsonl", dest="audit_jsonl", help="Write per-claim verifier audit records JSONL to this path")
  a = ap.parse_args()

  def _read_pkg(path: str) -> EvidencePackage:
    with open(path, "r") as f:
      data = json.load(f)
    return EvidencePackage.model_validate(data)

  def _search_fn(enabled: bool):
    if not enabled:
      return None
    try:
      from engine import search as engine_search  # Person A's cutoff-safe Europe PMC search
      return engine_search
    except Exception as e:
      print(f"[warn] could not import engine.search; B3/B4 search disabled: {type(e).__name__}: {e}", flush=True)
      return None

  if a.in_path and not a.jsonl_path:
    pkg = _read_pkg(a.in_path)
    dossier = verify_and_synthesize(pkg, mode=a.mode, search=_search_fn(not a.no_contradictions))
    out = dossier.model_dump(mode="json")
    if a.out_path:
      with open(a.out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    if a.audit_jsonl:
      with open(a.audit_jsonl, "w") as f:
        for rec in get_audit_records() or audit_records_from_dossier(dossier):
          f.write(json.dumps(rec, default=str) + "\n")
  elif a.jsonl_path:
    # Batch mode: read JSONL of EvidencePackages, filter, run small batch, print/emit scores
    pkgs: list[EvidencePackage] = []
    with open(a.jsonl_path, "r") as f:
      for line in f:
        line = line.strip()
        if not line:
          continue
        try:
          data = json.loads(line)
          pkg = EvidencePackage.model_validate(data)
        except Exception:
          continue
        if a.disease_id and pkg.disease_id != a.disease_id:
          continue
        if a.disease_name and (a.disease_name.lower() not in pkg.disease_name.lower()):
          continue
        pkgs.append(pkg)
        if len(pkgs) >= max(1, a.limit):
          break

    dossiers: list[Dossier] = []
    audit_records: list[dict[str, Any]] = []
    for pkg in pkgs:
      dossier = verify_and_synthesize(pkg, mode=a.mode, search=_search_fn(not a.no_contradictions))
      dossiers.append(dossier)
      audit_records.extend(get_audit_records() or audit_records_from_dossier(dossier))

    # Emit JSONL if requested
    if a.out_jsonl:
      with open(a.out_jsonl, "w") as out_f:
        for d in dossiers:
          out_f.write(json.dumps(d.model_dump(mode="json"), default=str) + "\n")

    if a.audit_jsonl:
      with open(a.audit_jsonl, "w") as audit_f:
        for rec in audit_records:
          audit_f.write(json.dumps(rec, default=str) + "\n")

    # Always print a concise summary to stdout
    for d in dossiers:
      print(json.dumps({
        "pair_id": d.pair_id,
        "target_symbol": d.target_symbol,
        "disease_name": d.disease_name,
        "viability_score": d.viability_score,
        "recommendation": d.recommendation.value,
        "supported": d.supported_count,
        "unsupported": d.unsupported_count,
      }))

    # Exit after batch processing
    raise SystemExit(0)
  else:
    # Smoke test using the mock engine
    from workflow_example import run_engine, search  # type: ignore
    from datetime import date
    pkg = run_engine("ENSG00000169174", "EFO_0004911", date(2010, 1, 1))
    dossier = verify_and_synthesize(pkg, mode=a.mode, search=None if a.no_contradictions else search)
    out = dossier.model_dump(mode="json")
    if a.out_path:
      with open(a.out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    if a.audit_jsonl:
      with open(a.audit_jsonl, "w") as f:
        for rec in get_audit_records() or audit_records_from_dossier(dossier):
          f.write(json.dumps(rec, default=str) + "\n")
