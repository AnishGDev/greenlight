"""
modal_verifier.py — CPU-only per-claim verifier worker for Modal.

Takes one EvidenceItem (as JSON), runs an OpenAI entailment judge against
its cited source text, and returns a VerifiedClaim (as JSON). Designed to be
spawned in parallel per claim by verifier.py.

Secrets: requires Modal secret named "openai-secret" that provides OPENAI_API_KEY.
"""

from __future__ import annotations

import json
import os
from typing import Literal

import modal


app = modal.App("greenlight-verifier-cpu")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "openai>=1.30.0",
        "pydantic>=2",
        "tenacity>=8",
    )
)


def _get_openai_model() -> str:
    # Safe, configurable default; override via env OPENAI_MODEL
    return os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


def _judge_entailment(source_text: str, claim: str) -> tuple[str, float, str, str, str]:
    """
    Calls OpenAI Responses API to judge whether the source_text supports the claim.

    Returns: (verdict_str, confidence_float, model_used, reason, supporting_span)
    verdict_str in {supported, contradicted, unsupported, uncertain}
    confidence_float in [0,1]
    """
    from tenacity import retry, wait_exponential, stop_after_attempt
    from openai import OpenAI

    model = _get_openai_model()
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])  # Modal secret

    system = (
        "You are a strict entailment judge. Given a claim and a cited source "
        "passage, decide if the passage SUPPORTS the claim, CONTRADICTS it, "
        "or does not provide enough evidence (UNSUPPORTED). If ambiguous, mark UNCERTAIN. "
        "Be conservative and do not use outside knowledge beyond the passage."
    )
    user = (
        "Return ONLY a JSON object with keys: verdict (supported|contradicted|unsupported|uncertain) "
        "confidence (0..1), reason (one concise sentence), and supporting_span "
        "(exact quote from the source if supported, else empty string). "
        "Be conservative; do not infer beyond the passage.\n\n"
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
        # Expecting a compact JSON string in output_text
        txt = resp.output_text or "{}"
        try:
            data = json.loads(txt)
        except json.JSONDecodeError:
            # Fallback: attempt to extract JSON object substring
            start = txt.find("{")
            end = txt.rfind("}")
            if start != -1 and end != -1 and end > start:
                data = json.loads(txt[start : end + 1])
            else:
                raise
        return {"data": data, "model": model}

    out = _call()
    data = out["data"]
    model_used = out["model"]
    verdict = str(data.get("verdict", "uncertain")).lower()
    if verdict not in {"supported", "contradicted", "unsupported", "uncertain"}:
        verdict = "uncertain"
    try:
        conf = float(data.get("confidence", 0.0))
    except Exception:
        conf = 0.0
    conf = max(0.0, min(1.0, conf))
    reason = str(data.get("reason", "")).strip()
    supporting_span = str(data.get("supporting_span", "")).strip()
    return verdict, conf, f"openai:{model_used}", reason, supporting_span


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("openai-secret")],
    timeout=120,
)
def verify_claim_cpu(item_json: dict) -> dict:
    """
    Input: EvidenceItem as JSON (schemas.EvidenceItem.model_dump(mode="json")).
    Output: VerifiedClaim as JSON (schemas.VerifiedClaim.model_dump(mode="json")).
    """
    # Local import to keep worker light
    from schemas import EvidenceItem, VerifiedClaim, TIER_WEIGHTS, Verdict

    it = EvidenceItem.model_validate(item_json)
    # Guard: need source_text to judge entailment; if missing -> unsupported low confidence
    if not it.source_text:
        verdict_str, conf, model_used = "unsupported", 0.0, _get_openai_model()
        reason = "No source_text was provided, so the claim cannot be verified."
        supporting_span = ""
    else:
        verdict_str, conf, model_used, reason, supporting_span = _judge_entailment(it.source_text, it.claim)

    verdict_map: dict[str, Verdict] = {
        "supported": Verdict.supported,
        "contradicted": Verdict.contradicted,
        "unsupported": Verdict.unsupported,
        "uncertain": Verdict.uncertain,
    }
    v = VerifiedClaim(
        claim=it.claim,
        evidence_type=it.evidence_type,
        verdict=verdict_map[verdict_str],
        verdict_confidence=conf,
        tier_weight=TIER_WEIGHTS[it.evidence_type],
        is_negative_evidence=False,
        source_db=it.source_db,
        source_id=it.source_id,
        source_url=it.source_url,
        source_text=it.source_text or "",
        publication_date=it.publication_date,
    )
    out = v.model_dump(mode="json")
    out["_model_used"] = model_used
    out["_audit_reason"] = reason
    out["_audit_supporting_span"] = supporting_span
    return out


@app.local_entrypoint()
def main():  # pragma: no cover - local smoke
    print("greenlight-verifier-cpu Modal app ready.")
