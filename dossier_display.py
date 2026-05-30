"""
GreenLight — Dossier display component for Streamlit demo.

Renders Dossier outputs with:
- Top-level summary: viability_score, recommendation, rationale
- Expandable details: supported/contradicted/unsupported counts, tier breakdown, flags
"""

import streamlit as st

from schemas import Dossier, VerifiedClaim


VERDICT_STYLE = {
    "supported": ("✅", "success"),
    "contradicted": ("❌", "error"),
    "unsupported": ("⚠️", "warning"),
    "uncertain": ("❔", "info"),
}


def _render_verdict_badge(claim: VerifiedClaim) -> None:
    verdict = claim.verdict.value
    icon, tone = VERDICT_STYLE.get(verdict, ("❔", "info"))
    text = f"{icon} Verdict: {verdict.title()} | Confidence: {claim.verdict_confidence:.2f}"
    if tone == "success":
        st.success(text)
    elif tone == "error":
        st.error(text)
    elif tone == "warning":
        st.warning(text)
    else:
        st.info(text)


def display_verified_claims(claims: list[VerifiedClaim], key_suffix: str = "") -> None:
    """
    Display claim-level explanation for each verified claim in a dossier.

    Args:
        claims: Verified claims from the dossier
        key_suffix: Optional suffix for Streamlit element key uniqueness
    """
    st.markdown("**Claim-Level Interpretation**")
    if not claims:
        st.caption("(no verified claims)")
        return

    for idx, claim in enumerate(claims, 1):
        header = (
            f"Claim {idx} | {claim.evidence_type.value} | "
            f"{claim.verdict.value.title()} | Source: {claim.source_db.value}:{claim.source_id}"
        )
        with st.expander(header, expanded=False):
            _render_verdict_badge(claim)

            if claim.is_negative_evidence:
                st.warning("Negative evidence marker: this claim contributes risk signal.")

            st.markdown("**Claim Text**")
            st.write(claim.claim)

            left, right = st.columns(2)
            with left:
                st.caption(f"**Evidence Type:** {claim.evidence_type.value}")
                st.caption(f"**Tier Weight:** {claim.tier_weight:.2f}")
                st.caption(f"**Source DB:** {claim.source_db.value}")
                st.caption(f"**Source ID:** {claim.source_id}")
            with right:
                if claim.publication_date:
                    st.caption(f"**Publication Date:** {claim.publication_date}")
                if claim.source_url:
                    st.caption(f"**Source URL:** {claim.source_url}")

            st.markdown("**Source Text**")
            st.code(claim.source_text)


def display_dossier_summary(dossier: Dossier, key_suffix: str = "") -> None:
    """
    Displays the dossier with a collapsible detailed metrics section.
    
    Args:
        dossier: The Dossier object to display
        key_suffix: Optional suffix for Streamlit keys (for uniqueness when displaying multiple)
    """
    
    # ─────────────────────────────────────────────────────────────────────────
    # TOP-LEVEL: viability score, recommendation, rationale
    # ─────────────────────────────────────────────────────────────────────────
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    # Viability score with color coding
    with col1:
        score = dossier.viability_score
        if score >= 0.7:
            color = "🟢"
            score_color = "green"
        elif score >= 0.4:
            color = "🟡"
            score_color = "orange"
        else:
            color = "🔴"
            score_color = "red"
        
        st.metric("Viability Score", f"{score:.2f}", delta=None)
        st.caption(f"{color} {['Deprioritize', 'Investigate Further', 'Pursue'][min(2, int(score * 3))]}")
    
    # Recommendation
    with col2:
        recommendation_text = dossier.recommendation.value.replace("_", " ").title()
        st.metric("Recommendation", recommendation_text)
    
    # Target-Disease pair
    with col3:
        st.subheader(f"{dossier.target_symbol} → {dossier.disease_name}")
        st.caption(f"IDs: {dossier.target_id} / {dossier.disease_id}")
    
    # Rationale
    st.markdown("**Rationale**")
    st.write(dossier.rationale)
    
    st.divider()
    
    # ─────────────────────────────────────────────────────────────────────────
    # EXPANDABLE DETAILS
    # ─────────────────────────────────────────────────────────────────────────
    
    with st.expander("📊 Detailed Metrics", expanded=False):
        # Evidence counts
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Supported", dossier.supported_count)
        with col2:
            st.metric("Contradicted", dossier.contradicted_count)
        with col3:
            st.metric("Unsupported", dossier.unsupported_count)
        with col4:
            total = dossier.supported_count + dossier.contradicted_count + dossier.unsupported_count
            st.metric("Total Claims", total)
        
        st.markdown("**Evidence Tier Breakdown**")
        if dossier.tier_breakdown:
            # Display as a nice table
            tier_data = []
            for tier_type, count in sorted(dossier.tier_breakdown.items()):
                tier_data.append({"Evidence Type": tier_type, "Count": count})
            st.dataframe(tier_data, width='stretch', hide_index=True)
        else:
            st.caption("(no tier breakdown available)")
        
        # Flags
        if dossier.flags:
            st.markdown("**Flags**")
            for flag in dossier.flags:
                st.warning(f"🚩 {flag}")
        else:
            st.markdown("**Flags**")
            st.caption("(no flags)")
        
        # Additional metadata
        col1, col2 = st.columns(2)
        with col1:
            st.caption(f"**Cutoff Date:** {dossier.cutoff_date}")
        with col2:
            if dossier.model_used:
                st.caption(f"**Model:** {dossier.model_used}")
        
        if dossier.runtime_seconds > 0:
            st.caption(f"**Runtime:** {dossier.runtime_seconds:.2f}s")

    display_verified_claims(dossier.verified_claims, key_suffix=key_suffix)


def display_dossier_compact(dossier: Dossier) -> str:
    """
    Returns a compact string representation of the dossier (for logging/export).
    
    Args:
        dossier: The Dossier object
        
    Returns:
        A formatted string summary
    """
    return f"""
Dossier: {dossier.target_symbol} → {dossier.disease_name}
═══════════════════════════════════════════════════════════════
Viability Score:  {dossier.viability_score:.2f}
Recommendation:   {dossier.recommendation.value}
Rationale:        {dossier.rationale}
───────────────────────────────────────────────────────────────
Supported:        {dossier.supported_count}
Contradicted:     {dossier.contradicted_count}
Unsupported:      {dossier.unsupported_count}
Tier Breakdown:   {dossier.tier_breakdown}
Flags:            {', '.join(dossier.flags) if dossier.flags else 'None'}
""".strip()


def display_multiple_dossiers(dossiers: list[Dossier], sort_by: str = "viability_score") -> None:
    """
    Displays multiple dossiers in a ranked table with expandable detail rows.
    
    Args:
        dossiers: List of Dossier objects
        sort_by: Field to sort by ("viability_score", "target_symbol", "disease_name")
    """
    
    if not dossiers:
        st.warning("No dossiers to display.")
        return
    
    # Sort dossiers
    if sort_by == "viability_score":
        sorted_dossiers = sorted(dossiers, key=lambda d: d.viability_score, reverse=True)
    elif sort_by == "target_symbol":
        sorted_dossiers = sorted(dossiers, key=lambda d: d.target_symbol)
    else:
        sorted_dossiers = sorted(dossiers, key=lambda d: d.disease_name)
    
    st.markdown(f"**Ranked Results** ({len(sorted_dossiers)} pairs)")
    
    for idx, dossier in enumerate(sorted_dossiers, 1):
        with st.expander(
            f"#{idx} | {dossier.target_symbol:10s} | {dossier.disease_name:20s} | "
            f"Score: {dossier.viability_score:.2f}",
            key=f"dossier_{idx}_{dossier.pair_id}"
        ):
            display_dossier_summary(dossier, key_suffix=f"_{idx}")


# ─────────────────────────────────────────────────────────────────────────────
# Example usage / smoke test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from datetime import date
    from schemas import EvidenceType, Recommendation, SourceDB, Verdict
    
    # Create a mock dossier for testing
    mock_dossier = Dossier(
        pair_id="ENSG00000169174::EFO_0004911",
        target_symbol="PCSK9",
        target_id="ENSG00000169174",
        disease_name="Hypercholesterolemia",
        disease_id="EFO_0004911",
        cutoff_date=date(2020, 1, 1),
        verified_claims=[
            VerifiedClaim(
                claim="PCSK9 loss-of-function variants lower LDL cholesterol.",
                evidence_type=EvidenceType.genetic,
                verdict=Verdict.supported,
                verdict_confidence=0.94,
                tier_weight=1.0,
                source_db=SourceDB.pubmed,
                source_id="16554528",
                source_text="Nonsense mutations in PCSK9...",
            )
        ],
        viability_score=0.87,
        recommendation=Recommendation.pursue,
        rationale="Strong human genetic support; LoF lowers LDL.",
        supported_count=3,
        contradicted_count=0,
        unsupported_count=1,
        tier_breakdown={"genetic": 2, "clinical": 1},
        flags=["high_confidence", "validated_in_clinic"],
        model_used="gpt-4",
        runtime_seconds=12.5,
    )
    
    st.set_page_config(layout="wide")
    st.title("Dossier Display Test")
    
    st.subheader("Single Dossier View")
    display_dossier_summary(mock_dossier)
    
    st.subheader("Compact Output")
    st.code(display_dossier_compact(mock_dossier))
    
    st.subheader("Multiple Dossiers")
    display_multiple_dossiers([mock_dossier, mock_dossier])
