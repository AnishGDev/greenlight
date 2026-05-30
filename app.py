"""Mock interface for creating a BenchmarkPair to send into the GreenLight pipeline.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

from datetime import date

import streamlit as st
from schemas import (
    BenchmarkPair,
    Split,
    make_pair_id,
    Dossier,
    VerifiedClaim,
    Verdict,
    EvidenceType,
    SourceDB,
    Recommendation,
)
from registries import resolve_disease_id, resolve_target_id
from dossier_display import display_dossier_summary, display_dossier_compact


def build_benchmark_pair(
    target_symbol: str,
    target_id: str,
    disease_name: str,
    disease_id: str,
) -> BenchmarkPair:
    pair_id = make_pair_id(target_id.strip(), disease_id.strip())
    return BenchmarkPair(
        pair_id=pair_id,
        target_symbol=target_symbol.strip(),
        target_id=target_id.strip(),
        disease_name=disease_name.strip(),
        disease_id=disease_id.strip(),
    )


def build_mock_dossier(
    target_symbol: str,
    target_id: str,
    disease_name: str,
    disease_id: str,
    cutoff_date: date,
) -> Dossier:
    """Create a mock dossier for demonstration purposes."""
    pair_id = make_pair_id(target_id.strip(), disease_id.strip())
    
    # Create sample verified claim
    vclaim = VerifiedClaim(
        claim=f"{target_symbol} loss-of-function variants reduce disease severity in {disease_name}.",
        evidence_type=EvidenceType.genetic,
        verdict=Verdict.supported,
        verdict_confidence=0.92,
        tier_weight=1.0,
        source_db=SourceDB.pubmed,
        source_id="mock-12345",
        source_text="A study showing that genetic variants in the target gene are associated with disease outcomes...",
        publication_date=date(2018, 6, 15),
    )
    
    return Dossier(
        pair_id=pair_id,
        target_symbol=target_symbol.strip(),
        target_id=target_id.strip(),
        disease_name=disease_name.strip(),
        disease_id=disease_id.strip(),
        cutoff_date=cutoff_date,
        verified_claims=[vclaim],
        viability_score=0.78,
        recommendation=Recommendation.pursue,
        rationale=f"Strong genetic evidence for {target_symbol} in {disease_name}. Multiple studies show LoF variants improve outcomes. Well-druggable protein family.",
        supported_count=5,
        contradicted_count=0,
        unsupported_count=1,
        tier_breakdown={"genetic": 3, "clinical": 2},
        flags=["high_confidence"],
        model_used="gpt-4",
        runtime_seconds=8.3,
    )



def main() -> None:
    st.set_page_config(page_title="GreenLight BenchmarkPair Builder", page_icon="🧬")
    st.title("GreenLight BenchmarkPair Mock Interface")

    st.markdown(
        "Enter only a target symbol and a disease name. IDs are resolved behind the scenes and shown in grey. "
        "Use the cutoff switch to choose a 2010 retrospective cutoff or include all data up to today."
    )

    with st.form("benchmark_pair_form"):
        target_col, target_id_col = st.columns([3, 2])
        target_symbol = target_col.text_input("Target symbol", value="PCSK9")
        target_id, target_suggestions = resolve_target_id(target_symbol)
        target_id_col.text_input("Resolved target ID", value=target_id, disabled=True)

        disease_col, disease_id_col = st.columns([3, 2])
        disease_name = disease_col.text_input("Disease name", value="hypercholesterolemia")
        disease_id, disease_suggestions = resolve_disease_id(disease_name)
        disease_id_col.text_input("Resolved disease ID", value=disease_id, disabled=True)

        if target_id == "unknown" and target_suggestions:
            target_col.caption(
                "Target lookup failed. Did you mean: "
                + ", ".join(target_suggestions)
                + "?"
            )
        elif target_id == "unknown":
            target_col.caption("Target lookup failed. Please enter a standard symbol or gene name.")

        if disease_id == "unknown" and disease_suggestions:
            disease_col.caption(
                "Disease lookup failed. Did you mean: "
                + ", ".join(disease_suggestions)
                + "?"
            )
        elif disease_id == "unknown":
            disease_col.caption("Disease lookup failed. Please enter a standard disease name.")

        use_2010_cutoff = st.checkbox("Cutoff at 2010 (hide future data after 2010)", value=True)
        cutoff_date = date(2010, 1, 1) if use_2010_cutoff else date.today()

        submitted = st.form_submit_button("Create BenchmarkPair")

    if submitted:
        if target_id == "unknown" or disease_id == "unknown":
            warning_text = "One or both IDs could not be resolved automatically. Please verify the target symbol and disease name."
            if target_suggestions:
                warning_text += " Target suggestions: " + ", ".join(target_suggestions) + "."
            if disease_suggestions:
                warning_text += " Disease suggestions: " + ", ".join(disease_suggestions) + "."
            st.warning(warning_text)

        try:
            benchmark_pair = build_benchmark_pair(
                target_symbol,
                target_id,
                disease_name,
                disease_id,
            )
            st.success("BenchmarkPair created successfully")
            st.write("### Generated BenchmarkPair")
            st.write(f"**pair_id:** `{benchmark_pair.pair_id}`")
            st.json(benchmark_pair.model_dump(mode="json"))

            st.markdown(
                "---\n"
                "The hidden benchmark fields are populated automatically for this mock interface. "
                "You can extend this app later to collect labels and validation metadata."
            )
        except Exception as exc:
            st.error(f"Error creating BenchmarkPair: {exc}")

    st.divider()

    with st.expander("📋 View Sample Dossier Output (from Person B's Verifier)", expanded=False):
        st.markdown(
            "This shows what the Dossier output from Person B (the Verifier) will look like. "
            "The interface displays the key metrics up top with detailed evidence metrics expandable below."
        )
        try:
            mock_dossier = build_mock_dossier(
                target_symbol if target_id != "unknown" else "PCSK9",
                target_id if target_id != "unknown" else "ENSG00000169174",
                disease_name if disease_id != "unknown" else "Hypercholesterolemia",
                disease_id if disease_id != "unknown" else "EFO_0004911",
                cutoff_date,
            )
            display_dossier_summary(mock_dossier)
        except Exception as exc:
            st.error(f"Error creating mock dossier: {exc}")


if __name__ == "__main__":
    main()
