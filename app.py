"""Mock interface for creating a BenchmarkPair to send into the GreenLight pipeline.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

from datetime import date
from typing import Any

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
from retrospective import run_retrospective_validation


def build_benchmark_pair(
    target_symbol: str,
    target_id: str,
    disease_name: str,
    disease_id: str,
    cutoff_date: date,
) -> BenchmarkPair:
    pair_id = make_pair_id(target_id.strip(), disease_id.strip())
    return BenchmarkPair(
        pair_id=pair_id,
        target_symbol=target_symbol.strip(),
        target_id=target_id.strip(),
        disease_name=disease_name.strip(),
        disease_id=disease_id.strip(),
        cutoff_date=cutoff_date,
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


def build_mock_retrospective_checks(
    dossier: Dossier,
    target_symbol: str,
    disease_name: str,
) -> list[dict[str, str]]:
    """Build post-cutoff checks that adapt to the selected target-disease pair."""
    expected_direction = "Pursue" if dossier.recommendation == Recommendation.pursue else "Do Not Pursue"
    pair_key = (target_symbol.strip().lower(), disease_name.strip().lower())

    curated_templates: dict[tuple[str, str], list[dict[str, str]]] = {
        (
            "pcsk9",
            "hypercholesterolemia",
        ): [
            {
                "Post-Cutoff Signal": "PCSK9 inhibitors approved and widely used",
                "Modern Evidence Window": "2015-2024",
                "What Happened": "Alirocumab and evolocumab reached approval; LDL reduction and CV benefit were shown in large outcomes trials.",
                "Signal Direction": "Supports Pursue",
            },
            {
                "Post-Cutoff Signal": "Additional modality validation",
                "Modern Evidence Window": "2020-2024",
                "What Happened": "siRNA approach (inclisiran) provided durable LDL lowering, reinforcing target tractability.",
                "Signal Direction": "Supports Pursue",
            },
            {
                "Post-Cutoff Signal": "Safety and biology consistency",
                "Modern Evidence Window": "2010-2024",
                "What Happened": "Human genetics and interventional data remained directionally consistent with PCSK9 inhibition benefit.",
                "Signal Direction": "Supports Pursue",
            },
        ],
        (
            "hmgcr",
            "hypercholesterolemia",
        ): [
            {
                "Post-Cutoff Signal": "Long-run statin outcome confirmation",
                "Modern Evidence Window": "2010-2024",
                "What Happened": "Large real-world and trial follow-up continued to support LDL reduction and cardiovascular risk lowering.",
                "Signal Direction": "Supports Pursue",
            },
            {
                "Post-Cutoff Signal": "Genetic triangulation",
                "Modern Evidence Window": "2012-2024",
                "What Happened": "Mendelian randomization and genetics remained consistent with on-target LDL-mediated benefit.",
                "Signal Direction": "Supports Pursue",
            },
        ],
    }

    base_rows = curated_templates.get(
        pair_key,
        [
            {
                "Post-Cutoff Signal": f"Post-2010 literature for {target_symbol} in {disease_name}",
                "Modern Evidence Window": "2010-2026",
                "What Happened": "No curated mock evidence set exists for this pair yet. Replace this with live retrieval or a manually curated retrospective packet.",
                "Signal Direction": "Needs Curation",
            }
        ],
    )

    checks: list[dict[str, str]] = []
    for row in base_rows:
        signal_direction = row["Signal Direction"]
        if signal_direction == "Needs Curation":
            result = "Needs Curation"
        elif signal_direction == "Supports Pursue":
            result = "Correct" if expected_direction == "Pursue" else "Incorrect"
        else:
            result = "Correct" if expected_direction == "Do Not Pursue" else "Incorrect"

        checks.append(
            {
                "Post-Cutoff Signal": row["Post-Cutoff Signal"],
                "Modern Evidence Window": row["Modern Evidence Window"],
                "What Happened": row["What Happened"],
                "2010 Dossier Direction": expected_direction,
                "Result": result,
            }
        )

    return checks


def render_retrospective_checks(checks: list[dict[str, Any]]) -> None:
    """Render retrospective checks as responsive cards to avoid dataframe scroll traps."""
    for idx, row in enumerate(checks, start=1):
        result = row["Result"]
        with st.container(border=True):
            st.markdown(f"**{idx}. {row['Post-Cutoff Signal']}**")

            meta_left, meta_right = st.columns(2)
            meta_left.caption(f"Evidence window: {row['Modern Evidence Window']}")
            meta_right.caption(f"2010 direction: {row['2010 Dossier Direction']}")

            st.write(row["What Happened"])

            paper_links = row.get("Paper Links", [])
            if paper_links:
                st.markdown("**Papers Quoted**")
                render_paper_links_grid(paper_links)

            if result == "Correct":
                st.success("Result: Correct")
            elif result == "Incorrect":
                st.error("Result: Incorrect")
            else:
                st.warning("Result: Needs Curation")


def _source_to_clickable_link(src: dict[str, Any]) -> dict[str, str]:
    source_id = str(src.get("source_id", "n/a"))
    explicit_url = str(src.get("url", "")).strip()
    if explicit_url:
        return {"label": source_id, "url": explicit_url}

    if source_id.startswith("PMID:"):
        pmid = source_id.split(":", 1)[1].strip()
        if pmid:
            return {"label": source_id, "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"}

    if source_id.startswith("DOI:"):
        doi = source_id.split(":", 1)[1].strip()
        if doi:
            return {"label": source_id, "url": f"https://doi.org/{doi}"}

    return {"label": source_id, "url": ""}


def render_paper_links_grid(links: list[dict[str, str]]) -> None:
    """Render evidence links as boxed cards in a compact two-column grid."""
    if not links:
        return

    for idx in range(0, len(links), 2):
        cols = st.columns(2)
        for col_idx, col in enumerate(cols):
            link_idx = idx + col_idx
            if link_idx >= len(links):
                continue

            link = links[link_idx]
            label = link.get("label", "source")
            url = link.get("url", "")

            with col:
                with st.container(border=True):
                    st.caption(f"Paper {link_idx + 1}")
                    if url:
                        st.link_button(label, url, use_container_width=True)
                    else:
                        st.write(label)


def _pick_links_for_finding(
    finding: dict[str, str],
    sources: list[dict[str, Any]],
    max_links: int = 3,
) -> list[dict[str, str]]:
    finding_text = (
        f"{finding.get('signal', '')} {finding.get('finding', '')}".lower()
    )
    words = [w for w in finding_text.replace("/", " ").replace("-", " ").split() if len(w) > 4]

    scored_sources: list[tuple[int, dict[str, Any]]] = []
    for src in sources:
        src_blob = f"{src.get('detail', '')} {src.get('source_id', '')}".lower()
        score = sum(1 for word in words if word in src_blob)
        scored_sources.append((score, src))

    scored_sources.sort(key=lambda item: item[0], reverse=True)
    selected = [item[1] for item in scored_sources[:max_links]]
    links = [_source_to_clickable_link(src) for src in selected if src.get("source_id")]
    return links


def findings_to_retrospective_checks(
    findings: list[dict[str, str]],
    recommendation_2010: str,
    sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for finding in findings:
        signal_result = finding.get("result", "Mixed")
        if signal_result == "Supports":
            result = "Correct" if recommendation_2010 == "Pursue" else "Incorrect"
        elif signal_result == "Refutes":
            result = "Correct" if recommendation_2010 != "Pursue" else "Incorrect"
        else:
            result = "Needs Curation"

        checks.append(
            {
                "Post-Cutoff Signal": finding.get("signal", "Signal"),
                "Modern Evidence Window": "2011-present",
                "What Happened": finding.get("finding", "No finding provided."),
                "2010 Dossier Direction": recommendation_2010,
                "Result": result,
                "Paper Links": _pick_links_for_finding(finding, sources),
            }
        )
    return checks



def main() -> None:
    st.set_page_config(page_title="GreenLight", page_icon="🧬")
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 82% 10%, rgba(16, 185, 129, 0.16), transparent 40%),
                radial-gradient(circle at 12% 8%, rgba(34, 197, 94, 0.12), transparent 36%),
                #08110f;
            color: #d9efe8;
        }

        h1, h2, h3, h4, label {
            color: #e6fff6 !important;
        }

        p, .stCaption, .stMarkdown {
            color: #b9d6cc;
        }

        [data-testid="stForm"],
        [data-testid="stMetric"],
        [data-testid="stExpander"] {
            background: linear-gradient(160deg, rgba(15, 23, 20, 0.9), rgba(10, 19, 16, 0.9));
            border: 1px solid rgba(52, 211, 153, 0.22);
            padding: 10px;
            border-radius: 14px;
        }

        [data-testid="stTextInputRootElement"] input {
            background: rgba(4, 10, 8, 0.85) !important;
            color: #d7f9ec !important;
            border: 1px solid rgba(74, 222, 128, 0.35) !important;
        }

        .stButton > button {
            background: linear-gradient(135deg, #16a34a, #059669) !important;
            color: #ecfdf5 !important;
            border: 1px solid rgba(167, 243, 208, 0.35) !important;
            border-radius: 10px !important;
            font-weight: 600;
        }

        .stButton > button:hover {
            filter: brightness(1.08);
            box-shadow: 0 0 0 2px rgba(52, 211, 153, 0.25);
        }

        .gl-hero {
            border: 1px solid rgba(52, 211, 153, 0.28);
            border-radius: 16px;
            background: linear-gradient(125deg, rgba(6, 14, 11, 0.95), rgba(8, 24, 18, 0.95));
            padding: 18px 20px;
            margin-bottom: 14px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.28);
        }

        .gl-kicker {
            display: inline-block;
            font-size: 12px;
            color: #86efac;
            border: 1px solid rgba(134, 239, 172, 0.35);
            border-radius: 999px;
            padding: 4px 10px;
            margin-bottom: 8px;
            background: rgba(22, 163, 74, 0.12);
        }

        .gl-title {
            font-size: 38px;
            line-height: 1.05;
            font-weight: 700;
            margin: 0 0 8px 0;
            color: #dcfce7;
        }

        .gl-subtitle {
            margin: 0;
            color: #b8dcd0;
            line-height: 1.55;
            max-width: 900px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <section class="gl-hero">
          <div class="gl-kicker">Medical Research Validation Workspace</div>
          <h1 class="gl-title">GreenLight</h1>
          <p class="gl-subtitle">
            Build a target-disease benchmark pair, run retrospective evidence validation, and inspect finding-level
            interpretation with direct paper traceability from a single interface.
          </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    target_options = ["PCSK9", "HMGCR", "APOE", "TNF"]
    disease_options = [
        "hypercholesterolemia",
        "Alzheimer disease",
        "rheumatoid arthritis",
    ]

    with st.form("benchmark_pair_form"):
        selected_targets = st.multiselect(
            "Targets",
            options=target_options,
            default=["PCSK9"],
            help="Select one or more targets.",
        )
        selected_diseases = st.multiselect(
            "Diseases",
            options=disease_options,
            default=["hypercholesterolemia"],
            help="Select one or more diseases.",
        )

        candidate_pair_count = len(selected_targets) * len(selected_diseases)
        st.caption(f"This will create {candidate_pair_count} BenchmarkPair object(s).")

        use_2010_cutoff = st.checkbox("Cutoff at 2010 (hide future data after 2010)", value=True)
        cutoff_date = date(2010, 1, 1) if use_2010_cutoff else date.today()

        submitted = st.form_submit_button("Check Theory for Viability")

    if submitted:
        if not selected_targets or not selected_diseases:
            st.warning("Select at least one target and one disease.")
        else:
            created_pairs: list[BenchmarkPair] = []
            unresolved_messages: list[str] = []

            for target_symbol in selected_targets:
                for disease_name in selected_diseases:
                    label = f"{target_symbol} | {disease_name}"
                    target_id, target_suggestions = resolve_target_id(target_symbol)
                    disease_id, disease_suggestions = resolve_disease_id(disease_name)

                    if target_id == "unknown" or disease_id == "unknown":
                        warning_text = f"{target_symbol} | {disease_name}: could not fully resolve IDs."
                        if target_suggestions:
                            warning_text += " Target suggestions: " + ", ".join(target_suggestions) + "."
                        if disease_suggestions:
                            warning_text += " Disease suggestions: " + ", ".join(disease_suggestions) + "."
                        unresolved_messages.append(warning_text)

                    try:
                        created_pairs.append(
                            build_benchmark_pair(
                                target_symbol,
                                target_id,
                                disease_name,
                                disease_id,
                                cutoff_date,
                            )
                        )
                    except Exception as exc:
                        st.error(f"Error creating BenchmarkPair for {label}: {exc}")

            if created_pairs:
                st.session_state["benchmark_pairs"] = created_pairs
                st.success(f"Created {len(created_pairs)} BenchmarkPair object(s).")
                with st.expander("View generated BenchmarkPairs", expanded=False):
                    st.json([pair.model_dump(mode="json") for pair in created_pairs])

            for msg in unresolved_messages:
                st.warning(msg)

    benchmark_pairs = st.session_state.get("benchmark_pairs", [])
    if benchmark_pairs:
        active_pair_labels = [f"{pair.target_symbol} | {pair.disease_name}" for pair in benchmark_pairs]
        active_pair_lookup = dict(zip(active_pair_labels, benchmark_pairs))
        selected_active_pair_label = st.selectbox(
            "Active benchmark pair for dossier and retrospective views",
            options=active_pair_labels,
            index=0,
        )
        active_pair = active_pair_lookup[selected_active_pair_label]
        target_symbol = active_pair.target_symbol
        target_id = active_pair.target_id
        disease_name = active_pair.disease_name
        disease_id = active_pair.disease_id
    else:
        # Default pair keeps the rest of the page usable before first submit.
        target_symbol = "PCSK9"
        disease_name = "hypercholesterolemia"
        target_id, _ = resolve_target_id(target_symbol)
        disease_id, _ = resolve_disease_id(disease_name)

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

    with st.expander("🔬 Retrospective Check with Modern Research (Did the 2010 call hold up?)", expanded=False):
        st.markdown(
            "This section compares the 2010-limited dossier recommendation to major post-2010 "
            "clinical and translational outcomes."
        )
        try:
            retrospective_dossier = build_mock_dossier(
                target_symbol if target_id != "unknown" else "PCSK9",
                target_id if target_id != "unknown" else "ENSG00000169174",
                disease_name if disease_id != "unknown" else "Hypercholesterolemia",
                disease_id if disease_id != "unknown" else "EFO_0004911",
                date(2010, 1, 1),
            )

            active_target_symbol = target_symbol if target_id != "unknown" else "PCSK9"
            active_target_id = target_id if target_id != "unknown" else "ENSG00000169174"
            active_disease_name = disease_name if disease_id != "unknown" else "Hypercholesterolemia"
            active_disease_id = disease_id if disease_id != "unknown" else "EFO_0004911"

            recommendation_2010 = retrospective_dossier.recommendation.value.replace("_", " ").title()
            retro_state_key = f"retro::{active_target_id}::{active_disease_id}"
            run_key = f"run-retro::{active_target_id}::{active_disease_id}"

            if st.button("Run post-2010 evidence validation", key=run_key, use_container_width=True):
                with st.spinner("Querying Open Targets and Europe PMC, then summarizing with GPT..."):
                    st.session_state[retro_state_key] = run_retrospective_validation(
                        target_symbol=active_target_symbol,
                        target_id=active_target_id,
                        disease_name=active_disease_name,
                        disease_id=active_disease_id,
                        recommendation_2010=recommendation_2010,
                        cutoff_year=2010,
                    )

            retro_result = st.session_state.get(retro_state_key)
            if not retro_result:
                st.info("Click the button above to run a live retrospective check for this pair.")
            else:
                checks = findings_to_retrospective_checks(
                    retro_result.get("findings", []),
                    recommendation_2010,
                    retro_result.get("sources", []),
                )
                aligned_count = sum(1 for row in checks if row["Result"] == "Correct")
                total_findings_count = len(checks)

                col1, col2, col3 = st.columns(3)
                col1.metric("2010 Recommendation", recommendation_2010)
                col2.metric(
                    "Findings Aligned with 2010 Call",
                    f"{aligned_count}/{total_findings_count}" if total_findings_count else "0/0",
                    help="Includes all findings in the denominator, including mixed/needs-curation findings.",
                )
                col3.metric("Retrospective Verdict", retro_result.get("overall_verdict", "Mixed / Inconclusive"))

                col4, col5, col6 = st.columns(3)
                col4.metric("Confidence", f"{float(retro_result.get('confidence', 0.0)):.2f}")
                col5.metric("Open Targets Evidence", str(retro_result.get("ot_evidence_count", 0)))
                col6.metric("Europe PMC Hits", str(retro_result.get("europe_pmc_count", 0)))

                st.markdown("**Summary**")
                st.write(retro_result.get("summary", "No summary available."))

                if checks:
                    st.markdown("**Finding-Level Interpretation**")
                    render_retrospective_checks(checks)

                sources = retro_result.get("sources", [])
                if sources:
                    with st.expander("Sources used in retrospective check", expanded=False):
                        for src in sources:
                            source_line = (
                                f"{src.get('source', 'source')} | {src.get('source_id', 'n/a')} | "
                                f"{src.get('year', 'n/a')} | {src.get('detail', '')}"
                            )
                            st.caption(source_line)
                            if src.get("url"):
                                st.caption(src["url"])

                for warning in retro_result.get("warnings", []):
                    st.warning(warning)

                if retro_result.get("used_gpt", False):
                    st.caption("Summary engine: GPT")
                else:
                    st.caption("Summary engine: heuristic fallback (GPT unavailable)")

            st.caption(
                "Demo note: this mock retrospective view is illustrative and not a substitute for a full systematic review pipeline."
            )
        except Exception as exc:
            st.error(f"Error generating retrospective check: {exc}")


if __name__ == "__main__":
    main()
