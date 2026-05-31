"""Mock interface for creating a BenchmarkPair to send into the GreenLight pipeline.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from typing import Any

import streamlit as st
from schemas import (
    BenchmarkPair,
    EvidencePackage,
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
    pair_key = (target_symbol.strip().lower(), disease_name.strip().lower())
    if pair_key in {
        ("gapdh", "hypercholesterolemia"),
        ("actb", "type 2 diabetes mellitus"),
        ("apoe", "hypercholesterolemia"),
    }:
        vclaim = VerifiedClaim(
            claim=f"{target_symbol} has only weak or nonspecific literature signal for {disease_name}.",
            evidence_type=EvidenceType.correlational,
            verdict=Verdict.uncertain,
            verdict_confidence=0.38,
            tier_weight=0.15,
            source_db=SourceDB.pubmed,
            source_id="mock-negative-control",
            source_text="The available evidence is indirect, nonspecific, or not clearly target-disease causal.",
            publication_date=cutoff_date,
        )
        return Dossier(
            pair_id=pair_id,
            target_symbol=target_symbol.strip(),
            target_id=target_id.strip(),
            disease_name=disease_name.strip(),
            disease_id=disease_id.strip(),
            cutoff_date=cutoff_date,
            verified_claims=[vclaim],
            viability_score=0.18,
            recommendation=Recommendation.deprioritize,
            rationale=(
                f"{target_symbol} is treated as a low-viability/negative-control example for "
                f"{disease_name}: evidence is weak, nonspecific, and not enough for a pursue call."
            ),
            supported_count=0,
            contradicted_count=0,
            unsupported_count=0,
            tier_breakdown={"correlational": 1},
            flags=["negative_control", "insufficient_strong_evidence"],
            model_used="gpt-mock",
            runtime_seconds=1.2,
        )

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


KNOWN_PAIR_IDS: dict[tuple[str, str], tuple[str, str]] = {
    ("pcsk9", "hypercholesterolemia"): ("ENSG00000169174", "EFO_0004911"),
    ("hmgcr", "hypercholesterolemia"): ("ENSG00000113161", "EFO_0004911"),
    ("ldlr", "hypercholesterolemia"): ("ENSG00000130164", "EFO_0004911"),
    ("apoe", "alzheimer disease"): ("ENSG00000130203", "EFO_0000249"),
    ("tnf", "rheumatoid arthritis"): ("ENSG00000232810", "EFO_0000685"),
    ("gapdh", "hypercholesterolemia"): ("ENSG00000111640", "EFO_0004911"),
    ("actb", "type 2 diabetes mellitus"): ("ENSG00000075624", "EFO_0001360"),
}

KNOWN_DISEASE_IDS: dict[str, str] = {
    "hypercholesterolemia": "EFO_0004911",
    "alzheimer disease": "EFO_0000249",
    "rheumatoid arthritis": "EFO_0000685",
    "type 2 diabetes mellitus": "EFO_0001360",
}


def _pair_ids_for_pipeline(pair: BenchmarkPair, use_mock_engine: bool) -> tuple[str | None, str | None]:
    if use_mock_engine:
        return None, None
    fallback = KNOWN_PAIR_IDS.get((pair.target_symbol.strip().lower(), pair.disease_name.strip().lower()))
    if fallback:
        return fallback
    target_id = None if pair.target_id == "unknown" else pair.target_id
    disease_id = KNOWN_DISEASE_IDS.get(pair.disease_name.strip().lower())
    if disease_id is None:
        disease_id = None if pair.disease_id == "unknown" else pair.disease_id
    if target_id and disease_id:
        return target_id, disease_id
    return target_id, disease_id


def run_pipeline_dossier(
    pair: BenchmarkPair,
    *,
    engine_mode: str,
    verifier_mode: str,
    include_search: bool,
    enrich_budget: int = 8,
    max_items: int = 40,
    allow_demo_fallback: bool = True,
) -> tuple[Dossier, dict[str, Any]]:
    """Run A -> B for one pair and return the dossier plus package metadata."""
    from engine import search as engine_search
    from verifier import verify_and_synthesize

    pkg, metadata = build_pipeline_package(
        pair,
        engine_mode=engine_mode,
        enrich_budget=enrich_budget,
        max_items=max_items,
    )
    use_mock_engine = bool(metadata["used_mock_engine"])
    search_fn = engine_search if include_search and not use_mock_engine else None
    dossier = verify_and_synthesize(pkg, mode=verifier_mode, search=search_fn)
    verifier_fallback_used = False
    if (
        verifier_mode in {"local", "modal"}
        and pkg.evidence_items
        and dossier.viability_score == 0
        and dossier.verified_claims
        and all(v.verdict == Verdict.uncertain and v.verdict_confidence == 0 for v in dossier.verified_claims)
    ):
        if not allow_demo_fallback:
            raise RuntimeError(
                f"{verifier_mode} verifier returned only zero-confidence uncertain claims. "
                "Enable demo fallback or check OpenAI/Modal credentials."
            )
        verifier_fallback_used = True
        dossier = verify_and_synthesize(pkg, mode="fixture", search=None)
        dossier.flags = sorted(set(dossier.flags + ["verifier_unavailable_fallback"]))

    live_empty_fallback_used = False
    if not use_mock_engine and not pkg.evidence_items:
        if not allow_demo_fallback:
            raise RuntimeError(
                "Live retrieval returned no usable pre-cutoff evidence and demo fallback is disabled. "
                f"resolved_pair_id={pkg.pair_id}; target_id={pkg.target_id}; disease_id={pkg.disease_id}; "
                f"engine_errors={pkg.errors}"
            )
        live_empty_fallback_used = True
        pkg, metadata = build_pipeline_package(
            pair,
            engine_mode="Mock/offline engine",
            enrich_budget=enrich_budget,
            max_items=max_items,
        )
        dossier = verify_and_synthesize(pkg, mode="fixture", search=None)
        dossier.flags = sorted(set(dossier.flags + ["live_evidence_empty_fallback"]))

    metadata.update({
        "verifier_mode": verifier_mode,
        "effective_verifier_mode": "fixture" if verifier_fallback_used or live_empty_fallback_used else verifier_mode,
        "included_independent_search": bool(search_fn),
        "verifier_fallback_used": verifier_fallback_used,
        "live_empty_fallback_used": live_empty_fallback_used,
        "allow_demo_fallback": allow_demo_fallback,
    })
    return dossier, metadata


def build_pipeline_package(
    pair: BenchmarkPair,
    *,
    engine_mode: str,
    enrich_budget: int,
    max_items: int,
) -> tuple[EvidencePackage, dict[str, Any]]:
    from engine import build_evidence_package

    use_mock_engine = engine_mode == "Mock/offline engine"
    target_id, disease_id = _pair_ids_for_pipeline(pair, use_mock_engine)
    cutoff = pair.cutoff_date or date(2010, 1, 1)

    pkg = build_evidence_package(
        pair.target_symbol,
        pair.disease_name,
        cutoff,
        mock=use_mock_engine,
        target_id=target_id,
        disease_id=disease_id,
        enrich_budget=enrich_budget,
        max_items=max_items,
    )
    metadata = {
        "pair_id": pkg.pair_id,
        "evidence_items": len(pkg.evidence_items),
        "papers_processed": pkg.papers_processed,
        "sources_queried": pkg.sources_queried,
        "engine_runtime_seconds": pkg.runtime_seconds,
        "engine_errors": pkg.errors,
        "used_mock_engine": use_mock_engine,
        "enrich_budget": enrich_budget,
        "max_items": max_items,
    }
    return pkg, metadata


def _pipeline_failure_result(
    pair: BenchmarkPair,
    label: str,
    exc: Exception,
    *,
    engine_mode: str,
    verifier_mode: str,
    allow_demo_fallback: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "label": label,
        "pair": pair.model_dump(mode="json"),
        "dossier": None,
        "metadata": {
            "requested_engine_mode": engine_mode,
            "requested_verifier_mode": verifier_mode,
            "allow_demo_fallback": allow_demo_fallback,
        },
        "fallback": False,
        "error": f"{type(exc).__name__}: {exc}",
    }
    if allow_demo_fallback:
        fallback = build_mock_dossier(
            pair.target_symbol,
            pair.target_id,
            pair.disease_name,
            pair.disease_id,
            pair.cutoff_date or date(2010, 1, 1),
        )
        result.update({
            "dossier": fallback.model_dump(mode="json"),
            "metadata": {
                "used_mock_dossier": True,
                "requested_engine_mode": engine_mode,
                "requested_verifier_mode": verifier_mode,
                "allow_demo_fallback": True,
            },
            "fallback": True,
        })
    return result


def run_pipeline_batch(
    pairs: list[BenchmarkPair],
    *,
    engine_mode: str,
    verifier_mode: str,
    include_search: bool,
    enrich_budget: int,
    max_items: int,
    allow_demo_fallback: bool,
    max_workers: int,
) -> list[dict[str, Any]]:
    """Run A -> B for multiple pairs concurrently."""
    if not pairs:
        return []

    worker_count = max(1, min(max_workers, len(pairs)))

    if verifier_mode == "modal" and not include_search:
        results: list[dict[str, Any] | None] = [None] * len(pairs)
        package_slots: list[tuple[BenchmarkPair, EvidencePackage, dict[str, Any]] | None] = [None] * len(pairs)

        def build_one(index: int, pair: BenchmarkPair) -> tuple[int, BenchmarkPair, EvidencePackage | None, dict[str, Any], Exception | None]:
            try:
                pkg, metadata = build_pipeline_package(
                    pair,
                    engine_mode=engine_mode,
                    enrich_budget=enrich_budget,
                    max_items=max_items,
                )
                return index, pair, pkg, metadata, None
            except Exception as exc:
                return index, pair, None, {}, exc

        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = [pool.submit(build_one, idx, pair) for idx, pair in enumerate(pairs)]
            for future in as_completed(futures):
                idx, pair, pkg, metadata, exc = future.result()
                label = f"{pair.target_symbol} | {pair.disease_name}"
                if exc is not None:
                    results[idx] = _pipeline_failure_result(
                        pair,
                        label,
                        exc,
                        engine_mode=engine_mode,
                        verifier_mode=verifier_mode,
                        allow_demo_fallback=allow_demo_fallback,
                    )
                    continue
                assert pkg is not None
                if not metadata["used_mock_engine"] and not pkg.evidence_items:
                    err = RuntimeError(
                        "Live retrieval returned no usable pre-cutoff evidence and demo fallback is disabled. "
                        f"resolved_pair_id={pkg.pair_id}; target_id={pkg.target_id}; disease_id={pkg.disease_id}; "
                        f"engine_errors={pkg.errors}"
                    )
                    if allow_demo_fallback:
                        results[idx] = _pipeline_failure_result(
                            pair,
                            label,
                            err,
                            engine_mode=engine_mode,
                            verifier_mode=verifier_mode,
                            allow_demo_fallback=True,
                        )
                    else:
                        results[idx] = _pipeline_failure_result(
                            pair,
                            label,
                            err,
                            engine_mode=engine_mode,
                            verifier_mode=verifier_mode,
                            allow_demo_fallback=False,
                        )
                    continue
                package_slots[idx] = (pair, pkg, metadata)

        verify_indexes = [idx for idx, slot in enumerate(package_slots) if slot is not None]
        verify_pkgs = [package_slots[idx][1] for idx in verify_indexes]  # type: ignore[index]
        if verify_pkgs:
            try:
                from verifier import verify_and_synthesize_batch

                dossiers = verify_and_synthesize_batch(verify_pkgs, mode="modal")
                for idx, dossier in zip(verify_indexes, dossiers):
                    pair, _pkg, metadata = package_slots[idx]  # type: ignore[misc]
                    metadata.update({
                        "verifier_mode": "modal",
                        "effective_verifier_mode": "modal",
                        "included_independent_search": False,
                        "verifier_fallback_used": False,
                        "live_empty_fallback_used": False,
                        "allow_demo_fallback": allow_demo_fallback,
                    })
                    results[idx] = {
                        "label": f"{pair.target_symbol} | {pair.disease_name}",
                        "pair": pair.model_dump(mode="json"),
                        "dossier": dossier.model_dump(mode="json"),
                        "metadata": metadata,
                        "fallback": False,
                        "error": "",
                    }
            except Exception as exc:
                for idx in verify_indexes:
                    pair = package_slots[idx][0]  # type: ignore[index]
                    results[idx] = _pipeline_failure_result(
                        pair,
                        f"{pair.target_symbol} | {pair.disease_name}",
                        exc,
                        engine_mode=engine_mode,
                        verifier_mode=verifier_mode,
                        allow_demo_fallback=allow_demo_fallback,
                    )

        return [result for result in results if result is not None]

    results: list[dict[str, Any] | None] = [None] * len(pairs)

    def run_one(index: int, pair: BenchmarkPair) -> tuple[int, dict[str, Any]]:
        label = f"{pair.target_symbol} | {pair.disease_name}"
        try:
            dossier, metadata = run_pipeline_dossier(
                pair,
                engine_mode=engine_mode,
                verifier_mode=verifier_mode,
                include_search=include_search,
                enrich_budget=enrich_budget,
                max_items=max_items,
                allow_demo_fallback=allow_demo_fallback,
            )
            return index, {
                "label": label,
                "pair": pair.model_dump(mode="json"),
                "dossier": dossier.model_dump(mode="json"),
                "metadata": metadata,
                "fallback": False,
                "error": "",
            }
        except Exception as exc:
            return index, _pipeline_failure_result(
                pair,
                label,
                exc,
                engine_mode=engine_mode,
                verifier_mode=verifier_mode,
                allow_demo_fallback=allow_demo_fallback,
            )

    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        future_to_label = {
            pool.submit(run_one, idx, pair): f"{pair.target_symbol} | {pair.disease_name}"
            for idx, pair in enumerate(pairs)
        }
        for future in as_completed(future_to_label):
            idx, result = future.result()
            results[idx] = result

    return [result for result in results if result is not None]


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

    target_options = ["PCSK9", "HMGCR", "APOE", "TNF", "GAPDH", "ACTB"]
    disease_options = [
        "hypercholesterolemia",
        "Alzheimer disease",
        "rheumatoid arthritis",
        "type 2 diabetes mellitus",
    ]

    with st.form("benchmark_pair_form"):
        selected_targets = st.multiselect(
            "Targets",
            options=target_options,
            default=["PCSK9"],
            help="Select one or more targets.",
            accept_new_options=True
        )
        selected_diseases = st.multiselect(
            "Diseases",
            options=disease_options,
            default=["hypercholesterolemia"],
            help="Select one or more diseases.",
            accept_new_options=True
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
                    known_ids = KNOWN_PAIR_IDS.get((target_symbol.strip().lower(), disease_name.strip().lower()))
                    if known_ids:
                        target_id, disease_id = known_ids
                    elif disease_name.strip().lower() in KNOWN_DISEASE_IDS:
                        disease_id = KNOWN_DISEASE_IDS[disease_name.strip().lower()]

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
        pipeline_pairs = benchmark_pairs
    else:
        # Default pair keeps the rest of the page usable before first submit.
        target_symbol = "PCSK9"
        disease_name = "hypercholesterolemia"
        target_id, _ = resolve_target_id(target_symbol)
        disease_id = KNOWN_DISEASE_IDS[disease_name]
        pipeline_pairs = [
            build_benchmark_pair(
                target_symbol,
                target_id if target_id != "unknown" else "ENSG00000169174",
                disease_name,
                disease_id,
                date(2010, 1, 1),
            )
        ]

    preview_pair = pipeline_pairs[0]
    target_symbol = preview_pair.target_symbol
    target_id = preview_pair.target_id
    disease_name = preview_pair.disease_name
    disease_id = preview_pair.disease_id

    st.divider()

    st.markdown("### Run 2010 Dossiers")
    st.caption(
        "Runs Person A's evidence package builder and Person B's verifier for every generated "
        "target-disease pair. Use fixture mode for reliable offline demos; switch to live modes "
        "when keys and network are ready."
    )
    st.caption(
        "Candidate pairs: "
        + ", ".join(f"{pair.target_symbol} | {pair.disease_name}" for pair in pipeline_pairs)
    )

    pcol1, pcol2, pcol3 = st.columns(3)
    with pcol1:
        engine_mode = st.selectbox(
            "Evidence source",
            ["Mock/offline engine", "Live Open Targets + Europe PMC"],
            index=0,
            help="Mock/offline avoids network calls. Live mode queries external biomedical APIs.",
        )
    with pcol2:
        verifier_mode = st.selectbox(
            "Verifier mode",
            ["fixture", "local", "modal"],
            index=0,
            help="Fixture is deterministic for demos. Local/Modal use the OpenAI entailment judge.",
        )
    with pcol3:
        include_search = st.checkbox(
            "B3/B4 independent search",
            value=False,
            help="Adds Europe PMC corroboration and contradiction search. Best used in live mode.",
        )
    allow_demo_fallback = st.checkbox(
        "Allow demo fallback",
        value=verifier_mode == "fixture" or engine_mode == "Mock/offline engine",
        help=(
            "When enabled, the app may fall back to fixture/mock output if live retrieval or verification fails. "
            "Turn this off to confirm Modal/local verification is really being used."
        ),
    )
    enrich_budget = 8
    max_items = 40
    if engine_mode == "Live Open Targets + Europe PMC":
        bcol1, bcol2 = st.columns(2)
        with bcol1:
            enrich_budget = st.slider(
                "Live enrichment budget",
                min_value=1,
                max_value=25,
                value=8,
                help="Caps Europe PMC/OpenAI extraction attempts. Higher is slower but can find more evidence.",
            )
        with bcol2:
            max_items = st.slider(
                "Max evidence items",
                min_value=5,
                max_value=100,
                value=40,
                step=5,
                help="Caps claims handed to the verifier.",
            )
    parallel_workers = st.slider(
        "Parallel pair workers",
        min_value=1,
        max_value=max(1, min(8, len(pipeline_pairs))),
        value=max(1, min(4, len(pipeline_pairs))),
        help="Number of target-disease pairs to run concurrently.",
    )

    config_key = (
        f"{engine_mode}|{verifier_mode}|search={include_search}|fallback={allow_demo_fallback}|"
        f"budget={enrich_budget}|max={max_items}|workers={parallel_workers}"
    )
    pair_key = "|".join(f"{pair.target_symbol}:{pair.disease_name}:{pair.pair_id}" for pair in pipeline_pairs)
    run_pipeline_key = f"run-pipeline-batch::{pair_key}::{config_key}"
    pipeline_state_key = f"pipeline-batch-result::{pair_key}::{config_key}"
    st.caption(
        f"Selected run: evidence={engine_mode}, verifier={verifier_mode}, "
        f"fallback={'on' if allow_demo_fallback else 'off'}, pairs={len(pipeline_pairs)}."
    )
    if st.button(f"Run A -> B dossiers ({len(pipeline_pairs)} pairs)", key=run_pipeline_key, use_container_width=True):
        with st.spinner("Building evidence, verifying claims, and synthesizing dossier..."):
            st.write(
                f"Launching {len(pipeline_pairs)} pair run(s) with {parallel_workers} parallel worker(s)."
            )
            batch_results = run_pipeline_batch(
                pipeline_pairs,
                engine_mode=engine_mode,
                verifier_mode=verifier_mode,
                include_search=include_search,
                enrich_budget=enrich_budget,
                max_items=max_items,
                allow_demo_fallback=allow_demo_fallback,
                max_workers=parallel_workers,
            )
            st.session_state[pipeline_state_key] = batch_results

    pipeline_results = st.session_state.get(pipeline_state_key, [])
    if pipeline_results:
        summary_rows = []
        for result in pipeline_results:
            dossier_data = result.get("dossier")
            if dossier_data:
                dossier = Dossier.model_validate(dossier_data)
                metadata = result.get("metadata", {})
                summary_rows.append({
                    "Pair": result["label"],
                    "Score": dossier.viability_score,
                    "Recommendation": dossier.recommendation.value,
                    "Effective Verifier": metadata.get("effective_verifier_mode", metadata.get("verifier_mode", "n/a")),
                    "Evidence Items": metadata.get("evidence_items", "n/a"),
                    "Error": result.get("error", ""),
                })
            else:
                summary_rows.append({
                    "Pair": result["label"],
                    "Score": None,
                    "Recommendation": "failed",
                    "Effective Verifier": "n/a",
                    "Evidence Items": "n/a",
                    "Error": result.get("error", ""),
                })
        st.dataframe(summary_rows, width="stretch", hide_index=True)

        tabs = st.tabs([result["label"] for result in pipeline_results])
        for idx, (tab, result) in enumerate(zip(tabs, pipeline_results), start=1):
            with tab:
                if result.get("error") and not result.get("dossier"):
                    st.error(f"Pipeline failed: {result['error']}")
                    with st.expander("Pipeline run metadata", expanded=True):
                        st.json(result.get("metadata", {}))
                    continue
                if result.get("fallback"):
                    st.warning(
                        "Live pipeline was unavailable, so the page is showing a fallback mock dossier. "
                        f"Error: {result.get('error', 'unknown error')}"
                    )
                try:
                    dossier = Dossier.model_validate(result["dossier"])
                    metadata = result.get("metadata", {})
                    effective_mode = metadata.get("effective_verifier_mode", metadata.get("verifier_mode"))
                    if effective_mode:
                        st.info(f"Effective verifier mode for this result: {effective_mode}")
                    display_dossier_summary(dossier, key_suffix=f"_pipeline_{idx}")
                    with st.expander("Pipeline run metadata", expanded=False):
                        st.json(metadata)
                    if metadata.get("verifier_fallback_used"):
                        st.warning(
                            "The selected verifier returned only zero-confidence uncertain claims, "
                            "so the app reran the same evidence with fixture verification for a usable demo."
                        )
                    if metadata.get("live_empty_fallback_used"):
                        st.warning(
                            "Live retrieval returned no usable pre-cutoff evidence, so the app displayed "
                            "the offline fixture dossier instead."
                        )
                except Exception as exc:
                    st.error(f"Could not render pipeline dossier: {exc}")

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
