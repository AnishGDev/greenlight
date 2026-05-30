## Plan: Person C Visual Interfaces

Person C owns only two surfaces: a visual interface for creating `BenchmarkPair` objects, and an intuitive interface for interpreting `Dossier` and `VerifiedClaim` outputs. Person C does not own benchmark scoring, retrospective metrics, pipeline orchestration, cached evaluation results, pitch artifacts, A/B logic, or `eval_demo.py`.

**Steps**
1. **Confirm the schemas Person C consumes**
   - Use the existing schema contracts only as inputs/outputs: `BenchmarkPair`, `Dossier`, `VerifiedClaim`, `Recommendation`, `Verdict`, `EvidenceType`, `Split`, and `make_pair_id`.
   - Do not modify schema definitions unless the team explicitly assigns schema ownership to Person C.
   - For C-owned code, import from the schema path the team has agreed to use. Current code mostly imports from root `schemas.py`, while `dossier_display.py` imports from `schemas.schemas`; pick one consistent import path only after the team confirms the canonical schema location.
   - Dependency: this must be settled before cleaning up imports in `app.py` or `dossier_display.py`.

2. **Build the visual `BenchmarkPair` creation interface in `app.py`**
   - Keep and polish the current target/disease input flow: user enters target symbol and disease name.
   - Resolve IDs using `registries.resolve_target_id()` and `registries.resolve_disease_id()`.
   - Show resolved target and disease IDs read-only, with clear failure/suggestion states.
   - Let the user set C-to-A fields needed by `BenchmarkPair`: `cutoff_date`, `validated_after_cutoff` when required by the selected schema, `validation_evidence`, `is_negative_control`, `split`, and optional notes.
   - Generate `pair_id` with `make_pair_id(target_id, disease_id)`.
   - Render the resulting `BenchmarkPair` as structured JSON using `.model_dump(mode="json")`.
   - Keep this interface focused on creating the contract object; do not run scoring or evaluation.

3. **Design the `BenchmarkPair` UI states**
   - Add validation states for unknown target ID, unknown disease ID, missing required fields, invalid cutoff dates, and partial optional fields.
   - Make it obvious which fields are user-entered versus registry-resolved.
   - Provide a simple copy/download/export affordance for the generated JSON if Streamlit support is already available.
   - Keep labels and validation metadata inside the created object only; do not pass labels into A/B execution code.
   - Depends on step 2.

4. **Build the intuitive `Dossier` summary interface**
   - Use `dossier_display.display_dossier_summary()` as the main rendering surface for a `Dossier` object.
   - Present the headline fields prominently: target/disease, `viability_score`, `recommendation`, and rationale.
   - Show evidence counts clearly: supported, contradicted, unsupported, and total claims.
   - Show `tier_breakdown`, flags, model/runtime metadata, and cutoff date in secondary/detail sections.
   - Keep the display read-only: Person C interprets and presents B's output, but does not compute or change the dossier.

5. **Build the `VerifiedClaim` interpretation interface**
   - Add a claim-level table or expandable list for every `VerifiedClaim` in a dossier.
   - For each claim, show claim text, verdict, confidence, evidence type, tier weight, negative-evidence marker, source database, source ID, source URL when present, source text, and publication date.
   - Use clear visual treatments for verdicts: supported, contradicted, unsupported, and uncertain.
   - Highlight unsupported/contradicted claims and negative evidence so a user can understand why a dossier is risky.
   - Keep this as explanation/rendering only; do not perform entailment, contradiction search, or evidence tiering.
   - Depends on step 4.

6. **Connect app-level dossier input to the display components**
   - Add a section in `app.py` where a user can load or paste a `Dossier` JSON object and render it with the dossier/claim display components.
   - Optionally include a minimal built-in sample dossier only for UI demonstration, if needed to make the display testable without B's live output.
   - Do not call `engine.py`, `verifier.py`, or `eval_demo.py` from this C-owned interface unless the team later expands Person C's scope.
   - Depends on steps 4 and 5.

7. **Verify the two C-owned interfaces**
   - Run `streamlit run app.py`.
   - Manually create a `BenchmarkPair` from the UI and confirm the JSON validates against the active schema.
   - Load or paste a sample `Dossier` JSON and confirm summary fields and every `VerifiedClaim` render correctly.
   - Test failure states: unresolved target, unresolved disease, missing required fields, empty claim list, unsupported claim, contradicted claim, and negative-evidence claim.
   - Confirm no C-owned UI path performs scoring, benchmark evaluation, retrieval, verification, synthesis, Modal fan-out, or pitch artifact generation.

**Relevant files**
- `app.py` — Primary Person C surface for visual `BenchmarkPair` creation and loading/rendering dossier output.
- `dossier_display.py` — Person C display helpers for `Dossier` and `VerifiedClaim` interpretation.
- `registries.py` — Existing ID-resolution helpers used by the visual `BenchmarkPair` creator.
- `schemas.py` — Shared contract definitions if root schema is canonical.
- `schemas/schemas.py` — Shared contract definitions if package schema is canonical; use only if the team confirms this path.
- `requirements.txt` — Streamlit and display dependencies already exist.

**Verification**
1. `streamlit run app.py` opens the Person C interface.
2. Generated `BenchmarkPair` JSON can be validated by constructing a `BenchmarkPair` model and dumping with `.model_dump(mode="json")`.
3. A `Dossier` with multiple `VerifiedClaim` verdicts renders with clear summary and claim-level detail.
4. The UI handles missing/empty/unknown states gracefully.
5. There are no changes to `eval_demo.py`, no scoring implementation, no benchmark metrics, and no A/B pipeline execution in C-owned work.

**Decisions**
- Person C owns visual creation of `BenchmarkPair` objects.
- Person C owns interpretation and presentation of `Dossier` and `VerifiedClaim` objects.
- Person C does not own `EvalResult`, scoring metrics, benchmark construction beyond UI-created `BenchmarkPair` objects, pipeline orchestration, verifier logic, evidence retrieval, pitch outputs, cached evaluation artifacts, or `eval_demo.py`.
- Any schema consolidation is a team/shared responsibility unless explicitly reassigned to Person C.

**Further Considerations**
1. Confirm whether root `schemas.py` or `schemas/schemas.py` is canonical before implementation starts.
2. Confirm whether the active `BenchmarkPair` schema requires labels in the UI; the attached `schemas/schemas.py` currently makes several benchmark fields optional, while root `schemas.py` may differ.
3. If B provides dossier JSON examples, use those as display fixtures rather than creating C-owned synthetic pipeline data.