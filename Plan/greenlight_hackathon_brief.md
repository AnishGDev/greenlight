# GreenLight — an autonomous research agent that decides if a drug target is worth pursuing
*(working name — swap freely)*

## One breath
Give it a **disease** and a **candidate target** (a gene/protein you might drug). A swarm of agents reads the literature and biomedical databases, builds an **evidence dossier**, and returns a **calibrated viability score** — a go/no-go — with **every claim verified against a real source**. We prove it works by **testing it on the past**: hide everything published after a cutoff and check that its top picks are the targets science later validated.

## The one track we are winning: **Applied Autonomous Research**
This is an end-to-end autoresearch agent in a high-impact domain (computational biology — explicitly wanted). Everything else in the build exists to make *this* claim undeniable. We are **not** trying to also win the architecture or retrieval tracks — focus is the point.

## How each judging box gets checked
- **Technical Depth** → real parallel fan-out on Modal (the engine + the scale number) and **one deeply-engineered component: the Verifier** (claim-checking + contradiction-hunting), not eight shallow ones.
- **Originality** → the *closed system + the evaluation method*: a self-verifying autoresearch agent scored on a **held-out temporal benchmark** with **calibrated** viability — not "another bio chatbot."
- **Demo Clarity** → one killer line ("we hid the future and it still found the winners"), one live action (catch a fabricated claim on stage), one number (throughput).
- **Standout = Applied** → the retrospective result is unambiguous, domain-grounded proof.

## The deep core (what we actually build well)
Two things get real engineering effort; everything else is glue.

1. **Parallel evidence engine (the scale).** The planner splits a hypothesis into evidence questions, then fans out **one worker per question/candidate** across Modal containers. This is what lets us run a big sweep and show throughput.
2. **The Verifier (the depth + the credibility).** For every claim in the dossier it:
   - extracts the claim and its cited source,
   - checks the source actually **supports** it (entailment / NLI) — kills hallucinated mechanisms,
   - actively searches for **contradictory evidence and prior failures** (the thing real discovery teams miss),
   - **tiers** the evidence (genetic > in-vivo > in-vitro > correlational) so weak studies don't get overweighted.

The Verifier is also our on-stage moment and doubles as structured-knowledge extraction.

## The proof: test it on the past
You can't run a wet lab in 8 hours, so validate retrospectively:
- feed agents **only data published before a cutoff** (e.g. before 2020),
- have them rank candidate target–disease pairs by viability,
- check that the **top picks are the ones validated later**.
Real number, not a vibe — and the judge never has to evaluate the biology themselves.

## Architecture (solid = we build it; dashed = stretch upside)

```
        Disease + candidate target(s)
                    |
                    v
            [ PLANNER ]  splits into evidence questions, fans out
                    |
   ┌──────────┬─────┼──────┬──────────┐
   v          v     v      v          v
[evidence] [evidence] ...        (1 worker each)   <-- PARALLEL on Modal
  worker ]   worker ]                                  (this is the scale story)
   └──────────┴────┬─┴──────┴──────────┘
                   v
          ╔═══════════════════╗
          ║   VERIFIER  ★     ║  <-- the DEEP component
          ║ claim ⇄ source    ║      entailment check, contradiction
          ║ check, neg-evidence║      hunt, evidence tiering
          ╚═════════╤═════════╝
                    v
            [ SYNTHESIZER ]  -> dossier + calibrated viability score + citations
                    |
                    v
        [ RETROSPECTIVE EVAL HARNESS ]  "hide the future", rank vs what got validated
                    |
   - - - - - - - - -+- - - - - - - - - - - - - - - - - - -
   :  STRETCH (only if core is done & working):           :
   :  [ RAINDROP traces ] -> cluster failures ->          :
   :  [ improvement loop ] edits prompts, re-runs,         :
   :  score climbs across rounds (replay as backup)        :
   - - - - - - - - - - - - - - - - - - - - - - - - - - - -
```

## APIs, tools, datasets (all public)

**Models / reasoning**
- **OpenAI API** — planning, reading papers, writing the dossier *(uses $5k ChatGPT credits)*

**Compute / infra**
- **Modal** — serverless parallel fan-out; one worker per evidence question or candidate pair *(uses $20k Modal credits; the fan-out IS the scalability evidence)*
- **Raindrop + Workshop** (open source, MIT) — traces every step; powers the stretch self-improvement loop *(targets $5k Raindrop prize)*

**Biomedical data**
- **Open Targets Platform** (GraphQL) — target–disease evidence + scores; doubles as **ground truth** for the retrospective test
- **Europe PMC / PubMed** (E-utilities) — **date-filterable** literature (needed for the "hide the future" trick)
- **ChEMBL** (REST) — bioactivity / druggability
- **DisGeNET** — gene–disease links
- **Comparative Toxicogenomics Database (CTD)** — chemical–gene–disease
- **ClinicalTrials.gov** (API) — downstream "did it advance" signal
- *(optional)* **Semantic Scholar** — citation graph
*(Confirm endpoints + rate limits in current docs before building.)*

**Glue**
- OpenAI Agents SDK (or LangChain / LlamaIndex — Workshop plugs into all)
- Python + a lightweight vector store for fetched papers

## The demo (60–90 sec, disciplined)
1. **Value line:** "It decides whether an early drug target is worth pursuing — with every claim verified."
2. **The proof:** "We hid everything published after 2020. Its top-ranked targets are the ones science actually validated — Nx better than random." *(slide)*
3. **Live:** a judge names a target; we run a fresh dossier and the **Verifier catches a fabricated claim** in real time.
4. **Scale:** "We ran this across [1,000+] target–disease pairs — [X] pairs/min, [Y] papers processed." *(throughput number on screen)*
5. *(if working)* "...and it improves itself: here's the mistake Raindrop caught last round and the fix that lifted the score." Otherwise show the recorded replay.

## Metrics (keep them simple to say)
- **Top-k hit rate** — are the later-validated targets in the top picks? *(headline, on held-out diseases)*
- **Calibration** — when it says "70% viable," is it right ~70% of the time?
- **Hallucination catch rate / citation validity** — % of claims the Verifier confirms vs flags *(the depth number)*
- **Throughput** — pairs/min and papers processed *(the scale number)*

## 8-hour plan (working core first, loop is upside)
- **Hr 0–1:** lock scope, pick **one disease area**, pre-fetch data, build the ~40-pair held-out eval set + cutoff.
- **Hr 1–4:** parallel evidence engine on Modal + synthesizer → a working dossier.
- **Hr 3–5:** go **deep on the Verifier** (entailment check + contradiction hunt + tiering).
- **Hr 4–6:** retrospective benchmark working + run the **large throughput sweep**.
- **Hr 5–6:** freeze the working demo, **record a backup**.
- **Stretch:** wire the self-improvement loop; if it doesn't converge in time, replay.
- **Last hour:** rehearse the pitch until it's tight.

**Cut order if behind:** drop the self-improvement loop first, then the throughput sweep. **Never** cut the Verifier or the retrospective benchmark — they are the win.

---

# Team execution plan (3 people)

## Shared setup (first 30 min, together)
- One Git repo, three modules (`engine/`, `verifier/`, `eval_demo/`), one `mocks/` fixture so everyone is unblocked.
- Shared secrets: OpenAI API key, Modal token.
- **Freeze the three interface contracts before writing real logic:**
  - **A → B:** evidence-item schema `{claim, source_id, source_text, evidence_type, date}`
  - **B → C:** dossier schema `{target, disease, claims[], verdicts[], viability_score, citations[]}`
  - **C → A:** benchmark schema `{target, disease, cutoff_date, validated_after_cutoff: bool}`
- Pick the ONE disease area.

## Person A — Data & evidence engine (scale)
**Stack:** Python, `modal` (`.map()` fan-out), `httpx`/`aiohttp` + `tenacity`, `chembl_webresource_client`, GraphQL for Open Targets (`gql`), `pydantic`, `faiss-cpu`/`chromadb`, `pandas`/`pyarrow`, `openai`.
**Tasks**
- Hr 0–1: co-freeze schemas; pick disease.
- Hr 1–3: pre-fetch + cache data (Open Targets, Europe PMC/PubMed with dates, ChEMBL, ClinicalTrials.gov) as jsonl/parquet.
- Hr 1–3: build retrieval layer (chunk + embed + FAISS).
- Hr 2–4: single-pair evidence worker (query → retrieve → LLM extracts evidence items).
- Hr 2–4: date-cutoff filter (coordinate with C).
- Hr 4–5: wrap worker as Modal function; fan out with `.map()`.
- Hr 4–6: large throughput sweep + timing → the pairs/min + papers-processed number.
- Expose a `search()` function for B's contradiction hunt.

## Person B — Verifier & synthesizer (depth)
**Stack:** Python, `openai`, `pydantic`; entailment via strict OpenAI prompt first, optional biomedical NLI model via HuggingFace `transformers` on Modal GPU; `scikit-learn` for the calibratable score.
**Tasks**
- Hr 0–1: co-freeze schemas (own the dossier schema).
- Hr 1–3: claim extractor (claim + cited source) — build against mocks.
- Hr 2–4: entailment Verifier (supports / contradicts / unsupported).
- Hr 3–5: contradiction + negative-evidence hunt via A's `search()`.
- Hr 3–5: evidence tiering (genetic > in-vivo > in-vitro > correlational).
- Hr 4–6: synthesizer → dossier + calibrated viability score + citations.
- Build a fixed "fabricated claim" demo fixture.

## Person C — Benchmark, metrics, demo & pitch (proof + integration lead)
**Stack:** Python, `pandas`/`numpy`/`scikit-learn`, `matplotlib`/`plotly`, **Streamlit**, OBS/QuickTime, slides.
**Tasks**
- Hr 0–1: co-freeze schemas; set the cutoff.
- Hr 1–3: build the retrospective benchmark (~40 pairs, labels from Open Targets/ClinicalTrials.gov, freeze held-out set) — build scoring against mocks.
- Hr 2–4: scoring functions (top-k hit rate, AUPRC, calibration, citation validity).
- Hr 3–5: Streamlit demo (run-a-dossier button, verifier-catch view, metrics dashboard, throughput readout).
- Hr 4–6: own integration; first full end-to-end pass at the mid-afternoon checkpoint.
- Hr 5–6: rehearse the 90-sec pitch + result slide; record backup demo.

## Critical path & cut order
- Critical path: **A → B → C**. Each builds against `mocks/` until integration.
- Mid-afternoon: one integration checkpoint, then freeze scope.
- Last 1–2 hrs: whole team swarms on integration + demo + backup.
- Stretch (unassigned, picked up by whoever finishes): Raindrop self-improvement loop.
- **Cut order if behind:** loop first, then throughput sweep. Never cut the Verifier or the retrospective benchmark.

## Access note
Confirm DisGeNET's current license before relying on it (now registered/freemium); Open Targets + CTD cover most of the same ground freely.
