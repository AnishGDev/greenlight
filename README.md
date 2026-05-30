# GreenLight

Autonomous research agent that scores whether an early drug target is worth pursuing,
verifies every claim against its source, and proves itself by testing on the past.

Three roles, one repo, one shared virtual environment:
- **A** owns `engine.py` (data & evidence engine)
- **B** owns `verifier.py` (verifier & synthesizer)
- **C** owns `eval_demo.py` + `app.py` (benchmark, metrics, Streamlit demo)
- **Shared:** `schemas.py` (the three contracts — freeze first), `workflow_example.py` (worked end-to-end reference + mocks)

---

## 1. Create the GitHub repo (Person A does this once)

```bash
cd greenlight
git init
git add .
git commit -m "scaffold: schemas, engine step 1, stubs, env"

# Option A — GitHub CLI (fastest):
gh repo create greenlight --private --source=. --remote=origin --push

# Option B — manual: create an empty repo on github.com, then:
#   git remote add origin git@github.com:YOURORG/greenlight.git
#   git branch -M main && git push -u origin main
```

Add B and C as collaborators: repo **Settings -> Collaborators**, or
`gh repo edit --add-collaborator <username>` (or via `gh api`).

B and C then clone:
```bash
git clone git@github.com:YOURORG/greenlight.git && cd greenlight
```

## 2. Python environment (each teammate, one shared venv)

Requires **Python 3.11**. One command:
```bash
bash setup.sh
```
Or manually:
```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```
(Optional faster alternative if you have it: `uv venv && uv pip install -r requirements.txt`.)

## 3. Secrets

**Local runs** — put your key in `.env` (gitignored):
```
OPENAI_API_KEY=sk-...
```
Loaded in code with:
```python
from dotenv import load_dotenv; load_dotenv()
```

**Modal (Person A)** — one-time browser auth, then a secret for remote functions:
```bash
modal setup                                   # authenticate this machine
modal secret create openai-secret OPENAI_API_KEY=$OPENAI_API_KEY
```
Reference it in your Modal function:
```python
@app.function(secrets=[modal.Secret.from_name("openai-secret")])
def worker(...): ...
```

## 4. Verify it works
```bash
python workflow_example.py          # full pipeline trace (mocked stages)
python engine.py --mock             # A's engine, offline
```

## 5. Working in parallel (avoid the hour-7 merge disaster)

1. **Commit `schemas.py` to `main` first and announce it frozen.** If a schema must change, change it together and everyone `git pull`.
2. Each person works on **their own file** — conflicts are rare because you don't share files.
3. Optional light branches: `git checkout -b a/engine` etc., merge at the mid-afternoon integration checkpoint. Or commit straight to `main` with frequent `git pull --rebase`.
4. Build against `workflow_example.py`'s mock functions until the real pieces are ready.

## 6. Run each part
```bash
python engine.py --target PCSK9 --disease hypercholesterolemia   # A
python verifier.py                                               # B (against mock pkg)
streamlit run app.py                                             # C (the demo)
```
