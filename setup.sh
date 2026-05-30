#!/usr/bin/env bash
# GreenLight one-command setup. Run from the repo root:  bash setup.sh
set -e

echo "==> Creating virtual environment (.venv)"
python3 -m venv .venv

echo "==> Activating and installing requirements"
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

if [ ! -f .env ]; then
  echo "==> Creating .env from template (fill in your OPENAI_API_KEY)"
  cp .env.example .env
fi

echo ""
echo "Done. Next:"
echo "  1. source .venv/bin/activate      (Windows: .venv\\Scripts\\activate)"
echo "  2. edit .env -> add your OPENAI_API_KEY"
echo "  3. modal setup                    (one-time Modal auth, Person A)"
echo "  4. python workflow_example.py     (verify the pipeline runs end to end)"
