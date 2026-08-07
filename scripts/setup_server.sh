#!/usr/bin/env bash
#
# One-shot setup for the LoCoMo benchmark on the Linux server.
#
# Everything it creates stays inside the repository directory, which is expected to be
# /home/kenton/projects/memora:
#
#     .venv-linux/         fresh virtualenv (the repo ships a WINDOWS .venv/ which cannot
#                          work here -- a separate name so both can coexist)
#     .cache/huggingface/  sentence-transformers model (else it goes to ~/.cache)
#     data/redis/          Redis bind mount (else /var/lib/docker/volumes)
#     data/qdrant/         Qdrant bind mount
#     data/locomo/         dataset
#     results/, logs/      output
#
# Usage:
#     cd /home/kenton/projects/memora
#     bash scripts/setup_server.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV="$REPO_ROOT/.venv-linux"
EXPECTED="/home/kenton/projects/memora"

echo "======================================================================"
echo "Memora LoCoMo benchmark setup"
echo "======================================================================"
echo "repo root: $REPO_ROOT"
if [[ "$REPO_ROOT" != "$EXPECTED" ]]; then
  echo "note: expected $EXPECTED - continuing, everything stays under the actual root"
fi
echo

# ------------------------------------------------------------------ 1. python
echo "--- python ---"
PY=""
for c in python3.12 python3.11 python3.10 python3 python; do
  if command -v "$c" >/dev/null 2>&1; then
    if "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then
      PY="$c"; break
    fi
  fi
done
if [[ -z "$PY" ]]; then
  echo "ERROR: need Python >= 3.9 on PATH" >&2
  exit 1
fi
echo "using $PY ($("$PY" --version 2>&1))"

# ------------------------------------------------------------------ 2. venv
echo
echo "--- virtualenv ---"
# The repo ships .venv/ built on Windows (.venv/Scripts/*). It is inert on Linux; we
# never touch it so the operator's Windows checkout stays intact.
if [[ -d "$VENV" ]]; then
  echo "reusing $VENV"
else
  "$PY" -m venv "$VENV"
  echo "created $VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --quiet --upgrade pip setuptools wheel

# ------------------------------------------------------------------ 3. deps
echo
echo "--- dependencies ---"
# Keep model/tokenizer caches inside the repo for the pip step too.
export HF_HOME="$REPO_ROOT/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export SENTENCE_TRANSFORMERS_HOME="$HF_HOME"
export TORCH_HOME="$REPO_ROOT/.cache/torch"
mkdir -p "$HF_HOME" "$TORCH_HOME"

python -m pip install --quiet -r requirements.txt
python -m pip install --quiet python-dotenv
echo "installed requirements.txt"

# CPU-only torch: sentence-transformers pulls the CUDA build by default, which is several
# GB of wheels this benchmark never uses (embedding on 32 cores is not the bottleneck).
if ! python -c 'import torch' 2>/dev/null; then
  echo "installing CPU-only torch"
  python -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu \
    || python -m pip install --quiet torch
fi

# ------------------------------------------------------------------ 4. dirs
echo
echo "--- directories ---"
mkdir -p data/redis data/qdrant data/locomo results/locomo/raw results/locomo/checkpoints logs
echo "created data/, results/, logs/"

# ------------------------------------------------------------------ 5. env file
echo
echo "--- .env ---"
if [[ -f .env ]]; then
  echo ".env already exists - leaving it alone"
else
  cp .env.benchmark.example .env
  echo "created .env from .env.benchmark.example"
  echo ">>> EDIT .env AND ADD YOUR GROQ_API_KEY BEFORE RUNNING <<<"
fi

# ------------------------------------------------------------------ 6. backends
echo
echo "--- backends (redis + qdrant) ---"
# Delegated to start_backends.sh, which uses docker when it is actually usable and
# otherwise runs both services as plain user processes. The distinction matters on shared
# nodes: docker is frequently installed but unusable (daemon down, or the user is not in
# the `docker` group), and a missing Qdrant does not stop Memora -- it degrades silently
# to Phase 1 retrieval and produces numbers that look fine and mean nothing.
bash scripts/start_backends.sh start || {
  echo
  echo "WARNING: backends did not come up. The benchmark CANNOT produce valid numbers"
  echo "without Qdrant. Investigate before running:"
  echo "    bash scripts/start_backends.sh status"
  echo "    tail -40 logs/redis.log logs/qdrant.log"
}

# ------------------------------------------------------------------ 7. dataset
echo
echo "--- dataset ---"
python -m benchmarks.download || echo "download failed - see instructions above"

# ------------------------------------------------------------------ 8. preflight
echo
echo "--- preflight ---"
python -m benchmarks.preflight || true

cat <<EOF

======================================================================
Next steps
======================================================================
  source $VENV/bin/activate

  # 0. if redis/qdrant are down, start them (docker if usable, else user processes)
  bash scripts/start_backends.sh status
  bash scripts/start_backends.sh start

  # 1. add GROQ_API_KEY to .env, then re-check
  python -m benchmarks.preflight

  # 2. project cost and wall clock for this box
  python -m benchmarks.estimate --workers 8

  # 3. SMOKE TEST FIRST - one conversation, 20 questions (~15 min, <\$1).
  #    This is where you find out the real Stage 3 escalation rate. If it is
  #    near 100%, every downstream estimate multiplies by ~7.
  python run_locomo.py --limit 1 --max-questions 20 --workers 1
  python -m benchmarks.report

  # 4. full run (10 conversations)
  python run_locomo.py --workers 8
  python -m benchmarks.report
======================================================================
EOF
