#!/usr/bin/env python3
"""
Run the LoCoMo benchmark against Memora.

Everything stays inside this repository directory (on the server:
/home/kenton/projects/memora) -- dataset, model cache, Redis/Qdrant volumes, results, logs.

Quick start on the server:

    bash scripts/setup_server.sh          # venv, deps, backends, dataset
    source .venv-linux/bin/activate
    # put GROQ_API_KEY in .env
    python -m benchmarks.preflight        # verify everything is live
    python -m benchmarks.estimate --workers 8

    python run_locomo.py --limit 1 --max-questions 20 --workers 1   # SMOKE TEST FIRST
    python -m benchmarks.report

    python run_locomo.py --workers 8      # full run
    python -m benchmarks.report

Resumable: conversations already present in results/locomo/raw/ are skipped, so an
interrupted run continues where it left off. Pass --force to redo them.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.runner import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
