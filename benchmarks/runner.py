"""
Orchestrates the LoCoMo run across parallel worker processes.

Isolation model (this is the whole reason the runner exists):

Memora's Redis keys are GLOBAL -- `mem:`, `type:`, `dedup:`, `recent_memories` carry no
user namespace, and `clear_memories()` wipes every user plus the entire Qdrant collection.
Running two conversations against one backend therefore cross-contaminates them and
corrupts the benchmark. So each worker slot gets:

  * its own Redis logical DB       (REDIS_DB=<slot>)
  * its own Qdrant collection      (QDRANT_COLLECTION=locomo_w<slot>)
  * its own flat-file user dir     (user_id=locomo_<sample_id>)

Redis ships with 16 logical DBs, so slots are capped at 16 (raise `databases` in
redis.conf to go higher). Each worker is a subprocess because src/config.py freezes those
env vars into module constants at import time.

Resumable: a conversation whose result JSON already exists is skipped, so an interrupted
run continues where it stopped.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from .dataset import load_conversations
from .paths import LOG_DIR, RAW_DIR, REPO_ROOT, describe, ensure_dirs

MAX_REDIS_DBS = int(os.getenv("BENCH_MAX_REDIS_DBS", "16"))
HEARTBEAT_SECONDS = int(os.getenv("BENCH_HEARTBEAT_SECONDS", "30"))


def _tail(path: Path, n: int) -> List[str]:
    """Last n non-blank lines of a file, or [] if unreadable.

    Workers write to their log continuously; this is only ever read for display, so any
    error here must be swallowed rather than killing a run that is otherwise fine.
    """
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            lines = [ln.rstrip() for ln in fh.readlines() if ln.strip()]
        return lines[-n:]
    except OSError:
        return []


class Slot:
    def __init__(self, index: int):
        self.index = index
        self.redis_db = index % MAX_REDIS_DBS
        self.collection = f"locomo_w{index}"
        self.proc: Optional[subprocess.Popen] = None
        self.sample_id: Optional[str] = None
        self.started: float = 0.0
        self.log_handle = None

    def launch(self, sample_id: str, out_path: Path, extra_args: List[str]) -> None:
        env = os.environ.copy()
        env["REDIS_DB"] = str(self.redis_db)
        env["QDRANT_COLLECTION"] = self.collection
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        # Each worker embeds on CPU; without this every worker grabs all cores and they
        # thrash. One thread per worker, parallelism comes from the process count.
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")

        cmd = [
            sys.executable, "-m", "benchmarks.worker",
            "--sample-id", sample_id,
            "--out", str(out_path),
        ] + extra_args

        # Optional NUMA pinning. On the dual-socket target box the intended layout is a
        # locally-served LLM on one node and the benchmark on the other, so each gets its
        # own memory controllers and they never contend for DRAM bandwidth -- which is the
        # binding resource for CPU inference. Set BENCH_NUMA_NODE=0 alongside
        # NUMA_NODE=1 for scripts/serve_local_model.sh.
        numa_node = os.getenv("BENCH_NUMA_NODE")
        if numa_node and shutil.which("numactl"):
            cmd = ["numactl", f"--cpunodebind={numa_node}",
                   f"--membind={numa_node}"] + cmd

        log_path = LOG_DIR / f"worker_{sample_id}.log"
        self.log_handle = log_path.open("w", encoding="utf-8")
        self.proc = subprocess.Popen(
            cmd, cwd=str(REPO_ROOT), env=env,
            stdout=self.log_handle, stderr=subprocess.STDOUT,
        )
        self.sample_id = sample_id
        self.started = time.time()

    def poll(self) -> Optional[int]:
        if self.proc is None:
            return None
        code = self.proc.poll()
        if code is not None:
            if self.log_handle:
                self.log_handle.close()
                self.log_handle = None
            self.proc = None
        return code

    @property
    def busy(self) -> bool:
        return self.proc is not None


def run(
    workers: int = 4,
    limit: Optional[int] = None,
    only: Optional[List[str]] = None,
    force: bool = False,
    extra_args: Optional[List[str]] = None,
) -> int:
    ensure_dirs()
    extra_args = extra_args or []

    if workers > MAX_REDIS_DBS:
        print(f"warning: {workers} workers requested but only {MAX_REDIS_DBS} Redis DBs "
              f"available; slots will share DBs and corrupt each other. Capping to "
              f"{MAX_REDIS_DBS}.")
        workers = MAX_REDIS_DBS

    try:
        conversations = load_conversations()
    except (FileNotFoundError, ValueError) as exc:
        print(exc)
        return 1
    if only:
        wanted = set(only)
        conversations = [c for c in conversations if c.sample_id in wanted]
    if limit:
        conversations = conversations[:limit]

    if not conversations:
        print("no conversations selected")
        return 1

    pending: List[str] = []
    skipped = 0
    for conv in conversations:
        out = RAW_DIR / f"{conv.sample_id}.json"
        if out.exists() and not force:
            skipped += 1
            continue
        pending.append(conv.sample_id)

    turn_total = sum(len(c.turns) for c in conversations)
    q_total = sum(len([q for q in c.questions if q.gold is not None]) for c in conversations)

    print(describe())
    print()
    print(f"conversations : {len(conversations)} selected, {skipped} already done, "
          f"{len(pending)} to run")
    print(f"turns         : {turn_total}")
    print(f"questions     : {q_total}")
    print(f"workers       : {workers} (Redis DBs 0-{min(workers, MAX_REDIS_DBS) - 1})")
    print()

    if not pending:
        print("nothing to do; pass --force to re-run")
        return 0

    slots = [Slot(i) for i in range(workers)]
    queue = list(pending)
    completed: Dict[str, int] = {}
    started_at = time.time()

    # Worker stdout is redirected to logs/worker_<id>.log, so without this the terminal
    # shows nothing between "-> launched" and "ok" -- minutes of apparent hang. The
    # heartbeat echoes each worker's latest progress line so the run is visibly alive.
    print(f"progress every {HEARTBEAT_SECONDS}s; full logs in {LOG_DIR}")
    print()
    last_beat = time.time()

    try:
        while queue or any(s.busy for s in slots):
            for slot in slots:
                if slot.busy:
                    code = slot.poll()
                    if code is not None:
                        sid = slot.sample_id or "?"
                        elapsed = time.time() - slot.started
                        completed[sid] = code
                        status = "ok" if code == 0 else f"FAILED (exit {code})"
                        print(f"  [{len(completed)}/{len(pending)}] {sid}: {status} "
                              f"in {elapsed / 60:.1f} min", flush=True)
                        if code != 0:
                            print(f"      last log lines from "
                                  f"{LOG_DIR / f'worker_{sid}.log'}:")
                            for ln in _tail(LOG_DIR / f"worker_{sid}.log", 5):
                                print(f"      | {ln}")
                        slot.sample_id = None
                elif queue:
                    sid = queue.pop(0)
                    slot.launch(sid, RAW_DIR / f"{sid}.json", extra_args)
                    print(f"  -> {sid} on slot {slot.index} "
                          f"(db={slot.redis_db}, collection={slot.collection})", flush=True)

            now = time.time()
            if now - last_beat >= HEARTBEAT_SECONDS:
                last_beat = now
                busy = [s for s in slots if s.busy and s.sample_id]
                if busy:
                    print(f"  [{(now - started_at) / 60:5.1f} min] "
                          f"{len(completed)}/{len(pending)} done, "
                          f"{len(busy)} running", flush=True)
                    for s in busy:
                        tail = _tail(LOG_DIR / f"worker_{s.sample_id}.log", 1)
                        if tail:
                            print(f"      {s.sample_id}: {tail[-1][:110]}", flush=True)
            time.sleep(2)
    except KeyboardInterrupt:
        print("\ninterrupted; terminating workers (completed conversations are kept)")
        for slot in slots:
            if slot.proc:
                slot.proc.terminate()
        return 130

    failures = {k: v for k, v in completed.items() if v != 0}
    total_min = (time.time() - started_at) / 60
    print()
    print(f"finished {len(completed)} conversations in {total_min:.1f} min "
          f"({len(failures)} failed)")
    if failures:
        print("failed conversations (see logs/worker_<id>.log):")
        for sid, code in failures.items():
            print(f"  {sid}: exit {code}")

    return 0 if not failures else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the LoCoMo benchmark against Memora")
    ap.add_argument("--workers", type=int, default=int(os.getenv("BENCH_WORKERS", "4")))
    ap.add_argument("--limit", type=int, default=None, help="only the first N conversations")
    ap.add_argument("--only", nargs="*", default=None, help="specific sample_ids")
    ap.add_argument("--force", action="store_true", help="re-run already-completed conversations")
    ap.add_argument("--no-adversarial", action="store_true")
    ap.add_argument("--no-dates", action="store_true")
    ap.add_argument("--max-questions", type=int, default=None,
                    help="Sample this many questions per conversation, spread across "
                         "categories (not the first N)")
    ap.add_argument("--save-context", action="store_true",
                    help="Record the retrieved memory context per question, for "
                         "diagnosing low scores")
    ap.add_argument("--reuse-store", action="store_true",
                    help="Skip ingest when the worker's store is already populated. "
                         "Fast iteration on the reader; invalid after extraction changes")
    ap.add_argument("--max-turns", type=int, default=None,
                    help="Ingest only the first N turns per conversation")
    args = ap.parse_args()

    extra: List[str] = []
    if args.no_adversarial:
        extra.append("--no-adversarial")
    if args.no_dates:
        extra.append("--no-dates")
    if args.max_questions:
        extra += ["--max-questions", str(args.max_questions)]
    if args.save_context:
        extra.append("--save-context")
    if args.reuse_store:
        extra.append("--reuse-store")
    if args.max_turns:
        extra += ["--max-turns", str(args.max_turns)]

    return run(workers=args.workers, limit=args.limit, only=args.only,
               force=args.force, extra_args=extra)


if __name__ == "__main__":
    raise SystemExit(main())
