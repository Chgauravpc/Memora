# Running the LoCoMo benchmark on the server

Companion to [BENCHMARK_PLAN.md](BENCHMARK_PLAN.md), which explains *why* LoCoMo and what
to expect. This file is the operator's guide.

Target: `/home/kenton/projects/memora` on the Xeon Gold 6530 box. Everything the benchmark
reads or writes stays inside that directory — see [Containment](#containment).

---

## Copying the repo across

```bash
# from the Windows machine, or however you prefer to move it
rsync -av --exclude '.venv' --exclude '.git' --exclude 'memory' \
      --exclude '__pycache__' --exclude 'data' \
      Memora/ kenton@server:/home/kenton/projects/memora/
```

**Exclude `.venv/`.** The repo ships a Windows virtualenv (`.venv/Scripts/`) that cannot
work on Linux. `setup_server.sh` builds a separate `.venv-linux/` so both can coexist if
you copy it anyway.

## Setup

```bash
cd /home/kenton/projects/memora
bash scripts/setup_server.sh
source .venv-linux/bin/activate

# add GROQ_API_KEY to .env, then:
python -m benchmarks.preflight
```

`setup_server.sh` creates the venv, installs deps (CPU-only torch), starts Redis + Qdrant
via `docker-compose.benchmark.yml`, downloads `locomo10.json`, and runs preflight.

Preflight is not a formality. It checks the one failure mode that silently invalidates
results: **if Qdrant is unreachable, Memora degrades to Phase 1 retrieval without erroring**
(the optional-dependency `try/except` in `src/__init__.py` and `memory_system.py` just logs
a warning). The run would finish and produce plausible, meaningless numbers. Preflight fails
loudly instead, and `report.py` flags any conversation whose health check shows the vector
store down.

## Running

```bash
# 1. Verify the dataset parses and the category mapping matches expectations
python -m benchmarks.dataset --inspect

# 2. Project cost and wall clock for your provider tier
python -m benchmarks.estimate --workers 10 --escalation 0.7 --rpm 1000 --tpm 300000

# 3. SMOKE TEST FIRST — one conversation, 20 questions
python run_locomo.py --limit 1 --max-questions 20 --workers 1
python -m benchmarks.report

# 4. Full run
python run_locomo.py --workers 10
python -m benchmarks.report
```

**Do the smoke test.** It measures the real Stage 3 escalation rate, which is the dominant
cost and time variable and is genuinely unknown until you look. `RESULTS_FEBRUARY_2026.md`
records 13.3% in-domain but ~100% on out-of-domain text before patterns were hand-added for
that domain. LoCoMo is out-of-domain. Feed the measured rate back into `estimate.py` before
committing to the full run.

Runs are **resumable** — conversations already in `results/locomo/raw/` are skipped, so
Ctrl-C and restart is safe. `--force` re-runs them.

## Time and cost on this box

Derived from the repo's own measured numbers (575 ms/turn processing, 294 ms retrieval,
`RESULTS_FEBRUARY_2026.md:395-412`) for ~6,000 turns and ~1,986 questions.

| Provider tier | Wall clock | Binding limit |
|---|---|---|
| Groq free (30 RPM / 12k TPM / 1k RPD) | **~8 days** | requests **per day** |
| Groq paid (1k RPM / 300k TPM) | **~40 min – 1 h** | tokens/min |
| Paid, 100% escalation (worst case) | **~50 min** | tokens/min |

Cost is **$7–9** on `groq/llama-3.3-70b` for the whole run — escalation rate barely moves it
because the reader's 3k-token context dominates the token bill, not extraction.

**The free tier cannot run this.** 8,172 calls against a ~1,000 requests/day cap is a
hard floor of ~8 days, and a daily cap cannot be parallelised around. This is the one thing
that will actually block you.

## Running the LLM locally on CPU

Optional, and it removes the rate-limit problem entirely — a self-hosted model has no
daily cap. It trades a hard blocker for slower throughput.

### Hardware, as reported by `lscpu`

2× Xeon Gold 6530 — **64 physical cores** (32/socket, hyperthreading **off**, so 64 CPUs =
64 cores), **2 NUMA nodes**, 320 MiB L3 total. Critically it has **AMX**
(`amx_bf16`/`amx_tile`/`amx_int8`) plus `avx512_bf16` and `avx512_fp16`.

Two facts drive everything below:

- **Decode is memory-bandwidth bound.** Generating one token reads every *active* weight
  from DRAM, so tok/s ≈ bandwidth ÷ active-weight-bytes. DDR5-4800 × 8 channels/socket is
  ~307 GB/s theoretical, ~200 GB/s achieved. This is why a **Mixture-of-Experts** model
  wins massively: Qwen3-30B-A3B activates only ~3.3B of its 30.5B parameters, reading
  ~2 GB per token instead of ~18 GB.
- **Prefill is compute bound, and AMX is a 3–4× lever** — but only with a runtime that
  uses it. Plain llama.cpp on AVX-512 leaves most of Emerald Rapids' matmul throughput on
  the table. Prefill dominates this benchmark (~10.8M tokens) because the reader ships ~3k
  tokens of memory context per question.

### Which model

```bash
python -m benchmarks.local_estimate --list --cores 22 --runtime amx
python -m benchmarks.local_estimate --model qwen3-30b-a3b --cores 22 --split hybrid
```

On 22 cores, batch 8, ~200 GB/s, AMX runtime — modelled, not measured:

| model | RAM | decode tok/s | prefill tok/s | extraction only | all-local |
|---|---|---|---|---|---|
| qwen2.5-7b | 10 G | 228 | 1230 | 1.8 h | 3.6 h |
| llama-3.1-8b | 10 G | 217 | 1169 | 1.9 h | 3.8 h |
| qwen2.5-14b | 14 G | 117 | 632 | 3.5 h | 7.1 h |
| **qwen3-30b-a3b** ★ | 24 G | **525** | **2833** | **0.8 h** | **1.6 h** |
| qwen2.5-32b | 25 G | 53 | 288 | 7.6 h | 15.5 h |
| llama-3.3-70b | 48 G | 25 | 132 | 16.6 h | 33.6 h |

Without AMX (plain llama.cpp), multiply the prefill column by ~0.25 and expect
`qwen3-30b-a3b` extraction to take ~1.7 h and all-local ~4.6 h.

**Recommendation: `Qwen3-30B-A3B-Instruct` at Q4_K_M.** It gives 30B-class extraction
quality at better-than-8B decode speed, because only ~3.3B parameters are active per
token. Dense 32B and 70B are not viable for the ~4,200 extraction calls — 8–25 tok/s means
days, not hours.

**Do not run the judge locally.** Judge quality moves the reported score directly, and
published LoCoMo numbers use strong judges; a 7–30B local judge adds grading noise to the
exact number you are trying to publish. The recommended split is **extraction local,
reader and judge on API** — that is 4,200 calls moved off the API and only 3,972 left,
which fits comfortably in modest rate limits and costs ~$3.

### Pointing Memora at it

No code change needed — the OpenAI SDK reads `OPENAI_BASE_URL` from the environment:

```bash
scripts/serve_local_model.sh /path/to/Qwen3-30B-A3B-Q4_K_M.gguf   # node1, 22 threads

# in .env
LLM_PROVIDER=openai
OPENAI_BASE_URL=http://127.0.0.1:8080/v1
OPENAI_API_KEY=local
LLM_EXTRACTION_MODEL=qwen3-30b-a3b
BENCH_LLM_PROVIDER=groq        # keep reader+judge on the API
```

Note this **changes the system under test**: the default is `llama-3.3-70b-versatile` on
Groq, so a local Qwen3-30B-A3B measures Memora+Qwen3-30B. Legitimate, but report it.

### Can both run at once? Yes — pin them to different sockets

The benchmark's own CPU appetite is small: 10 workers × 1 thread (the runner sets
`OMP_NUM_THREADS=1`), plus Redis and Qdrant ≈ **15–16 cores of the 64**. Cores are not the
contended resource.

**Memory bandwidth is.** Both the model's decode loop and the benchmark's embedding/Qdrant
traffic pull from DRAM, and on one socket they would fight for the same ~200 GB/s. Because
each socket has independent memory controllers, splitting them across NUMA nodes makes the
contention essentially disappear:

```bash
# terminal 1 - model on node1 (22-28 of its 32 cores)
NUMA_NODE=1 THREADS=22 scripts/serve_local_model.sh model.gguf

# terminal 2 - benchmark on node0
BENCH_NUMA_NODE=0 python run_locomo.py --workers 10
```

`numactl` must be installed (`sudo apt install numactl`); without it the model straddles
both sockets, every weight read risks crossing the UPI link, and inference can lose half
its speed or more. The runner honours `BENCH_NUMA_NODE`, and `serve_local_model.sh` honours
`NUMA_NODE`.

**RAM is the one thing to verify** — `lscpu` does not report it. Run `free -g`. Budget
~24 GB for the model plus ~13 GB for the benchmark (10 workers × ~1 GB, Redis, Qdrant), so
**~40 GB for the recommended setup**; ~64 GB if you insist on a dense 70B. A dual-socket
board with 16 DDR5 channels is very unlikely to have less than 128 GB, but check.

### The Xeon is not the bottleneck (for the API path)

64 physical cores across 2 sockets is heavily over-provisioned for the API path, for two
reasons:

1. **Non-LLM work is ~90 ms/turn.** Embedding (MiniLM, 384-dim), Redis writes, Qdrant
   upserts, and consolidation are a rounding error next to a ~1.2 s LLM round trip. At 10
   workers you will see well under 5% CPU utilisation.
2. **LoCoMo has only 10 conversations, so parallelism caps at 10.** Isolation is
   per-conversation (see below), so an 11th worker has nothing to do. `--workers 10` is the
   ceiling for this benchmark regardless of core count.

What actually governs wall clock is **API throughput**. The highest-leverage change is a
higher provider tier or additional Groq accounts — note Groq rate limits are per *account*,
so extra keys on one account do nothing (`GROQ_API_KEYS` rotation only helps across
accounts).

Constraint to watch: **RAM, not CPU.** Each worker loads its own sentence-transformers model
plus torch, ~0.8–1.2 GB, so 10 workers want ~10–12 GB. Check free memory before raising
`--workers`.

This revises the 3–6 h estimate in BENCHMARK_PLAN.md down to well under an hour, because
that figure assumed modest parallelism; with all 10 conversations running concurrently the
ingest phase fully overlaps and the API becomes the only limit.

## Containment

Three things default to outside the repo and are redirected:

| What | Default | Redirected to |
|---|---|---|
| HuggingFace / sentence-transformers cache | `~/.cache/huggingface` | `.cache/huggingface/` |
| Redis data | Docker named volume in `/var/lib/docker/volumes` | `data/redis/` |
| Qdrant data | Docker named volume | `data/qdrant/` |

The model cache is handled by `benchmarks.paths.redirect_caches_into_repo()`, called before
any transformers import (those libraries read the env vars at import time, so ordering
matters). Volumes are bind mounts in `docker-compose.benchmark.yml`.
`benchmarks.paths.assert_contained()` refuses any read/write that resolves outside the repo.

`docker-compose.benchmark.yml` also uses distinct container names (`memora-bench-*`) and
`${REDIS_PORT:-6379}` / `${QDRANT_PORT:-6333}`, so it will not collide with anything already
running on a shared server. Change the ports in `.env` if 6379/6333 are taken — `src/config.py`
reads the same variables, so the two stay in sync.

## Worker isolation — why subprocesses

**Memora's Redis keys are global.** `mem:`, `type:`, `dedup:`, and `recent_memories` carry
no user namespace; `count_memories()` counts everything; `clear_memories()` wipes every
user *and* drops the entire Qdrant collection. Two conversations sharing one backend
cross-contaminate, so each worker slot gets:

- its own Redis logical DB — `REDIS_DB=<slot>`
- its own Qdrant collection — `QDRANT_COLLECTION=locomo_w<slot>`
- its own flat-file dir — `user_id=locomo_<sample_id>`, wiped before each conversation
  (Phase 4 promotion writes core-memory files that are *always* injected verbatim, so
  leftovers would leak one person's memory into another's evaluation)

`src/config.py` freezes `REDIS_DB` and `QDRANT_COLLECTION` into module constants at import
time, so isolation is only achievable by setting the environment **before** `src` is
imported. That is why each worker is a subprocess and not a thread — a thread pool would
silently share DB 0. `redis.benchmark.conf` sets `databases 64` (default 16) to leave
headroom.

## Output

```
results/locomo/raw/<sample_id>.json   per-conversation records + operational stats
results/locomo/scorecard.json         aggregated
results/locomo/scorecard.txt          same, human-readable
logs/worker_<sample_id>.log           per-worker log
```

The scorecard reports **per category**, because the predicted failure profile is bimodal —
decent single-hop recall, poor temporal and multi-hop — and an aggregate hides exactly that.
It separates *graded* from *ungraded* questions so a failed judge call cannot be silently
counted as either correct or incorrect, and reports Stage 3 escalation, sec/turn, and store
size so a larger run can be costed from measurements.

Metrics: **LLM-as-judge is primary** (that is what published Mem0/Zep LoCoMo numbers use, and
comparability is the point of choosing LoCoMo). Token-F1 and exact-match are computed as
deterministic secondaries — if they diverge sharply from the judge, audit the judge first.

## Changes made to `src/` for this

Two, both minimal and disclosed because they alter the system under test:

1. **`src/llm_extractor.py` — Stage 3 retry/backoff.** The attempt count was
   `len(clients)`, so a single-key deployment made exactly **one** attempt with no backoff
   and any transient 429 propagated. Survivable for a demo, fatal for a run making tens of
   thousands of calls. Now floored at `STAGE_3_MAX_ATTEMPTS` (6) with exponential backoff,
   jitter, and respect for the provider's `try again in Xs` hint. Key rotation is retained
   and still tried first. This changes resilience, not extraction quality.
2. **`src/config.py`** — the three knobs for the above
   (`STAGE_3_MAX_ATTEMPTS`, `STAGE_3_BACKOFF_BASE`, `STAGE_3_BACKOFF_MAX`), env-overridable
   per the repo convention that config is the single source of truth.

Nothing else in `src/` is touched. The benchmark reads Memora through its public API only.

## Known limitations

Carried over from BENCHMARK_PLAN.md; these bound how much the resulting number means.

- **Assistant turns are ingested as user turns.** `process_turn(user_message)` has no role
  parameter, so both speakers go through the same path with the speaker name prefixed into
  the text. LoCoMo evidence lives in both speakers' turns, so ingesting one side would
  discard about half of it — but prefixing is a workaround, not attribution.
- **Session dates are prefixed into the message text, not stored as metadata.** Memora
  stamps `timestamp` at ingest time and `process_turn` takes no timestamp argument, so
  `[8 May, 2023]` rides along in the text and reaches the embedding and `source_text` but
  never becomes a queryable field. Expect temporal questions to score poorly for this
  reason as much as any retrieval weakness. `--no-dates` ablates it. Fixing it properly is
  the four-place change described in `CLAUDE.md`.
- **Images are reduced to their `blip_caption`.** LoCoMo is multi-modal; Memora is
  text-only. Any QA pair depending on image content beyond the caption is unanswerable in
  principle.
- **Retrieval uses the question as the query** via `get_prompt_context`, which retrieves
  without extracting, so grading does not write the questions into the store. (Note the
  in-repo `evaluation/` harness uses `process_turn` for its query, which does pollute.)
- **Access counts are double-incremented per retrieval** (`retriever` and
  `_compose_prompt_context`), inflating the frequency signal. Weighted 0.05, so the effect
  is small; left alone rather than papered over.
- **`MEMORY_DECAY_ENABLED = False`**, so the decay path is untested and no decay result
  should be reported.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `redis logical DBs: 16 but N workers` | Using stock `redis.conf`. Start with `docker-compose.benchmark.yml`, which mounts `redis.benchmark.conf` (`databases 64`). |
| Escalation rate ~100% | Expected off-domain. Stage 2 regexes were tuned on personal-assistant and payment phrasing. Cost impact is modest; recall impact is real. |
| `vector store was DOWN` in scorecard | Qdrant was unreachable and those conversations ran Phase 1 only. Re-run them; do not quote the number. |
| Model download fails | First run needs internet for MiniLM. Pre-seed `.cache/huggingface/` from another machine. |
| Workers OOM | ~1 GB each. Lower `--workers`. |
| All questions score 0 on adversarial | Check the loader is resolving `adversarial_answer`; category 5 has no `answer` key. |
