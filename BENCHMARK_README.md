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

`setup_server.sh` creates the venv, installs deps (CPU-only torch), starts Redis + Qdrant,
downloads `locomo10.json`, and runs preflight.

### Backends without Docker

Backends are started by `scripts/start_backends.sh`, which uses `docker compose` when it is
actually usable and otherwise runs both services as ordinary user processes — **no root, no
container runtime**. That fallback exists because on shared nodes Docker is often installed
but unusable (daemon down, or the user is not in the `docker` group), and the resulting

```
[ FAIL ] qdrant reachable - [Errno 111] Connection refused
```

is not a cosmetic failure: without Qdrant the benchmark still runs and still emits numbers,
just meaningless ones.

```bash
bash scripts/start_backends.sh          # start (auto-detects docker vs native)
bash scripts/start_backends.sh status
bash scripts/start_backends.sh stop
BACKEND_MODE=native bash scripts/start_backends.sh   # force native
```

The native path downloads Qdrant's static binary into `.cache/bin/` and uses any
`redis-server` on `PATH`, building one into `.cache/redis-stable/` only if none exists. It
loads the same `redis.benchmark.conf` as the Docker path, so `databases 64` still applies —
without it the runner cannot give each worker its own logical DB and parallelism silently
caps at 16. Ports come from `.env`, state from `data/`, logs from `logs/`.

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
  wins massively: Qwen3.6-35B-A3B activates only ~3B of its 35B parameters, reading
  ~1.8 GB per token instead of ~21 GB.
- **Prefill is compute bound, and AMX is a 3–4× lever** — but only with a runtime that
  uses it. Plain llama.cpp on AVX-512 leaves most of Emerald Rapids' matmul throughput on
  the table. Prefill dominates this benchmark (~10.8M tokens) because the reader ships ~3k
  tokens of memory context per question.

### Which model

```bash
python -m benchmarks.local_estimate --list --cores 22 --runtime amx
python -m benchmarks.local_estimate --model qwen3.6-35b-a3b --cores 22 --split hybrid
```

On 22 cores, batch 8, ~200 GB/s, AMX runtime — modelled, not measured:

| model | RAM | decode tok/s | prefill tok/s | extraction only | all-local |
|---|---|---|---|---|---|
| qwen2.5-7b | 10 G | 228 | 1230 | 1.8 h | 3.6 h |
| llama-3.1-8b | 10 G | 217 | 1169 | 1.9 h | 3.8 h |
| qwen2.5-14b | 14 G | 117 | 632 | 3.5 h | 7.1 h |
| qwen3-30b-a3b | 24 G | 525 | 2833 | 0.8 h | 1.6 h |
| **qwen3.6-35b-a3b (Q8_0)** ★ | 44 G | **315** | **3117** | **1.0 h** | **1.8 h** |
| qwen3.6-35b-a3b (Q4) | 26 G | 578 | 3117 | 0.7 h | 1.4 h |
| qwen3.6-27b (dense) | 21 G | 64 | 346 | 6.3 h | 12.9 h |
| qwen2.5-32b | 25 G | 53 | 288 | 7.6 h | 15.5 h |
| llama-3.3-70b | 48 G | 25 | 132 | 16.6 h | 33.6 h |

The RAM column includes KV cache and runtime overhead, not just weights. Against the box's
251 GB (189 GB free) every row is comfortable, which is exactly why the recommendation is
Q8 rather than Q4 — see below.

Without AMX (plain llama.cpp), multiply the prefill column by ~0.25 — `qwen3.6-35b-a3b`
extraction becomes ~1.5 h and all-local ~4 h. Still fine, which is why AMX is a
nice-to-have for the MoE and a necessity for anything dense.

**Recommendation: `Qwen3.6-35B-A3B` at `Q8_0`.** Only ~3B of its 35B parameters are active
per token, so it reads ~3.3 GB per token instead of ~70 GB for a dense 35B: 35B-class
extraction quality at *better than 8B* decode speed.

**Q8 rather than Q4, because the target box has 251 GB of RAM.** Size is the only thing Q4
buys that matters elsewhere, and here it buys nothing — ~39 GB versus ~22 GB is noise
against 189 GB free. What Q4 costs is fidelity, and extraction fidelity feeds straight into
the score. The speed difference is real but small in context: the table above puts Q8
extraction at ~1.9 h against ~1.6 h for Q4. **Spending 18 minutes to remove quantisation as
a confound in a published number is an easy trade.** Q4 (`UD-Q4_K_XL`, listed separately as
`qwen3.6-35b-a3b-q4`) remains the right choice on a RAM-constrained box.

If you want quantisation gone entirely, BF16 is ~70 GB and also fits — but it doubles
bytes-per-token against Q8 for a difference from Q8_0 that is very hard to measure.

Note the dense/MoE contrast in that table. `qwen3.6-27b` is *smaller* than
`qwen3.6-35b-a3b` and needs *less* RAM, yet it is ~9× slower, because a dense model reads
every weight for every token. Dense 27B/32B/70B are not viable for ~4,200 extraction
calls — that is days, not hours.

**Do not run the judge locally.** Judge quality moves the reported score directly, and
published LoCoMo numbers use strong judges; a 7–30B local judge adds grading noise to the
exact number you are trying to publish. The recommended split is **extraction local,
reader and judge on API** — that is 4,200 calls moved off the API and only 3,972 left,
which fits comfortably in modest rate limits and costs ~$3.

### Which runtime — this is the biggest lever

**Build llama.cpp from source with AMX. Do not use Ollama or HuggingFace `transformers`
for the actual run.** Runtime choice is worth 3–5× on this box, almost entirely because of
whether AMX kernels are compiled in.

| runtime | verdict |
|---|---|
| **llama.cpp, self-built with AMX** | **Use this.** Native continuous batching, OpenAI-compatible `/v1`, best MoE support, full thread/NUMA control, and `Q8_0` maps cleanly onto AMX-INT8 tiles. |
| `ik_llama.cpp` (fork) | Worth A/B testing — focused on CPU throughput and quantised MoE matmuls. Try if you want the last 20–30%. |
| vLLM CPU backend | Best continuous batching at high concurrency and uses AMX via oneDNN — but it wants BF16/INT8, not GGUF, and 35B-A3B at BF16 is ~70 GB. That *does* fit in this box's 251 GB, so it is a legitimate A/B if you want zero quantisation; expect ~half the decode speed of Q8. |
| OpenVINO GenAI / Model Server | Intel's own stack, excellent AMX-INT8 prefill. More setup friction (model conversion). |
| **Ollama** | **Avoid for throughput.** It wraps llama.cpp but ships generic prebuilt binaries that lack AMX for this CPU, and hides thread/NUMA/batch tuning. Community numbers show Ollama at 55–60 tok/s where raw llama.cpp with explicit flags hit 100+ on identical hardware. |
| HuggingFace `transformers` | **Worst option.** Python generate loop, no continuous batching, no AMX kernels without IPEX. Fine for a one-off sanity check, not for ~6,000 calls. |

```bash
bash scripts/build_llama_cpp.sh                              # ~5-10 min on 64 cores
export PATH="$PWD/.cache/llama.cpp/build/bin:$PATH"
```

The flags that matter, and the one people miss:

```
-DGGML_AMX_TILE=ON -DGGML_AMX_INT8=ON -DGGML_AMX_BF16=ON
-DGGML_AVX512_VNNI=ON      # <-- REQUIRED by the AMX code path; omit it and AMX is
                           #     silently left out of the build
-DGGML_NATIVE=ON
```

The build script checks `/proc/cpuinfo` for `amx_tile`/`amx_int8`/`amx_bf16`/`avx512_vnni`
first, so a VM that doesn't expose AMX to the guest fails visibly rather than quietly
producing a slow binary. Confirm AMX landed by checking the `system_info` line
`llama-server` prints when it loads a model.

### "But llama.cpp loses context / quality"

Partly true, and the true parts are specific and avoidable. Sorted by what actually bites:

**1. `--ctx-size` is divided across `--parallel` slots. This is the big one.**
`--ctx-size 8192 --parallel 10` gives each request ~819 tokens. The reader's prompt is
~3.5k tokens, so it gets **silently truncated** and the model answers from a fragment of
its memory context. That is almost certainly the "context loss" you have heard about, and
it is a misconfiguration, not an engine defect. `serve_local_model.sh` sizes the total as
`CTX_PER_SLOT × PARALLEL` — **but verify it**: `llama-server` logs `n_ctx_per_seq = ...` at
startup. If that is smaller than 8192, set `CTX_TOTAL` explicitly and re-check.

**2. Silent context shifting.** By default an over-long prompt can lose its *beginning* —
exactly where the memory context sits. The script passes `--no-context-shift` so it errors
instead. An error is recoverable; quietly corrupted data is not.

**3. Broken chat templates in GGUF conversions.** This is real and documented — Qwen3 GGUFs
had template bugs that were fixed and re-uploaded. A wrong template degrades output badly
and is easily mistaken for model or engine weakness. Use official or Unsloth GGUFs, pass
`--jinja`, and eyeball a few generations before trusting a run.

**4. KV-cache quantisation** measurably hurts. Default is `f16`; the script pins
`--cache-type-k/v f16` so nobody later "saves RAM" by quantising it.

**5. i-quants without an imatrix** degrade noticeably. Use imatrix-calibrated quants —
Unsloth's `UD-*` are.

**6. Q4 vs BF16 is a genuine fidelity loss** — and this one is quantisation, not llama.cpp.
Reported comparisons put Q8 vs Q4_K_M as "hard to notice in conversation", but *hard to
notice in conversation* is not the same as *no effect on a benchmark*.

**On the 251 GB target box this risk is simply bought off.** Q4 exists to fit models into
RAM you do not have; with 189 GB free there is nothing to fit. Run `Q8_0` (~39 GB) and the
concern shrinks to a rounding error, for ~18 minutes of extra extraction time. Only revisit
Q4 if decode proves too slow in practice — and then measure the cost with
`benchmarks.calibrate` rather than assuming it is negligible.

**What I found no evidence for:** llama.cpp being inherently worse than vLLM at the *same*
quantisation. The documented quality complaints trace to template misconfiguration and KV
quantisation, not to the engine.

### Don't take my word for it — measure it

```bash
python -m benchmarks.calibrate --turns 40 \
    --ref-provider groq --ref-model llama-3.3-70b-versatile \
    --alt-provider openai --alt-model qwen3.6-35b-a3b \
    --alt-base-url http://127.0.0.1:8080/v1
```

This pushes the same 40 real LoCoMo turns through Memora's actual extraction pipeline twice
— once on hosted 70B, once on your local server — and reports how much they agree: recall
against the reference, type agreement, content similarity, and how often the reference
extracted something where the local model extracted nothing. Extraction needs no Redis or
Qdrant, so it costs minutes and a few thousand tokens.

Above ~80% recall and type agreement, the local model is a fair substitute. Below ~60%,
extraction quality would confound the score — the number would reflect your local model
rather than Memora — so keep extraction on the API or fix the setup first.

**Why this matters more than the general argument:** worse extraction means worse memories
means lower recall, so a degraded local model depresses the benchmark for reasons unrelated
to Memora's architecture. That confound is worth ten minutes to rule out.

Note the risk is already structurally contained: the judge stays on the API, and extraction
is structured JSON with a low quality bar — the least fidelity-sensitive role in the
pipeline. Also remember `--no-dates` and similar ablations exist precisely so you can
isolate this kind of variable.

### Quantisation: default to Q8_0

**AMX natively supports only BF16 and INT8.** `Q8_0` maps almost directly onto AMX-INT8
tiles; `Q4_K_M` needs more per-block dequant work before the matmul. So Q8_0 can win on
*prefill* despite reading twice the bytes — and prefill is 10.3M of the 11.3M tokens in
this workload.

Decode moves the other way: ~1.8 GB/token at Q4 vs ~3.3 GB at Q8. Netted out over this
prefill-heavy workload the estimator puts extraction at ~1.9 h (Q8) vs ~1.6 h (Q4).

Given 251 GB of RAM, **Q8_0 is the default** — near-lossless, AMX-friendly, and it takes
quantisation off the list of things you would otherwise have to caveat in a published
number. Q4 is the fallback for a RAM-constrained box, and Unsloth's `UD-Q4_K_XL` dynamic
quant is the best quality-per-byte 4-bit option there.

### Free speedups worth taking

- **Continuous batching** (`--cont-batching`, `--parallel 10`) — the single biggest
  software win after AMX. Weights are read once and reused across all in-flight requests,
  so aggregate decode scales with slot count. Match `--parallel` to the benchmark's
  `--workers`.
- **`--mlock`** — pins weights in RAM. Without it a multi-GB model can be partially paged out
  and decode collapses, since every token touches weights.
- **Large prefill batches** (`--batch-size 2048 --ubatch-size 512`) — longer matmuls give
  AMX more to chew on.
- **MTP speculative decoding**, if your build supports it: `--spec-type draft-mtp
  --spec-draft-n-max 2`. Worth ~1.15–1.25× on MoE models (more on dense). Speculative
  decoding is *especially* valuable on bandwidth-bound CPU inference because it amortises
  one weight read across several accepted tokens.
- **`OMP_PROC_BIND=close`, `OMP_PLACES=cores`**, and `OMP_NUM_THREADS` = your thread count —
  stops OpenMP from oversubscribing against llama.cpp's own pool.

### Disable thinking mode

Qwen3 and Qwen3.6 ship a reasoning mode. Stage 3 extraction is structured JSON capped at
`STAGE_3_MAX_TOKENS = 500`; a reasoning trace will consume that entire budget, truncate the
JSON, and cost several times the tokens per call. `serve_local_model.sh` passes `--jinja` and
forwards extra arguments, so:

```bash
scripts/serve_local_model.sh model.gguf --chat-template-kwargs '{"enable_thinking":false}'
```

### Measure, don't trust the model

Every throughput number in this file is derived from first principles, not measured. Get a
real figure before planning around it:

```bash
llama-bench -m model.gguf -t 22 -p 3584 -n 128     # prefill 3.5k (reader-shaped), 128 decode
```

Then feed the result back in:
`python -m benchmarks.local_estimate --model qwen3.6-35b-a3b --cores 22 --batch 10`.
A community datapoint reports ~80 tok/s at 4-bit on this exact CPU, which suggests the
modelled numbers here are conservative.

### Pointing Memora at it

No code change needed — the OpenAI SDK reads `OPENAI_BASE_URL` from the environment:

```bash
bash scripts/build_llama_cpp.sh && export PATH="$PWD/.cache/llama.cpp/build/bin:$PATH"
scripts/serve_local_model.sh /path/to/Qwen3.6-35B-A3B-Q8_0.gguf \
  --chat-template-kwargs '{"enable_thinking":false}'

# in .env
LLM_PROVIDER=openai
OPENAI_BASE_URL=http://127.0.0.1:8080/v1
OPENAI_API_KEY=local
LLM_EXTRACTION_MODEL=qwen3.6-35b-a3b
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

## Running it completely free

```bash
python -m benchmarks.free_tier --keys 6
```

Groq free-tier limits (verified June 2026 — re-check, these move):

| model | RPM | RPD | TPM | **TPD** |
|---|---|---|---|---|
| `llama-3.3-70b-versatile` | 30 | 1,000 | 12,000 | **100,000** |
| `llama-3.1-8b-instant` | 30 | 14,400 | 6,000 | **500,000** |

**Tokens per day is the binding limit, not requests per minute.** RPM throttles you for
seconds; a daily cap cannot be waited out or parallelised around — only more accounts widen
it. And limits are enforced **per organization**, so N keys from N different friends are N
independent quotas, while N keys from one account are one quota.

Token cost by role, at 70% escalation:

| role | calls | tokens |
|---|---|---|
| extraction | 4,200 | 4,200,000 |
| **reader** | 1,986 | **7,070,160** |
| judge | 1,986 | **506,430** |

The reader dominates because `MEMORY_TOKEN_BUDGET = 3000` puts ~3.5k input tokens in front
of every question. That one setting is why an all-API free run is hopeless — it would need
**148 free 70B accounts** to finish in a day.

### The free configuration that works

**Extraction and reader local, judge only on Groq free 70B.** The judge is by far the
cheapest role in tokens *and* the one where model quality most directly moves the number
you intend to publish, so it is exactly the right thing to spend a hosted quota on.

| accounts | capacity/day | judge needs | verdict |
|---|---|---|---|
| 6 | 600,000 | 506,430 | fits, **18% headroom** |
| 8 | 800,000 | 506,430 | comfortable |

So **6 keys is already enough** — but the margin is thin, and a burst of retries or
longer-than-modelled answers can push you over, in which case the run spills to a second
day. That is survivable (the harness is resumable) but annoying. **8 keys is the comfortable
number.**

Alternatives, for completeness:

- **Zero keys — fully local.** Free by construction, ~1.6–4.6 h. The judge is then a local
  model, so grading is noisier; disclose it.
- **Judge on free `llama-3.1-8b-instant`** — fits in 2 accounts thanks to the 500k TPD, but
  an 8B grader is a weak judge and undermines the point of the exercise.
- **Do not** try to assemble enough free accounts to host the reader: 89 of them. You own a
  64-core AMX server; use it.

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
| `redis logical DBs: 16 but N workers` | Using stock `redis.conf`. Start via `scripts/start_backends.sh`, which applies `redis.benchmark.conf` (`databases 64`) on both the Docker and native paths. |
| `Error 111 connecting to localhost:6379` / qdrant refused | Backends not running. `bash scripts/start_backends.sh start`, then `status`. If Docker is installed but unusable, the script falls back to user processes automatically; `docker info` failing is the tell. |
| Escalation rate ~100% | Expected off-domain. Stage 2 regexes were tuned on personal-assistant and payment phrasing. Cost impact is modest; recall impact is real. |
| `vector store was DOWN` in scorecard | Qdrant was unreachable and those conversations ran Phase 1 only. Re-run them; do not quote the number. |
| Model download fails | First run needs internet for MiniLM. Pre-seed `.cache/huggingface/` from another machine. |
| Workers OOM | ~1 GB each. Lower `--workers`. |
| All questions score 0 on adversarial | Check the loader is resolving `adversarial_answer`; category 5 has no `answer` key. |
