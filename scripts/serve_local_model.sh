#!/usr/bin/env bash
#
# Serve a local model for Memora's Stage 3 extraction, pinned to ONE NUMA node.
#
# Why pinning matters on this box (2x Xeon Gold 6530, 2 NUMA nodes):
#
#   Token generation is memory-BANDWIDTH bound, not compute bound. Each socket has its own
#   8 DDR5 channels (~200 GB/s achieved). If the model's threads and pages straddle both
#   sockets, every weight read has a chance of crossing the UPI link, which is far slower
#   than local DRAM -- inference can lose half its speed or worse.
#
#   Pinning the model to node1 and the benchmark to node0 gives each its own memory
#   controllers, so they do not contend for bandwidth at all. That is what makes running
#   both simultaneously nearly free rather than mutually destructive.
#
# Usage:
#   scripts/serve_local_model.sh /path/to/model.gguf            # defaults: node1, 22 threads
#   NUMA_NODE=1 THREADS=28 PORT=8080 scripts/serve_local_model.sh model.gguf
#
# Then point Memora at it (no code change needed -- the OpenAI SDK reads OPENAI_BASE_URL):
#   LLM_PROVIDER=openai
#   OPENAI_BASE_URL=http://127.0.0.1:8080/v1
#   OPENAI_API_KEY=local
#   LLM_EXTRACTION_MODEL=<whatever name the server reports>

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL="${1:-}"
if [[ -z "$MODEL" ]]; then
  echo "usage: $0 /path/to/model.gguf" >&2
  exit 2
fi
if [[ ! -f "$MODEL" ]]; then
  echo "ERROR: model not found: $MODEL" >&2
  exit 1
fi

NUMA_NODE="${NUMA_NODE:-1}"
THREADS="${THREADS:-22}"
PORT="${PORT:-8080}"
PARALLEL="${PARALLEL:-10}"    # concurrent slots; match the benchmark's --workers
# Per-slot context. Total KV = CTX * PARALLEL, so this is 10 x 8192 = 81920 by default.
# The reader's prompt is ~3.5k tokens, so 8192 is comfortable; do not inflate it, KV
# cache costs RAM and larger contexts slow attention.
CTX_PER_SLOT="${CTX_PER_SLOT:-8192}"
CTX=$(( CTX_PER_SLOT * PARALLEL ))
# Prefill batch sizes. Bigger batches give AMX longer matmuls to chew on, which is where
# the throughput is on this CPU. 2048/512 is a reasonable throughput-oriented default.
BATCH="${BATCH:-2048}"
UBATCH="${UBATCH:-512}"
SERVER_BIN="${SERVER_BIN:-llama-server}"

if ! command -v "$SERVER_BIN" >/dev/null 2>&1; then
  # Fall back to the in-repo build if it exists but is not on PATH.
  if [[ -x "$REPO_ROOT/.cache/llama.cpp/build/bin/llama-server" ]]; then
    SERVER_BIN="$REPO_ROOT/.cache/llama.cpp/build/bin/llama-server"
    echo "using in-repo build: $SERVER_BIN"
  else
    cat >&2 <<EOF
ERROR: '$SERVER_BIN' not on PATH.

Build it WITH AMX (this CPU has amx_tile/amx_int8/amx_bf16, and prebuilt binaries
do not include those kernels):

    bash scripts/build_llama_cpp.sh
    export PATH="\$PWD/.cache/llama.cpp/build/bin:\$PATH"

Or set SERVER_BIN to any OpenAI-compatible server (vLLM CPU, ipex-llm,
OpenVINO Model Server). Anything exposing /v1/chat/completions works --
Memora reaches it through OPENAI_BASE_URL.
EOF
    exit 1
  fi
fi

NUMACTL=()
if command -v numactl >/dev/null 2>&1; then
  NUMACTL=(numactl "--cpunodebind=${NUMA_NODE}" "--membind=${NUMA_NODE}")
  echo "pinning to NUMA node ${NUMA_NODE} (cpu + memory)"
else
  echo "WARNING: numactl not installed - the model will straddle both sockets and lose"
  echo "         a large fraction of its speed. Install it: sudo apt install numactl"
fi

echo "model    : $MODEL"
echo "threads  : $THREADS"
echo "slots    : $PARALLEL x ${CTX_PER_SLOT} ctx = ${CTX} total"
echo "batch    : $BATCH / ubatch $UBATCH"
echo "endpoint : http://127.0.0.1:${PORT}/v1"
echo

# OpenMP must not oversubscribe: llama.cpp manages its own thread pool, and letting OpenMP
# also spawn per-core teams causes heavy contention on a 64-core box.
export OMP_NUM_THREADS="$THREADS"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# One thread per physical core. Hyperthreading is OFF on this CPU so there are no siblings
# to avoid, but oversubscribing past the node's 32 cores still costs throughput.
#
# --mlock pins weights in RAM: without it a 22 GB model can be partially paged out and
# decode (which touches weights every token) collapses.
# --cont-batching is what makes concurrency pay -- weights are read once and reused across
# all in-flight requests, so aggregate decode scales with slot count.
exec "${NUMACTL[@]}" "$SERVER_BIN" \
  --model "$MODEL" \
  --host 127.0.0.1 --port "$PORT" \
  --threads "$THREADS" \
  --threads-batch "$THREADS" \
  --ctx-size "$CTX" \
  --parallel "$PARALLEL" \
  --batch-size "$BATCH" \
  --ubatch-size "$UBATCH" \
  --cont-batching \
  --mlock \
  --numa numactl \
  --jinja \
  "$@"

# NOTE on Qwen3 / Qwen3.6: these ship a THINKING mode. For Stage 3 extraction -- a
# structured-JSON task capped at STAGE_3_MAX_TOKENS = 500 -- a reasoning trace will eat the
# entire output budget, truncate the JSON, and cost many times the tokens per call. Disable
# it. With --jinja the template accepts:
#     --chat-template-kwargs '{"enable_thinking":false}'
# Pass extra args straight through to this script; they are forwarded above.
