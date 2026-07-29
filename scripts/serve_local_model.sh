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
CTX="${CTX:-8192}"
PARALLEL="${PARALLEL:-8}"     # concurrent slots; batching is what makes decode fast
SERVER_BIN="${SERVER_BIN:-llama-server}"

if ! command -v "$SERVER_BIN" >/dev/null 2>&1; then
  cat >&2 <<EOF
ERROR: '$SERVER_BIN' not on PATH.

Build llama.cpp with AMX + AVX-512 (this CPU has amx_bf16/amx_tile/amx_int8):
    git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
    cmake --build build -j 64
    export PATH="\$PWD/build/bin:\$PATH"

Or set SERVER_BIN to an OpenAI-compatible server of your choice (vLLM CPU,
ipex-llm, OpenVINO GenAI). Anything exposing /v1/chat/completions works --
Memora reaches it through OPENAI_BASE_URL.
EOF
  exit 1
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
echo "context  : $CTX   slots: $PARALLEL"
echo "endpoint : http://127.0.0.1:${PORT}/v1"
echo

# One thread per physical core; this CPU has hyperthreading OFF so there is no sibling
# contention to worry about, but oversubscribing past the node's 32 cores still hurts.
exec "${NUMACTL[@]}" "$SERVER_BIN" \
  --model "$MODEL" \
  --host 127.0.0.1 --port "$PORT" \
  --threads "$THREADS" \
  --threads-batch "$THREADS" \
  --ctx-size "$CTX" \
  --parallel "$PARALLEL" \
  --cont-batching \
  --mlock \
  --numa numactl
