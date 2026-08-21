#!/usr/bin/env bash
#
# Build llama.cpp with AMX enabled, into .cache/llama.cpp inside this repo.
#
# WHY BUILD FROM SOURCE: this CPU (Xeon Gold 6530, Emerald Rapids) has AMX
# (amx_tile / amx_int8 / amx_bf16). AMX is roughly an order of magnitude more matmul
# throughput than AVX-512, and prefill -- which dominates this benchmark at ~10.8M tokens
# -- is compute bound. Prebuilt llama.cpp binaries, distro packages, and Ollama's bundled
# runtime are compiled for a generic baseline and DO NOT have these kernels. Building
# yourself is the single largest speed lever available here.
#
# The AMX code path additionally REQUIRES AVX512-VNNI to be enabled, which is easy to miss
# and silently leaves AMX out of the build.
#
# Usage:
#   bash scripts/build_llama_cpp.sh
#   PATH="$PWD/.cache/llama.cpp/build/bin:$PATH"    # then serve_local_model.sh finds it

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SRC="$REPO_ROOT/.cache/llama.cpp"
JOBS="${JOBS:-$(nproc)}"

echo "======================================================================"
echo "Building llama.cpp with AMX"
echo "======================================================================"

# ------------------------------------------------------------ verify the CPU has AMX
echo "--- CPU capability check ---"
MISSING=()
for flag in amx_tile amx_int8 amx_bf16 avx512_vnni avx512_bf16; do
  if grep -qm1 "\b${flag}\b" /proc/cpuinfo; then
    echo "  present: $flag"
  else
    echo "  MISSING: $flag"
    MISSING+=("$flag")
  fi
done
if [[ ${#MISSING[@]} -gt 0 ]]; then
  echo
  echo "WARNING: ${MISSING[*]} not reported by /proc/cpuinfo."
  echo "The build will still work but AMX kernels may be unused."
  echo "On a VM, check that the hypervisor exposes AMX to the guest."
fi

# ------------------------------------------------------------ toolchain
echo
echo "--- toolchain ---"
for tool in cmake git; do
  command -v "$tool" >/dev/null 2>&1 || { echo "ERROR: $tool not found" >&2; exit 1; }
done
# AMX intrinsics need a reasonably modern compiler. GCC 11+ / Clang 14+ are safe.
CC_BIN="${CC:-gcc}"
if command -v "$CC_BIN" >/dev/null 2>&1; then
  echo "  compiler: $($CC_BIN --version | head -1)"
else
  echo "ERROR: no C compiler ($CC_BIN)" >&2; exit 1
fi
echo "  jobs: $JOBS"

# ------------------------------------------------------------ source
echo
echo "--- source ---"
mkdir -p "$(dirname "$SRC")"
if [[ -d "$SRC/.git" ]]; then
  echo "updating $SRC"
  git -C "$SRC" pull --ff-only || echo "  (pull failed; building existing checkout)"
else
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$SRC"
fi

# ------------------------------------------------------------ configure
echo
echo "--- configure ---"
# GGML_NATIVE=ON lets the compiler target this exact CPU (-march=native).
# The AMX_* flags are the ones that matter; AVX512_VNNI is a hard prerequisite for them.
cmake -S "$SRC" -B "$SRC/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON \
  -DGGML_AVX512=ON \
  -DGGML_AVX512_VNNI=ON \
  -DGGML_AVX512_BF16=ON \
  -DGGML_AMX_TILE=ON \
  -DGGML_AMX_INT8=ON \
  -DGGML_AMX_BF16=ON \
  -DLLAMA_CURL=ON \
  -DGGML_OPENMP=ON

echo
echo "--- build ---"
cmake --build "$SRC/build" --config Release -j "$JOBS"

BIN="$SRC/build/bin"
echo
echo "======================================================================"
if [[ -x "$BIN/llama-server" ]]; then
  echo "built: $BIN/llama-server"
  echo
  echo "Confirm AMX actually landed in the binary:"
  echo "  $BIN/llama-server --version 2>&1 | head -20"
  echo "  (or check for AMX in the startup 'system_info' line when it loads a model)"
  echo
  echo "Add to PATH:"
  echo "  export PATH=\"$BIN:\$PATH\""
  echo
  echo "Then serve a model:"
  echo "  scripts/serve_local_model.sh /path/to/model.gguf"
else
  echo "ERROR: llama-server was not produced. Check the build output above." >&2
  exit 1
fi
echo "======================================================================"
