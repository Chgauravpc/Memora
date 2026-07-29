"""
Projection for running the extraction/reader/judge LLM **locally on CPU** instead of
against a hosted API.

Target hardware (from `lscpu`): 2x Intel Xeon Gold 6530, 32 cores/socket, hyperthreading
OFF => 64 physical cores, 2 NUMA nodes, 320 MiB L3 total, AMX (amx_bf16/amx_tile/amx_int8)
plus avx512_bf16 / avx512_fp16.

The model here is a first-principles one, not a measurement. Two regimes:

  DECODE  is memory-bandwidth bound. Generating one token requires reading every ACTIVE
          weight from DRAM, so single-stream tok/s ~= achieved_bandwidth / active_bytes.
          This is why a Mixture-of-Experts model with 3B active parameters decodes far
          faster than a dense 30B, and why quantisation buys speed as well as RAM.

          Batching changes this decisively: with B concurrent requests the weights are
          read ONCE and reused for all B tokens, so aggregate decode scales nearly
          linearly with B until it becomes compute-bound. Running the benchmark's workers
          against a batching server is worth several times the throughput of one stream.

  PREFILL is compute bound, and this is where AMX matters. Plain llama.cpp on AVX-512
          leaves most of Emerald Rapids' matmul throughput unused; an AMX-aware runtime
          (ipex-llm, OpenVINO GenAI, vLLM CPU) is several times faster. Prefill dominates
          this benchmark because the reader ships ~3k tokens of memory context per
          question.

Bandwidth basis: DDR5-4800 x 8 channels/socket = 4800 MT/s * 8 B * 8 = 307 GB/s
theoretical per socket; ~200 GB/s is a realistic achieved figure once >=16 cores are
pulling. Sockets have independent memory controllers, which is the whole argument for
pinning the model to one socket and the benchmark to the other.

Usage:
    python -m benchmarks.local_estimate --list
    python -m benchmarks.local_estimate --model qwen3-30b-a3b --cores 22 --batch 8
    python -m benchmarks.local_estimate --model llama-3.1-8b --split hybrid
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Optional

from .estimate import (JUDGE_IN, JUDGE_OUT, READER_IN, READER_OUT, STAGE3_IN,
                       STAGE3_OUT, load_shape)

# ---------------------------------------------------------------- hardware defaults
SOCKETS = 2
CORES_PER_SOCKET = 32
# Achieved (not theoretical) DRAM bandwidth for one socket with >=16 cores active.
SOCKET_BANDWIDTH_GBS = 200.0
# Fraction of bandwidth a real inference runtime converts into useful weight streaming.
DECODE_EFFICIENCY = 0.65
# Sustained matmul throughput per core, TFLOPS. AVX-512 vs AMX is the big lever.
TFLOPS_PER_CORE = {"llamacpp": 0.22, "amx": 0.85}


@dataclass
class LocalModel:
    key: str
    name: str
    total_b: float          # total parameters, billions
    active_b: float         # active per token (== total for dense models)
    bytes_per_param: float  # quantised weight size
    quant: str
    note: str

    @property
    def weights_gb(self) -> float:
        return self.total_b * self.bytes_per_param

    @property
    def active_gb(self) -> float:
        return self.active_b * self.bytes_per_param

    @property
    def is_moe(self) -> bool:
        return self.active_b < self.total_b * 0.9


# bytes_per_param 0.60 ~= Q4_K_M including quantisation overhead.
Q4 = 0.60
MODELS: Dict[str, LocalModel] = {
    "llama-3.1-8b": LocalModel(
        "llama-3.1-8b", "Llama-3.1-8B-Instruct", 8.0, 8.0, Q4, "Q4_K_M",
        "Fastest dense option. JSON adherence weaker than Qwen at the same size."),
    "qwen2.5-7b": LocalModel(
        "qwen2.5-7b", "Qwen2.5-7B-Instruct", 7.6, 7.6, Q4, "Q4_K_M",
        "Best small-model JSON/schema following. Safe default if RAM is tight."),
    "qwen2.5-14b": LocalModel(
        "qwen2.5-14b", "Qwen2.5-14B-Instruct", 14.8, 14.8, Q4, "Q4_K_M",
        "Noticeably better extraction than 7-8B, still tolerable decode speed."),
    "qwen3-30b-a3b": LocalModel(
        "qwen3-30b-a3b", "Qwen3-30B-A3B-Instruct (MoE)", 30.5, 3.3, Q4, "Q4_K_M",
        "RECOMMENDED. 30B-class quality at better-than-8B decode speed: only ~3.3B "
        "parameters are active per token, so ~2 GB is read per token instead of ~18 GB."),
    "qwen2.5-32b": LocalModel(
        "qwen2.5-32b", "Qwen2.5-32B-Instruct", 32.5, 32.5, Q4, "Q4_K_M",
        "Best dense quality within reach, but decode is slow; only sensible for the "
        "reader/judge roles, not for thousands of extraction calls."),
    "llama-3.3-70b": LocalModel(
        "llama-3.3-70b", "Llama-3.3-70B-Instruct", 70.6, 70.6, Q4, "Q4_K_M",
        "Matches the model currently configured on Groq, so it keeps the system under "
        "test identical - but decode is ~4 tok/s and extraction alone would take days."),
}


def throughput(model: LocalModel, cores: int, batch: int, runtime: str,
               bandwidth: float) -> tuple[float, float]:
    """(prefill tok/s, decode tok/s) aggregate across `batch` concurrent requests."""
    # A single socket's bandwidth is shared; more cores than one socket has does not add
    # bandwidth, it just crosses NUMA and usually makes things worse.
    effective_cores = min(cores, CORES_PER_SOCKET)
    compute_tflops = TFLOPS_PER_CORE[runtime] * effective_cores

    # Decode: bandwidth-bound per stream, but weights are read once per batch.
    single_stream = bandwidth * DECODE_EFFICIENCY / model.active_gb
    flops_per_token = 2 * model.active_b  # GFLOP
    compute_ceiling = compute_tflops * 1000.0 / flops_per_token
    decode = min(single_stream * batch, compute_ceiling)

    # Prefill: compute-bound; batching mostly improves arithmetic intensity already.
    prefill = compute_ceiling
    return prefill, decode


def hours(tokens: float, rate: float) -> float:
    return tokens / rate / 3600 if rate > 0 else float("inf")


def run(model_key: str, cores: int, batch: int, runtime: str, bandwidth: float,
        split: str, escalation: float) -> None:
    convs, turns, questions, real = load_shape()
    model = MODELS[model_key]

    stage3 = turns * escalation
    s3_pre, s3_dec = stage3 * STAGE3_IN, stage3 * STAGE3_OUT
    rd_pre, rd_dec = questions * READER_IN, questions * READER_OUT
    jd_pre, jd_dec = questions * JUDGE_IN, questions * JUDGE_OUT

    if split == "all":
        pre, dec, roles = s3_pre + rd_pre + jd_pre, s3_dec + rd_dec + jd_dec, \
            "extraction + reader + judge"
    elif split == "hybrid":
        pre, dec, roles = s3_pre, s3_dec, "extraction only (reader+judge stay on API)"
    else:  # extraction+reader
        pre, dec, roles = s3_pre + rd_pre, s3_dec + rd_dec, \
            "extraction + reader (judge stays on API)"

    prefill_rate, decode_rate = throughput(model, cores, batch, runtime, bandwidth)
    h_pre, h_dec = hours(pre, prefill_rate), hours(dec, decode_rate)
    total = h_pre + h_dec

    print("=" * 70)
    print(f"Local CPU inference estimate - {model.name}")
    print("=" * 70)
    if not real:
        print("!! dataset absent; using approximate shape. Run benchmarks.download.\n")
    print(f"workload      : {roles}")
    print(f"shape         : {turns:,} turns, {questions:,} questions, "
          f"escalation {escalation:.0%}")
    print(f"hardware      : {cores} cores (capped at {min(cores, CORES_PER_SOCKET)} "
          f"= 1 socket), {bandwidth:.0f} GB/s, runtime={runtime}, batch={batch}")
    print()
    print(f"model         : {model.total_b:.1f}B total"
          + (f", {model.active_b:.1f}B active (MoE)" if model.is_moe else " (dense)"))
    print(f"weights       : {model.weights_gb:.1f} GB {model.quant}")
    print(f"read/token    : {model.active_gb:.2f} GB")
    print()
    print("throughput (modelled, not measured)")
    print("-" * 70)
    print(f"  prefill     : {prefill_rate:>8.0f} tok/s")
    print(f"  decode      : {decode_rate:>8.0f} tok/s aggregate over {batch} streams")
    print(f"                ({decode_rate / batch:.0f} tok/s per stream)")
    print()
    print("time")
    print("-" * 70)
    print(f"  prefill     : {pre:>12,.0f} tokens -> {h_pre:>6.1f} h")
    print(f"  decode      : {dec:>12,.0f} tokens -> {h_dec:>6.1f} h")
    print(f"  TOTAL       : {total:>6.1f} h" +
          (f"  ({total / 24:.1f} days)" if total > 24 else ""))
    print()
    print("RAM for this role")
    print("-" * 70)
    kv = 0.4 * batch
    print(f"  weights {model.weights_gb:.1f} + KV ~{kv:.1f} (batch {batch}) + "
          f"runtime ~2 = ~{model.weights_gb + kv + 2:.0f} GB")
    print(f"  plus the benchmark itself ~13 GB (10 workers x ~1 GB, Redis, Qdrant)")
    print(f"  => ~{model.weights_gb + kv + 15:.0f} GB total")
    print()
    print(f"note: {model.note}")


def show_table(cores: int, batch: int, runtime: str, bandwidth: float,
               escalation: float) -> None:
    convs, turns, questions, real = load_shape()
    stage3 = turns * escalation
    s3_pre, s3_dec = stage3 * STAGE3_IN, stage3 * STAGE3_OUT
    all_pre = s3_pre + questions * READER_IN + questions * JUDGE_IN
    all_dec = s3_dec + questions * READER_OUT + questions * JUDGE_OUT

    print(f"Modelled on {min(cores, CORES_PER_SOCKET)} cores / {bandwidth:.0f} GB/s / "
          f"runtime={runtime} / batch={batch}")
    print(f"Workload: {turns:,} turns at {escalation:.0%} escalation, "
          f"{questions:,} questions")
    print()
    print(f"{'model':<26}{'RAM':>7}{'dec t/s':>9}{'pre t/s':>9}"
          f"{'extract':>10}{'all-local':>11}")
    print("-" * 72)
    for m in MODELS.values():
        pre_r, dec_r = throughput(m, cores, batch, runtime, bandwidth)
        h_extract = hours(s3_pre, pre_r) + hours(s3_dec, dec_r)
        h_all = hours(all_pre, pre_r) + hours(all_dec, dec_r)
        ram = m.weights_gb + 0.4 * batch + 2
        label = m.key + (" *" if m.key == "qwen3-30b-a3b" else "")
        print(f"{label:<26}{ram:>6.0f}G{dec_r:>9.0f}{pre_r:>9.0f}"
              f"{h_extract:>9.1f}h{h_all:>10.1f}h")
    print()
    print("extract   = Stage 3 extraction only (recommended local role)")
    print("all-local = extraction + reader + judge on CPU")
    print("* recommended")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Estimate local CPU inference for the LoCoMo benchmark")
    ap.add_argument("--model", choices=sorted(MODELS), default=None)
    ap.add_argument("--list", action="store_true", help="compare all models")
    ap.add_argument("--cores", type=int, default=22,
                    help="cores for inference (default 22; capped at one socket = 32)")
    ap.add_argument("--batch", type=int, default=8,
                    help="concurrent requests the server batches (default 8)")
    ap.add_argument("--runtime", choices=sorted(TFLOPS_PER_CORE), default="llamacpp",
                    help="'llamacpp' = AVX-512 only; 'amx' = AMX-aware runtime "
                         "(ipex-llm / OpenVINO / vLLM-CPU), several times faster prefill")
    ap.add_argument("--bandwidth", type=float, default=SOCKET_BANDWIDTH_GBS,
                    help="achieved DRAM GB/s for one socket (default 200)")
    ap.add_argument("--split", choices=["all", "hybrid", "extraction+reader"],
                    default="hybrid",
                    help="which roles run locally (default hybrid: extraction only)")
    ap.add_argument("--escalation", type=float, default=0.70)
    args = ap.parse_args()

    if args.list or not args.model:
        show_table(args.cores, args.batch, args.runtime, args.bandwidth, args.escalation)
        return 0

    run(args.model, args.cores, args.batch, args.runtime, args.bandwidth,
        args.split, args.escalation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
