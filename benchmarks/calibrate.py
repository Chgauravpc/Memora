"""
Measure whether a local model degrades extraction versus a hosted reference.

This exists because "llama.cpp / quantisation loses quality" is a reasonable worry that
should not be settled by argument. Extraction quality feeds straight into the benchmark
score -- worse extraction means worse memories means lower recall -- so if the local model
diverges from the reference, Memora would score badly for reasons that have nothing to do
with Memora. This turns that risk into a number, cheaply, before committing hours.

How it works: the same N turns are pushed through Memora's REAL extraction pipeline twice,
once per backend, each in its own subprocess (src/config.py freezes provider/model into
module constants at import time, so two configs cannot coexist in one process). Extraction
needs no Redis or Qdrant, so this is cheap and side-effect free.

    # reference vs local
    python -m benchmarks.calibrate --turns 40 \
        --ref-provider groq --ref-model llama-3.3-70b-versatile \
        --alt-provider openai --alt-model qwen3.6-35b-a3b \
        --alt-base-url http://127.0.0.1:8080/v1

Interpretation: type agreement and key overlap in the 80%+ range means the local model is
a fair substitute. Much below that and either raise the quant (Q4 -> Q8), fix the chat
template, or keep extraction on the API.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional

from .paths import RESULTS_DIR, REPO_ROOT, ensure_dirs


# --------------------------------------------------------------- subprocess side

def _extract_worker(turns_path: Path, out_path: Path) -> int:
    """Runs inside the subprocess: extract from each turn, dump JSON."""
    from src.extractor import MemoryExtractor
    from src.config import LLM_PROVIDER, LLM_EXTRACTION_MODEL, STAGE_3_ENABLED

    turns: List[str] = json.loads(turns_path.read_text(encoding="utf-8"))
    extractor = MemoryExtractor()

    records = []
    stage3_calls = 0
    llm = getattr(extractor, "llm_extractor", None)
    if llm is not None and hasattr(llm, "extract"):
        original = llm.extract

        def counted(*a, **kw):
            nonlocal stage3_calls
            stage3_calls += 1
            return original(*a, **kw)

        llm.extract = counted  # type: ignore[method-assign]

    for i, text in enumerate(turns, 1):
        try:
            mems = extractor.extract(text, i)
        except Exception as exc:  # noqa: BLE001
            mems = []
            print(f"turn {i} failed: {exc}", file=sys.stderr)
        records.append([
            {"type": m.get("type"), "key": m.get("key"), "value": m.get("value"),
             "confidence": m.get("confidence")}
            for m in mems
        ])

    out_path.write_text(json.dumps({
        "provider": LLM_PROVIDER,
        "model": LLM_EXTRACTION_MODEL,
        "stage3_enabled": STAGE_3_ENABLED,
        "stage3_calls": stage3_calls,
        "base_url": os.getenv("OPENAI_BASE_URL", ""),
        "records": records,
    }, indent=2), encoding="utf-8")
    return 0


# ------------------------------------------------------------------- comparison

def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()


def compare(ref: Dict[str, Any], alt: Dict[str, Any], threshold: float = 0.6) -> Dict:
    ref_recs, alt_recs = ref["records"], alt["records"]
    n = min(len(ref_recs), len(alt_recs))

    ref_total = sum(len(r) for r in ref_recs[:n])
    alt_total = sum(len(r) for r in alt_recs[:n])

    matched = 0
    type_agree = 0
    value_sim_sum = 0.0
    turns_same_count = 0
    turns_ref_found_alt_empty = 0

    for i in range(n):
        r_list, a_list = ref_recs[i], alt_recs[i]
        if len(r_list) == len(a_list):
            turns_same_count += 1
        if r_list and not a_list:
            turns_ref_found_alt_empty += 1

        used = set()
        for r in r_list:
            best, best_j = 0.0, None
            for j, a in enumerate(a_list):
                if j in used:
                    continue
                s = _sim(f"{r.get('key')} {r.get('value')}", f"{a.get('key')} {a.get('value')}")
                if s > best:
                    best, best_j = s, j
            if best_j is not None and best >= threshold:
                used.add(best_j)
                matched += 1
                value_sim_sum += best
                if r.get("type") == a_list[best_j].get("type"):
                    type_agree += 1

    return {
        "turns_compared": n,
        "ref": {"model": ref["model"], "memories": ref_total,
                "stage3_calls": ref.get("stage3_calls", 0)},
        "alt": {"model": alt["model"], "memories": alt_total,
                "stage3_calls": alt.get("stage3_calls", 0)},
        "matched_memories": matched,
        # Of the reference's memories, how many did the local model also find?
        "recall_vs_ref": round(matched / ref_total, 4) if ref_total else None,
        # Of the local model's memories, how many correspond to a reference one?
        "precision_vs_ref": round(matched / alt_total, 4) if alt_total else None,
        "type_agreement": round(type_agree / matched, 4) if matched else None,
        "mean_content_similarity": round(value_sim_sum / matched, 4) if matched else None,
        "turns_with_same_count": turns_same_count,
        "turns_ref_found_alt_empty": turns_ref_found_alt_empty,
        "extraction_volume_ratio": round(alt_total / ref_total, 4) if ref_total else None,
    }


def render(result: Dict) -> str:
    L = []
    L.append("=" * 68)
    L.append("Extraction calibration: local model vs hosted reference")
    L.append("=" * 68)
    L.append(f"turns compared : {result['turns_compared']}")
    L.append(f"reference      : {result['ref']['model']}  "
             f"-> {result['ref']['memories']} memories "
             f"({result['ref']['stage3_calls']} LLM calls)")
    L.append(f"alternative    : {result['alt']['model']}  "
             f"-> {result['alt']['memories']} memories "
             f"({result['alt']['stage3_calls']} LLM calls)")
    L.append("")

    def pct(v):
        return "n/a" if v is None else f"{v:.1%}"

    L.append(f"  recall vs reference     : {pct(result['recall_vs_ref'])}"
             "   (reference memories the local model also found)")
    L.append(f"  precision vs reference  : {pct(result['precision_vs_ref'])}"
             "   (local memories matching a reference one)")
    L.append(f"  type agreement          : {pct(result['type_agreement'])}")
    L.append(f"  mean content similarity : {pct(result['mean_content_similarity'])}")
    L.append(f"  extraction volume ratio : {result['extraction_volume_ratio']}"
             "        (1.0 = same amount extracted)")
    L.append(f"  turns where ref found something and local found nothing: "
             f"{result['turns_ref_found_alt_empty']}")
    L.append("")

    r = result["recall_vs_ref"] or 0
    t = result["type_agreement"] or 0
    if r >= 0.8 and t >= 0.8:
        L.append("  VERDICT: the local model is a fair substitute for extraction.")
    elif r >= 0.6:
        L.append("  VERDICT: noticeable divergence. Try a higher quant (Q4 -> Q8), verify")
        L.append("  the chat template, and confirm thinking mode is disabled before")
        L.append("  trusting a full run.")
    else:
        L.append("  VERDICT: substantial divergence. Extraction quality would confound the")
        L.append("  benchmark - the score would reflect the local model, not Memora.")
        L.append("  Keep extraction on the API, or fix the local setup first.")
    L.append("")
    L.append("  Note: this measures AGREEMENT with the reference, not correctness. A")
    L.append("  divergent local model is not necessarily worse - but it is different,")
    L.append("  and differences must be disclosed alongside any published number.")
    return "\n".join(L)


# ------------------------------------------------------------------------ driver

def _run_backend(label: str, turns_path: Path, out_path: Path, provider: str,
                 model: str, base_url: Optional[str]) -> Optional[Dict]:
    env = os.environ.copy()
    env["LLM_PROVIDER"] = provider
    env["LLM_EXTRACTION_MODEL"] = model
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["LOG_LEVEL"] = "WARNING"
    if base_url:
        env["OPENAI_BASE_URL"] = base_url
        env.setdefault("OPENAI_API_KEY", "local")
    else:
        env.pop("OPENAI_BASE_URL", None)

    print(f"  [{label}] {provider}/{model}" + (f" @ {base_url}" if base_url else ""))
    proc = subprocess.run(
        [sys.executable, "-m", "benchmarks.calibrate", "--_worker",
         "--turns-file", str(turns_path), "--out", str(out_path)],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True,
    )
    if proc.returncode != 0 or not out_path.exists():
        print(f"  [{label}] FAILED (exit {proc.returncode})")
        print((proc.stderr or proc.stdout)[-1500:])
        return None
    return json.loads(out_path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compare extraction between a local model and a hosted reference")
    ap.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--turns-file", type=Path, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--out", type=Path, default=None)

    ap.add_argument("--turns", type=int, default=40,
                    help="how many LoCoMo turns to sample (default 40)")
    ap.add_argument("--ref-provider", default="groq")
    ap.add_argument("--ref-model", default="llama-3.3-70b-versatile")
    ap.add_argument("--ref-base-url", default=None)
    ap.add_argument("--alt-provider", default="openai")
    ap.add_argument("--alt-model", default="qwen3.6-35b-a3b")
    ap.add_argument("--alt-base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--threshold", type=float, default=0.6,
                    help="content similarity to count as the same memory (default 0.6)")
    args = ap.parse_args()

    if args._worker:
        if not args.turns_file or not args.out:
            print("worker mode needs --turns-file and --out")
            return 2
        return _extract_worker(args.turns_file, args.out)

    ensure_dirs()
    out_dir = RESULTS_DIR / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Real benchmark text, so the comparison reflects the actual domain rather than
    # synthetic phrasing the regexes were tuned on.
    try:
        from .dataset import load_conversations
        convs = load_conversations()
    except (FileNotFoundError, ValueError) as exc:
        print(exc)
        return 1

    turns: List[str] = []
    for conv in convs:
        for t in conv.turns:
            turns.append(t.render())
            if len(turns) >= args.turns:
                break
        if len(turns) >= args.turns:
            break

    turns_path = out_dir / "turns.json"
    turns_path.write_text(json.dumps(turns, indent=2), encoding="utf-8")
    print(f"calibrating on {len(turns)} real LoCoMo turns\n")

    ref = _run_backend("reference", turns_path, out_dir / "ref.json",
                       args.ref_provider, args.ref_model, args.ref_base_url)
    alt = _run_backend("alternative", turns_path, out_dir / "alt.json",
                       args.alt_provider, args.alt_model, args.alt_base_url)
    if not ref or not alt:
        print("\nboth backends must succeed to compare")
        return 1

    result = compare(ref, alt, args.threshold)
    text = render(result)
    print()
    print(text)

    out = args.out or (out_dir / "calibration.json")
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    out.with_suffix(".txt").write_text(text, encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
