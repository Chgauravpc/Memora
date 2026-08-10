"""
Self-test for the conversation architecture. No Redis, no Qdrant, no API keys, no network.

    python -m benchmarks.selftest

Exists because the expensive checks (a benchmark run) take minutes and cost money, while
most of what can break here is pure logic: dedup identity, fusion, rendering order, query
intent, date parsing. Those are worth catching in two seconds.

The repo has no pytest suite and `.gitignore` excludes `test_*.py`, so this is a plain
module with asserts rather than a collected test file.

Each check reloads `src.*` under a chosen MEMORA_PROFILE, because src/config.py freezes
environment into module constants at import time -- two profiles cannot coexist in one
process without a reload.
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import Dict, List

FAILURES: List[str] = []


def _reload(profile: str, **env):
    os.environ["MEMORA_PROFILE"] = profile
    for k, v in env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(v)
    for mod in [m for m in sys.modules if m.startswith("src")]:
        del sys.modules[mod]
    return importlib.import_module("src.retriever")


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  [ OK ] {name}")
    else:
        print(f"  [FAIL] {name}" + (f" - {detail}" if detail else ""))
        FAILURES.append(name)


MEMS: List[Dict] = [
    {"memory_id": "m1", "type": "event", "key": "museum_visit",
     "value": "went to the art museum", "confidence": 0.9, "turn_number": 40,
     "speaker": "Melanie", "event_date": "8 May, 2023", "event_ts": 1683504000.0,
     "retrieval_score": 0.9,
     "source_text": "[8 May, 2023] Melanie: I finally went to the Ravensbourne art museum"},
    {"memory_id": "m2", "type": "fact", "key": "hobby", "value": "started guitar lessons",
     "confidence": 0.8, "turn_number": 120, "speaker": "Caroline",
     "event_date": "12 June, 2023", "event_ts": 1686528000.0, "retrieval_score": 0.7,
     "source_text": "[12 June, 2023] Caroline: I started guitar lessons this week"},
    {"memory_id": "m3", "type": "preference", "key": "coffee", "value": "prefers espresso",
     "confidence": 0.7, "turn_number": 5, "speaker": "Melanie",
     "event_date": "", "event_ts": 0.0, "retrieval_score": 0.3, "source_text": ""},
    # Type outside MEMORY_TYPES: used to be dropped from the prompt while still counted
    # as retrieved.
    {"memory_id": "m4", "type": "goal", "key": "plan", "value": "wants to move abroad",
     "confidence": 0.6, "turn_number": 200, "speaker": "Caroline",
     "event_date": "1 July, 2023", "event_ts": 1688169600.0, "retrieval_score": 0.2,
     "source_text": ""},
]


def test_lexical() -> None:
    print("\nlexical index (BM25 + RRF)")
    from src.lexical_index import (BM25Index, normalize_scores,
                                   reciprocal_rank_fusion)

    idx = BM25Index().build(MEMS)
    check("indexes every memory", len(idx) == len(MEMS))

    hits = idx.search("Ravensbourne", limit=3)
    check("rare proper noun ranks its document first",
          bool(hits) and hits[0][0] == "m1",
          "this is the failure mode dense retrieval has and BM25 fixes")

    check("stopword-only query returns nothing", idx.search("the and of") == [])
    check("out-of-vocabulary query returns nothing",
          idx.search("quantum chromodynamics") == [])
    check("empty index does not raise", BM25Index().build([]).search("x") == [])

    fused = reciprocal_rank_fusion([(["a", "b", "c"], 1.0), (["b", "a", "d"], 1.0)], k=60)
    check("agreed-on item beats single-channel item", fused["b"] > fused["c"])
    check("normalisation puts the top hit at 1.0",
          abs(max(normalize_scores(fused).values()) - 1.0) < 1e-9)
    check("normalising an empty dict is safe", normalize_scores({}) == {})


def test_dedup_identity() -> None:
    print("\ndedup identity")
    _reload("conversation")
    from src.redis_store import build_dedup_key

    mel_seattle = {"type": "entity", "key": "location", "value": "Seattle", "speaker": "Melanie"}
    car_boston = {"type": "entity", "key": "location", "value": "Boston", "speaker": "Caroline"}
    mel_portland = {"type": "entity", "key": "location", "value": "Portland", "speaker": "Melanie"}
    mel_seattle_again = {"type": "entity", "key": "location", "value": " seattle ", "speaker": "Melanie"}

    keys = {build_dedup_key(m) for m in (mel_seattle, car_boston, mel_portland)}
    check("different speakers and values stay distinct", len(keys) == 3,
          "legacy collapsed all three onto entity:location")
    check("verbatim repeat still dedups",
          build_dedup_key(mel_seattle) == build_dedup_key(mel_seattle_again))

    _reload("legacy")
    from src.redis_store import build_dedup_key as legacy
    check("legacy behaviour preserved for A/B",
          legacy(mel_seattle) == legacy(car_boston) == legacy(mel_portland))


def test_context_rendering() -> None:
    print("\ncontext rendering")
    r = _reload("conversation")
    out = r.MemoryRetriever.format_memories_for_prompt(
        object.__new__(r.MemoryRetriever), MEMS)

    check("renders a timeline", "TIMELINE" in out)
    check("orders by event date",
          out.index("8 May, 2023") < out.index("12 June, 2023"))
    check("attributes speakers", "Melanie" in out and "Caroline" in out)
    check("keeps undated memories", "UNDATED" in out and "espresso" in out)
    check("attaches evidence to top-ranked memories", "Ravensbourne" in out)
    check("unrecognised type still reaches the prompt", "move abroad" in out,
          "these used to be silently dropped while counted as retrieved")
    check("output is ASCII-safe", out == out.encode("ascii", "ignore").decode())

    r = _reload("legacy")
    old = r.MemoryRetriever.format_memories_for_prompt(
        object.__new__(r.MemoryRetriever), MEMS)
    check("legacy stays type-grouped", "=== EVENT ===" in old and "TIMELINE" not in old)
    check("legacy gains no speaker attribution", "Melanie" not in old)


def test_query_intent() -> None:
    print("\nquery-aware retrieval")
    r = _reload("conversation")

    check("detects 'when'", r.has_temporal_intent("When did Melanie visit?"))
    check("detects 'how long ago'", r.has_temporal_intent("How long ago was that?"))
    check("ignores non-temporal questions",
          not r.has_temporal_intent("What does she prefer to drink?"))

    ents = r.query_entities("When did Melanie visit the Ravensbourne museum?")
    check("finds named entities", {"melanie", "ravensbourne"} <= ents)
    check("ignores the sentence-initial word", "when" not in ents)


def test_query_dates() -> None:
    print("\ndate-aware scoring")
    r = _reload("conversation")

    d = r.query_dates("What did she do in May 2023?")
    check("finds year and month named in the query", "2023" in d and "may" in d)
    check("no dates in an undated question", r.query_dates("What is her job?") == set())

    may23 = r.query_dates("anything from May 2023?")
    check("matches a memory from the same month",
          r.date_overlap(may23, "8 May, 2023"))
    check("rejects a different month and year",
          not r.date_overlap(may23, "3 March, 2019"))
    check("year alone is enough", r.date_overlap({"2023"}, "12 June, 2023"))
    check("empty inputs are safe",
          not r.date_overlap(set(), "8 May, 2023") and not r.date_overlap({"2023"}, ""))


def test_always_on_floor() -> None:
    print("\nalways-on relevance floor")
    _reload("conversation")
    from src.config import (ALWAYS_ON_SEMANTIC_FLOOR as conv_floor,
                            RANKING_WEIGHTS_5_SIGNAL as w)
    # An always-on memory with zero relevance earns floor x semantic weight. Legacy paid
    # 0.5 x 0.55 = 0.275, enough to outrank a real single-channel hit normalising near
    # 0.28 -- the crowding the type-weight change was meant to remove.
    contribution = conv_floor * w["semantic"]
    check("irrelevant always-on memories cannot outrank real hits",
          contribution < 0.10, f"contribution {contribution:.3f}")

    _reload("legacy")
    from src.config import ALWAYS_ON_SEMANTIC_FLOOR as legacy_floor
    check("legacy floor preserved for A/B", legacy_floor == 0.5)


def test_ranking_profiles() -> None:
    print("\nranking weights")
    _reload("conversation")
    from src.config import RANKING_WEIGHTS_5_SIGNAL as conv
    check("relevance outweighs type", conv["semantic"] > conv["type"])
    check("weights sum to 1.0", abs(sum(conv.values()) - 1.0) < 1e-9,
          f"got {sum(conv.values())}")

    _reload("legacy")
    from src.config import RANKING_WEIGHTS_5_SIGNAL as leg
    check("legacy weights unchanged",
          leg == {"semantic": 0.30, "type": 0.40, "recency": 0.10,
                  "frequency": 0.05, "confidence": 0.15})

    _reload("conversation", RANK_W_SEMANTIC="0.7")
    from src.config import RANKING_WEIGHTS_5_SIGNAL as over
    check("env override wins", over["semantic"] == 0.7)
    os.environ.pop("RANK_W_SEMANTIC", None)


def test_embedding_text() -> None:
    print("\nembedding text")
    _reload("conversation")
    from src.embedding_service import EmbeddingService
    txt = EmbeddingService.memory_embedding_text(MEMS[0])
    check("reads as natural language with speaker and date",
          "Melanie" in txt and "8 May, 2023" in txt)
    check("drops the constant 'type:' component", "type:" not in txt,
          "it appears in every memory of a type and is pure noise in the vector")

    _reload("legacy")
    from src.embedding_service import EmbeddingService as Legacy
    check("legacy form preserved",
          "type: event" in Legacy.memory_embedding_text(MEMS[0]))


def test_speaker_prompt() -> None:
    print("\nspeaker-aware extraction prompt")
    _reload("conversation")
    from src.llm_extractor import LLMExtractor

    ex = object.__new__(LLMExtractor)
    base = LLMExtractor.EXTRACTION_PROMPT
    check("base prompt is single-user framed",
          "about the user" in base.lower(),
          "this is the framing the preamble corrects")

    captured = {}

    def fake_call(prompt):
        captured["prompt"] = prompt
        return "[]"

    ex._call_llm = fake_call
    ex.provider = "test"
    # Mirror every counter __init__ sets. object.__new__ skips __init__, and extract()
    # wraps its body in a broad `except Exception`, so a missing attribute surfaces only
    # as a logged line while the test still reports OK. Set them all, then assert the
    # call actually SUCCEEDED -- otherwise this check silently stops covering the
    # post-parse path it is supposed to cover.
    ex.extraction_count = 0
    ex.escalation_count = 0
    ex.api_call_count = 0
    ex.total_response_time_ms = 0.0
    ex.key_rotation_count = 0

    out = ex.extract("I went to the museum", 1,
                     speaker="Melanie", event_date="8 May, 2023")
    check("extraction completes without swallowing an exception",
          out == [] and ex.api_call_count == 1,
          "extract() catches everything, so a silent failure looks like success")

    p = captured.get("prompt", "")
    check("names the speaker", "Melanie" in p)
    check("forbids generic user keys", "user_name" in p)
    check("supplies the date for relative references", "8 May, 2023" in p)

    captured.clear()
    ex.extract("I went to the museum", 2)
    p2 = captured.get("prompt", "")
    check("no preamble when the speaker is unknown",
          "conversation between several people" not in p2)


def test_results_roundtrip() -> None:
    """The diagnose reader must find rows the worker actually writes.

    diagnose.py originally read a "questions" key the worker never produces, so every
    invocation printed "no question records found" -- indistinguishable from a run that
    had not happened. Writing a payload shaped like the worker's and reading it back
    pins the contract between the two modules.
    """
    print("\nresults reader/writer contract")
    import json
    import tempfile
    from pathlib import Path
    from benchmarks.diagnose import audit_abstentions, load_records

    tmp = Path(tempfile.mkdtemp())
    raw = tmp / "raw"
    raw.mkdir()
    (raw / "conv-1.json").write_text(json.dumps({
        "sample_id": "conv-1",
        "config": {"profile": "conversation"},
        "ingest": {"turns": 419, "stage3_calls": 337},
        "qa": {"count": 2, "seconds": 12.0},
        "records": [
            {"question": "Where?", "gold": "Seattle", "prediction": "Seattle",
             "category": 4, "category_name": "single_hop", "judge_correct": True,
             "abstained": False},
            {"question": "Her car?", "gold": "Not mentioned", "prediction": "NO_ANSWER",
             "category": 5, "category_name": "adversarial", "judge_correct": False,
             "abstained": True},
        ],
    }), encoding="utf-8")

    recs = load_records(tmp)
    check("reads the worker's 'records' key", len(recs) == 2,
          "this mismatch made diagnose report an empty run")
    check("stamps the sample id", recs[0].get("_sample") == "conv-1")

    a = audit_abstentions(recs)
    check("audit runs on real payload shape", a["graded"] == 2)
    check("adversarial refusal marked wrong is reported",
          len(a["understated"]) == 1)

    # A payload with no recognised rows must raise, not silently return nothing.
    (raw / "conv-2.json").write_text(json.dumps({"sample_id": "c2", "qa": {}}),
                                     encoding="utf-8")
    bad = tmp / "onlybad"
    (bad / "raw").mkdir(parents=True)
    (bad / "raw" / "x.json").write_text(json.dumps({"sample_id": "x", "rows": []}),
                                        encoding="utf-8")
    try:
        load_records(bad)
        check("unrecognised payload raises", False, "it returned quietly")
    except ValueError:
        check("unrecognised payload raises rather than reporting an empty run", True)


def test_dataset_dates() -> None:
    print("\ndataset date parsing")
    from benchmarks.dataset import Turn
    ok = Turn(session=1, session_date="8 May, 2023", speaker="M", text="x", dia_id="D:1")
    check("parses the common LoCoMo form", ok.event_ts > 0)
    bad = Turn(session=1, session_date="not a date", speaker="M", text="x", dia_id="D:2")
    check("unparseable date degrades to 0.0 rather than raising", bad.event_ts == 0.0)
    empty = Turn(session=1, session_date="", speaker="M", text="x", dia_id="D:3")
    check("missing date is safe", empty.event_ts == 0.0)


def test_stratified_sampling() -> None:
    print("\nquestion sampling")
    from benchmarks.worker import _stratified_sample

    class Q:
        def __init__(self, c):
            self.category = c

    qs = [Q(1)] * 30 + [Q(2)] * 40 + [Q(3)] * 10 + [Q(4)] * 50 + [Q(5)] * 20
    picked = _stratified_sample(qs, 20)
    cats = {q.category for q in picked}
    check("covers every category", cats == {1, 2, 3, 4, 5},
          "a head-slice returns category 1 only")
    check("returns exactly the requested count", len(_stratified_sample(qs, 13)) == 13)
    check("passes through when limit exceeds supply",
          len(_stratified_sample(qs, 9999)) == len(qs))
    check("is deterministic",
          [q.category for q in _stratified_sample(qs, 7)]
          == [q.category for q in _stratified_sample(qs, 7)])


def main() -> int:
    print("=" * 66)
    print("Memora conversation-architecture self-test")
    print("=" * 66)

    for fn in (test_lexical, test_dedup_identity, test_context_rendering,
               test_query_intent, test_query_dates, test_always_on_floor,
               test_ranking_profiles, test_embedding_text, test_speaker_prompt,
               test_results_roundtrip, test_dataset_dates, test_stratified_sampling):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] {fn.__name__} raised {type(exc).__name__}: {exc}")
            FAILURES.append(fn.__name__)

    os.environ.pop("MEMORA_PROFILE", None)
    print("\n" + "=" * 66)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
