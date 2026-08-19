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

    # k1 is configurable and currently set high (50), which flattens term-frequency
    # saturation. Ranking must still be sane at that setting, and the parameter must
    # actually reach the index rather than silently defaulting.
    _reload("conversation")
    from src.config import BM25_K1, BM25_B
    from src.retriever import MemoryRetriever  # noqa: F401  (import path check)
    hi = BM25Index(k1=BM25_K1, b=BM25_B).build(MEMS)
    check(f"index accepts the configured k1 ({BM25_K1})", hi.k1 == BM25_K1)
    hits_hi = hi.search("Ravensbourne", limit=3)
    check("rare proper noun still ranks first at the configured k1",
          bool(hits_hi) and hits_hi[0][0] == "m1")
    check("high k1 does not produce negative or nan scores",
          all(s > 0 and s == s for _, s in hits_hi))

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


class _FakeRedis:
    """Minimal Redis stand-in: just the set operations the entity index uses."""

    def __init__(self):
        self.sets = {}

    def pipeline(self):
        return self

    def sadd(self, key, val):
        self.sets.setdefault(key, set()).add(val)

    def execute(self):
        return None

    def smembers(self, key):
        return self.sets.get(key, set())

    def scan_iter(self, match=None, count=None):
        prefix = (match or "").rstrip("*")
        return [k for k in list(self.sets) if k.startswith(prefix)]

    def delete(self, key):
        self.sets.pop(key, None)


def test_entity_index() -> None:
    print("\nentity-centric index")
    _reload("conversation")
    from src.entity_index import EntityIndex, extract_entities

    ents = extract_entities({
        "speaker": "Melanie", "key": "museum visit",
        "value": "went to the Ravensbourne museum with Caroline",
        "source_text": "",
    })
    check("indexes the speaker", "melanie" in ents)
    check("indexes proper nouns from the value",
          "ravensbourne" in ents and "caroline" in ents)
    check("excludes capitalised non-entities",
          not ({"the", "went", "with"} & ents), sorted(ents))

    idx = EntityIndex(_FakeRedis())
    idx.add({"memory_id": "m1", "speaker": "Melanie", "key": "adoption",
             "value": "researching agencies", "source_text": ""})
    idx.add({"memory_id": "m2", "speaker": "Caroline", "key": "chat",
             "value": "talked with Melanie about adoption", "source_text": ""})
    idx.add({"memory_id": "m3", "speaker": "Caroline", "key": "hobby",
             "value": "plays guitar", "source_text": ""})

    got = idx.memories_for(["Melanie"])
    check("finds memories by entity", set(got) == {"m1", "m2"}, str(got))

    # The multi-hop property: a memory naming BOTH queried entities ranks above one
    # naming only one, which is what makes this a bridge-finder.
    counts = idx.match_counts(["Melanie", "Caroline"])
    check("memory mentioning both entities scores highest",
          counts.get("m2") == 2 and counts.get("m1") == 1, str(counts))
    ranked = idx.memories_for(["Melanie", "Caroline"])
    check("both-entity memory ranks first", ranked[0] == "m2", str(ranked))

    check("unknown entity returns nothing", idx.memories_for(["Nobody"]) == [])
    check("clear empties the index", idx.clear() > 0 and idx.memories_for(["Melanie"]) == [])

    _reload("legacy")
    from src.config import ENTITY_INDEX_ENABLED as legacy_off
    check("disabled under the legacy profile", legacy_off is False)


def test_extraction_quality_prompt() -> None:
    print("\nextraction value-quality guidance")
    _reload("conversation")
    from src.llm_extractor import LLMExtractor

    ex = object.__new__(LLMExtractor)
    captured = {"calls": 0}

    def _fake_llm(prompt: str) -> str:
        # setdefault returns the prompt (truthy), so `... or "[]"` yielded the
        # PROMPT, not an empty array -- which is non-JSON and drove the retry
        # path. Capture and return separately.
        captured.setdefault("p", prompt)
        captured["calls"] += 1
        return "[]"

    ex._call_llm = _fake_llm
    ex.provider = "test"
    for attr in ("extraction_count", "escalation_count", "api_call_count",
                 "key_rotation_count"):
        setattr(ex, attr, 0)
    ex.total_response_time_ms = 0.0

    ex.extract("The charity race was great", 1, speaker="Melanie")
    p = captured.get("p", "")
    check("demands self-contained values", "SELF-CONTAINED" in p)
    check("demands one memory per distinct fact", "ONE MEMORY PER DISTINCT FACT" in p)
    check("shows the over-compression failure as a counter-example",
          "purpose lost" in p,
          "the base prompt's bare-token examples taught the compression")
    check("asks for states as well as actions", "relationship status" in p)

    check("valid JSON costs exactly one LLM call", captured["calls"] == 1,
          "a fake returning non-JSON used to recurse until RecursionError")

    # A model that never emits JSON must cost at most one retry, not one API
    # call per stack frame.
    bad = object.__new__(LLMExtractor)
    bad_calls = {"n": 0}

    def _never_json(prompt: str) -> str:
        bad_calls["n"] += 1
        return "I could not find anything worth remembering."

    bad._call_llm = _never_json
    bad.provider = "test"
    for attr in ("extraction_count", "escalation_count", "api_call_count",
                 "key_rotation_count"):
        setattr(bad, attr, 0)
    bad.total_response_time_ms = 0.0
    out = bad.extract("The charity race was great", 1, speaker="Melanie")
    check("unparseable output returns empty, not an exception", out == [])
    check("unparseable output retries at most once", bad_calls["n"] == 2,
          f"took {bad_calls['n']} LLM calls; unbounded recursion burns quota per frame")

    from src.config import STAGE_3_MAX_TOKENS
    check("token cap leaves room for richer values", STAGE_3_MAX_TOKENS >= 1000,
          f"got {STAGE_3_MAX_TOKENS}; truncated JSON is swallowed as 'nothing found'")


def test_temporal_enrichment() -> None:
    """Deterministic recovery of dates the LLM compressed away."""
    print("\ndeterministic temporal enrichment")
    _reload("conversation")
    from src.extractor import MemoryExtractor

    ex = object.__new__(MemoryExtractor)
    enrich = lambda mems, msg: MemoryExtractor._enrich_temporal(ex, mems, msg)  # noqa: E731

    # The measured failure: gold was 2022, the value kept only the subject.
    out = enrich([{"value": "lake sunrise", "key": "painting title"}],
                 "[8 May, 2023] Melanie: I painted a lake sunrise back in 2022")
    check("recovers a year the value lost", "2022" in out[0]["value"],
          out[0]["value"])

    # The utterance-date prefix must not be mistaken for content -- otherwise every
    # memory gets stamped with the day it was mentioned.
    check("ignores the [utterance date] prefix", "2023" not in out[0]["value"],
          out[0]["value"])

    unchanged = enrich([{"value": "camping trip June 2023", "key": "trip"}],
                       "We're going camping in June 2023")
    check("does not duplicate a date the value already has",
          unchanged[0]["value"].count("2023") == 1, unchanged[0]["value"])

    none = enrich([{"value": "likes coffee", "key": "pref"}],
                  "I really like coffee in the morning")
    check("adds nothing when no date is stated", none[0]["value"] == "likes coffee")

    # Ambiguity guard: several dates and no way to know which applies.
    multi = enrich([{"value": "moved house", "key": "move"}],
                   "I moved in 2019 and again in 2021")
    check("refuses to guess between multiple dates",
          multi[0]["value"] == "moved house",
          "a wrong date turns an abstention into a confident error")

    check("empty input is safe", enrich([], "in 2022") == [])

    # The measured LoCoMo failure: gold "7 May", predicted "8 May" -- the support group
    # visit was mentioned during the 8 May session but happened "yesterday". Without a
    # relative-day pattern, the value carried no signal that it wasn't the same day.
    yesterday = enrich([{"value": "went to the support group", "key": "support group"}],
                       "[8 May, 2023] Caroline: I went to the support group yesterday")
    check("recovers a relative-day reference",
          "yesterday" in yesterday[0]["value"], yesterday[0]["value"])


def test_context_noise() -> None:
    """Two sources of context noise found by reading a real failing case."""
    print("\ncontext noise reduction")
    r = _reload("conversation")

    check("strips time-of-day from dates",
          r._clean_date("1:56 pm on 8 May, 2023") == "8 May, 2023")
    check("leaves a plain date alone",
          r._clean_date("8 May, 2023") == "8 May, 2023")
    check("handles an empty date", r._clean_date("") == "")

    check("drops a duplicated speaker from the key",
          r._strip_redundant_speaker("Caroline current activity", "Caroline")
          == "current activity")
    check("handles the possessive form",
          r._strip_redundant_speaker("Melanie's children", "Melanie") == "children")
    check("leaves an unrelated key intact",
          r._strip_redundant_speaker("camping trip", "Melanie") == "camping trip")
    check("never returns an empty key",
          r._strip_redundant_speaker("Caroline", "Caroline") == "Caroline")

    mems = [{"memory_id": "m", "type": "event", "key": "Melanie camping trip",
             "value": "June 2023", "confidence": 0.9, "turn_number": 10,
             "speaker": "Melanie", "event_date": "1:14 pm on 25 May, 2023",
             "event_ts": 0.0, "retrieval_score": 0.9, "source_text": ""}]
    out = r.MemoryRetriever.format_memories_for_prompt(
        object.__new__(r.MemoryRetriever), mems)
    check("rendered line is clean", "1:14 pm" not in out and "Melanie - camping trip" in out,
          out.strip())


def test_reader_prompt() -> None:
    print("\nreader prompt")
    import importlib
    import benchmarks.qa as qa

    os.environ.pop("BENCH_READER_PROMPT", None)
    qa = importlib.reload(qa)
    p = qa.READER_SYSTEM
    check("v3 is the default", qa.READER_PROMPT_VERSION == "v3")
    check("explains the context format", "HOW TO READ THE CONTEXT" in p)
    check("explains that the bracketed date is when it was said",
          "WHEN IT WAS SAID" in p,
          "the camping failure came from two dates with no stated relationship")
    check("tells the reader a date inside the value can be the answer",
          "inside the value" in p)
    check("counterweights the refusal instruction",
          "expected and correct" in p)
    check("still constrains verbosity", "terse" in p.lower())
    check("drops the attribution prohibition that doubled abstentions",
          "do not transfer it" not in p,
          "present in v2, which scored 32% with abstentions 8 -> 16")
    check("the closing line does not invite refusal",
          "NO_ANSWER" not in qa.READER_TEMPLATE,
          "the last line before generation outweighs the system prompt")
    check("the closing line is still a prompt to answer",
          qa.READER_TEMPLATE.rstrip().endswith("Answer:"))

    for ver, marker in (("v1", "HOW TO READ THE CONTEXT"),
                        ("v2", "do not transfer it")):
        os.environ["BENCH_READER_PROMPT"] = ver
        qa = importlib.reload(qa)
        present = marker in qa.READER_SYSTEM
        check(f"{ver} is selectable for A/B",
              qa.READER_PROMPT_VERSION == ver and (present if ver == "v2" else not present))
        if ver == "v1":
            check("v1 keeps its original template verbatim",
                  "NO_ANSWER" in qa.READER_TEMPLATE,
                  "otherwise the v1 baseline is not the thing that was measured")
    os.environ.pop("BENCH_READER_PROMPT", None)
    importlib.reload(qa)


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
               test_entity_index, test_extraction_quality_prompt,
               test_temporal_enrichment, test_context_noise, test_reader_prompt,
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
