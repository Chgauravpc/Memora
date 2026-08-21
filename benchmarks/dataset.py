"""
LoCoMo dataset loader.

Schema (confirmed against snap-research/locomo):

    [
      {
        "sample_id": "...",
        "conversation": {
          "speaker_a": "Caroline",
          "speaker_b": "Melanie",
          "session_1_date_time": "1:56 pm on 8 May, 2023",
          "session_1": [
            {"speaker": "Caroline", "dia_id": "D1:1", "text": "...",
             "img_url": [...], "blip_caption": "..."},
            ...
          ],
          "session_2_date_time": "...",
          "session_2": [...]
        },
        "qa": [
          {"question": "...", "answer": "...", "evidence": ["D1:3"], "category": 4},
          {"question": "...", "adversarial_answer": "...", "category": 5}
        ],
        "event_summary": {...}, "observation": {...}, "session_summary": {...}
      },
      ...
    ]

Two traps this loader handles explicitly:

1. **Category 5 has no `answer` key** -- it carries `adversarial_answer` instead. Reading
   `qa["answer"]` blindly scores every adversarial question as wrong. `Question.gold`
   resolves the right field.
2. **Sessions must be ordered numerically, not lexically.** `sorted()` on the raw key
   strings gives session_1, session_10, session_11, session_2 ... which scrambles
   chronology and silently destroys every temporal question. We sort on the parsed int.

Category integers (1=multi-hop, 2=temporal, 3=open-domain, 4=single-hop, 5=adversarial)
are the community-standard mapping. `inspect()` prints observed counts so the mapping can
be sanity-checked against the actual file rather than trusted.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from .paths import LOCOMO_JSON, assert_contained

CATEGORY_NAMES: Dict[int, str] = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial",
}

_SESSION_RE = re.compile(r"^session_(\d+)$")


@dataclass
class Turn:
    session: int
    session_date: str
    speaker: str
    text: str
    dia_id: str
    blip_caption: Optional[str] = None

    @property
    def event_ts(self) -> float:
        """`session_date` as epoch seconds, or 0.0 if it cannot be parsed.

        Used for ordering and proximity comparisons. LoCoMo dates appear in a few forms
        ("8 May, 2023", "8 May 2023"), so several patterns are tried and an unparseable
        date degrades to 0.0 rather than raising -- the human-readable `event_date` is
        still stored and remains answerable either way.
        """
        raw = (self.session_date or "").strip()
        if not raw:
            return 0.0
        cleaned = raw.replace(",", " ")
        cleaned = " ".join(cleaned.split())
        for fmt in ("%d %B %Y", "%d %b %Y", "%B %d %Y", "%b %d %Y", "%Y-%m-%d"):
            try:
                return _dt.datetime.strptime(cleaned, fmt).timestamp()
            except ValueError:
                continue
        return 0.0

    def render(self, include_date: bool = True) -> str:
        """
        Flatten to the single string `MemorySystem.process_turn` accepts.

        `process_turn(user_message)` takes no speaker and no timestamp argument, so both
        have to be folded into the text. This is an adapter-level workaround for two
        genuine API gaps -- see BENCHMARK_README.md "Known limitations".

        Images: LoCoMo turns may carry an image whose content is only available as a
        `blip_caption`. We append it so the text channel is not silently missing evidence
        that a QA pair may depend on.
        """
        body = self.text or ""
        if self.blip_caption:
            body = f"{body} [shares an image: {self.blip_caption}]".strip()
        if include_date:
            return f"[{self.session_date}] {self.speaker}: {body}"
        return f"{self.speaker}: {body}"


@dataclass
class Question:
    question: str
    answer: Optional[str]
    adversarial_answer: Optional[str]
    category: int
    evidence: List[str] = field(default_factory=list)

    @property
    def gold(self) -> Optional[str]:
        """Category 5 stores its gold answer under `adversarial_answer`."""
        if self.category == 5:
            return self.adversarial_answer
        return self.answer

    @property
    def category_name(self) -> str:
        return CATEGORY_NAMES.get(self.category, f"category_{self.category}")


@dataclass
class Conversation:
    sample_id: str
    speaker_a: str
    speaker_b: str
    turns: List[Turn]
    questions: List[Question]

    @property
    def num_sessions(self) -> int:
        return len({t.session for t in self.turns})


def _parse_conversation(raw: Dict[str, Any], sample_id: str) -> tuple[List[Turn], str, str]:
    conv = raw.get("conversation") or {}
    speaker_a = conv.get("speaker_a", "SpeakerA")
    speaker_b = conv.get("speaker_b", "SpeakerB")

    # Numeric sort -- lexical sort would scramble session_10 before session_2.
    session_ids: List[int] = []
    for key, value in conv.items():
        m = _SESSION_RE.match(key)
        if m and isinstance(value, list):
            session_ids.append(int(m.group(1)))
    session_ids.sort()

    turns: List[Turn] = []
    for sid in session_ids:
        date = conv.get(f"session_{sid}_date_time") or ""
        for entry in conv[f"session_{sid}"]:
            if not isinstance(entry, dict):
                continue
            text = entry.get("text") or ""
            caption = entry.get("blip_caption")
            if not text and not caption:
                continue
            turns.append(Turn(
                session=sid,
                session_date=str(date),
                speaker=str(entry.get("speaker") or "?"),
                text=str(text),
                dia_id=str(entry.get("dia_id") or ""),
                blip_caption=str(caption) if caption else None,
            ))
    return turns, speaker_a, speaker_b


def load_conversations(path: Path | None = None) -> List[Conversation]:
    path = assert_contained(path or LOCOMO_JSON, "dataset")
    if not path.exists():
        raise FileNotFoundError(
            f"LoCoMo dataset not found at {path}\n"
            "Fetch it with:  python -m benchmarks.download\n"
            "or copy locomo10.json from https://github.com/snap-research/locomo "
            f"into {path.parent}"
        )

    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    if isinstance(raw, dict):  # tolerate a {"data": [...]} wrapper
        for key in ("data", "samples", "conversations"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break
    if not isinstance(raw, list):
        raise ValueError(
            f"Expected a JSON list of samples in {path}, got {type(raw).__name__}. "
            "Run `python -m benchmarks.dataset --inspect` to see the structure."
        )

    out: List[Conversation] = []
    for i, sample in enumerate(raw):
        sample_id = str(sample.get("sample_id") or f"conv_{i}")
        turns, speaker_a, speaker_b = _parse_conversation(sample, sample_id)

        questions: List[Question] = []
        for qa in sample.get("qa") or []:
            if not isinstance(qa, dict) or not qa.get("question"):
                continue
            try:
                category = int(qa.get("category", -1))
            except (TypeError, ValueError):
                category = -1
            ans = qa.get("answer")
            questions.append(Question(
                question=str(qa["question"]),
                answer=None if ans is None else str(ans),
                adversarial_answer=(
                    str(qa["adversarial_answer"])
                    if qa.get("adversarial_answer") is not None else None
                ),
                category=category,
                evidence=[str(e) for e in (qa.get("evidence") or [])],
            ))

        if turns:
            out.append(Conversation(sample_id, speaker_a, speaker_b, turns, questions))
    return out


def iter_answerable(conv: Conversation, include_adversarial: bool = True) -> Iterator[Question]:
    """Questions that have a gold answer we can actually grade against."""
    for q in conv.questions:
        if q.gold is None:
            continue
        if not include_adversarial and q.category == 5:
            continue
        yield q


def inspect(path: Path | None = None) -> None:
    """Print observed structure so schema assumptions can be verified, not trusted."""
    convs = load_conversations(path)
    total_turns = sum(len(c.turns) for c in convs)
    total_q = sum(len(c.questions) for c in convs)
    gradable = sum(len(list(iter_answerable(c))) for c in convs)

    print(f"conversations : {len(convs)}")
    print(f"turns         : {total_turns} (mean {total_turns / max(len(convs), 1):.0f}/conv)")
    print(f"questions     : {total_q} ({gradable} with a gold answer)")
    print()

    by_cat: Dict[int, int] = {}
    missing_gold: Dict[int, int] = {}
    for c in convs:
        for q in c.questions:
            by_cat[q.category] = by_cat.get(q.category, 0) + 1
            if q.gold is None:
                missing_gold[q.category] = missing_gold.get(q.category, 0) + 1

    print("category breakdown (verify this matches the documented mapping):")
    for cat in sorted(by_cat):
        name = CATEGORY_NAMES.get(cat, "UNKNOWN")
        miss = missing_gold.get(cat, 0)
        warn = f"  <-- {miss} missing gold answer" if miss else ""
        print(f"  {cat} {name:<12} {by_cat[cat]:>5}{warn}")

    print()
    for c in convs[:3]:
        print(f"  {c.sample_id}: {len(c.turns)} turns / {c.num_sessions} sessions "
              f"/ {len(c.questions)} questions  [{c.speaker_a} & {c.speaker_b}]")
    if convs and convs[0].turns:
        print(f"\nfirst rendered turn:\n  {convs[0].turns[0].render()[:200]}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Inspect the LoCoMo dataset")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--path", type=Path, default=None)
    args = ap.parse_args()
    inspect(args.path)
