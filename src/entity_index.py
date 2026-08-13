"""
Entity-centric index: which memories mention which entity.

WHY THIS INSTEAD OF A KNOWLEDGE GRAPH

Most of what a KG buys on conversational QA is entity-centred recall: a question names
someone or something, and you want every memory about it, whether or not any single one is
a good embedding match for the question's phrasing. Multi-hop questions in particular name
one entity and ask about another reachable only through it -- and the bridging memory is
often a poor lexical and semantic match for the question, so neither dense nor BM25 search
reliably surfaces it.

A real KG adds typed relations, a schema, and entity resolution. Those cost days and bring
their own failure modes: relation-schema drift, and merging two people who share a name.
An inverted index from entity to memories needs none of that and captures the recall
benefit. What it cannot do is INFER across edges ("no current partner + a 2019 breakup
therefore single") -- that remains the genuine reason to build a graph later.

It is also worth being clear about the ordering: a graph built on lossy extraction inherits
those losses. Extraction quality bounds both.

Entity detection is deliberately shallow -- capitalised tokens plus the speaker -- because a
NER model in the ingest path is another dependency and another thing to be wrong. Precision
matters more than recall here: a spurious entity adds noise to one bucket, while a missed
one merely leaves retrieval as it was.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, Iterable, List, Optional, Set

logger = logging.getLogger(__name__)

# Capitalised words of 3+ characters. Sentence-initial position is NOT excluded here (unlike
# in query parsing) because memory values are fragments rather than sentences, so the first
# word is as likely to be a name as any other.
_CAP_RE = re.compile(r"\b([A-Z][a-zA-Z]{2,})\b")

# Words that are capitalised for reasons other than being a name. Without these, every
# memory acquires entities like "The" or a weekday and the buckets stop discriminating.
_NOT_ENTITIES = frozenset("""
the this that these those and but for with from into over under
i you he she it we they his her their our your its
monday tuesday wednesday thursday friday saturday sunday
january february march april may june july august september october november december
yes no not okay ok thanks hi hello hey well just also then than when what where who why how
my me am is are was were be been being do does did have has had will would can could should
""".split())

ENTITY_PREFIX = "ent:"


def normalize(entity: str) -> str:
    return (entity or "").strip().lower()


def extract_entities(memory: Dict, max_entities: int = 12) -> Set[str]:
    """Entities a memory is about.

    The speaker is always included: it is the one attribution known to be reliable, since
    it comes from the caller rather than from a model's reading of the text.
    """
    found: Set[str] = set()

    speaker = normalize(str(memory.get('speaker') or ''))
    if speaker:
        found.add(speaker)
        # Index the first name too, so "Melanie" matches a "Melanie Carter" speaker.
        first = speaker.split()[0]
        if len(first) > 2:
            found.add(first)

    # source_text is searched as well as key and value: extraction compresses away proper
    # nouns, and the original sentence is where they survive.
    for field in ('key', 'value', 'source_text'):
        text = str(memory.get(field) or '')
        if not text:
            continue
        for match in _CAP_RE.finditer(text):
            token = normalize(match.group(1))
            if token and token not in _NOT_ENTITIES:
                found.add(token)
            if len(found) >= max_entities:
                break

    return found


class EntityIndex:
    """Redis-backed inverted index, entity -> set of memory_ids.

    Lives in the same logical DB as the memories it indexes, so the benchmark's
    per-worker isolation covers it with no extra work.
    """

    def __init__(self, client):
        self.client = client

    def add(self, memory: Dict) -> int:
        """Index one memory. Returns how many entity buckets it joined."""
        memory_id = memory.get('memory_id')
        if not memory_id:
            return 0
        entities = extract_entities(memory)
        if not entities:
            return 0
        try:
            pipe = self.client.pipeline()
            for ent in entities:
                pipe.sadd(f"{ENTITY_PREFIX}{ent}", memory_id)
            pipe.execute()
            return len(entities)
        except Exception as exc:  # noqa: BLE001
            # Indexing is an enhancement to retrieval, never a precondition for storing a
            # memory. A failure here must not lose the write that just succeeded.
            logger.warning("Entity indexing failed for %s: %s", memory_id, exc)
            return 0

    def memories_for(self, entities: Iterable[str], limit: int = 100) -> List[str]:
        """Memory ids mentioning any of `entities`, most-shared first.

        Ordering by how many of the query's entities a memory matches is what makes this
        useful for multi-hop: a memory mentioning BOTH named entities is far more likely to
        be the bridge between them than one mentioning either alone.
        """
        counts: Dict[str, int] = {}
        try:
            for ent in entities:
                key = f"{ENTITY_PREFIX}{normalize(ent)}"
                for mem_id in self.client.smembers(key) or []:
                    mid = mem_id.decode() if isinstance(mem_id, bytes) else str(mem_id)
                    counts[mid] = counts.get(mid, 0) + 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("Entity lookup failed: %s", exc)
            return []

        ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        return [mid for mid, _ in ranked[:limit]]

    def match_counts(self, entities: Iterable[str]) -> Dict[str, int]:
        """{memory_id: how many of these entities it mentions}."""
        counts: Dict[str, int] = {}
        try:
            for ent in entities:
                for mem_id in self.client.smembers(f"{ENTITY_PREFIX}{normalize(ent)}") or []:
                    mid = mem_id.decode() if isinstance(mem_id, bytes) else str(mem_id)
                    counts[mid] = counts.get(mid, 0) + 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("Entity lookup failed: %s", exc)
        return counts

    def clear(self) -> int:
        """Drop every entity bucket. Called alongside clearing memories."""
        removed = 0
        try:
            for key in self.client.scan_iter(match=f"{ENTITY_PREFIX}*", count=500):
                self.client.delete(key)
                removed += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("Entity index clear failed: %s", exc)
        return removed
