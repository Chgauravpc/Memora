"""
BM25 lexical retrieval over the memory store.

WHY A SECOND RETRIEVAL CHANNEL AT ALL

Dense retrieval with a 384-dimensional MiniLM is good at paraphrase and topical similarity
and consistently weak on RARE PROPER NOUNS -- personal names, place names, band names,
book titles. Those are precisely what questions about a conversation turn on ("what did
Melanie say about Ravensbourne?"). A rare token contributes little to a sentence embedding,
because the model has barely seen it; but it is *maximally* informative in the
inverse-document-frequency sense, which is exactly what BM25 rewards.

So the two methods fail on DIFFERENT queries. That is the whole argument for fusing them
rather than tuning either one harder: their errors are weakly correlated, so the union
recalls substantially more than the better of the two alone.

Fusion is Reciprocal Rank Fusion (Cormack et al., 2009): score = sum over channels of
w / (k + rank). RRF combines by RANK rather than by score, which matters because a cosine
similarity and a BM25 score are not on comparable scales and normalising them against each
other requires tuning constants that do not transfer between corpora. RRF needs none.

No new dependency: this is a few hundred lines of standard BM25 over an in-process index,
and memory stores here are thousands of records, not millions.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Deliberately permissive: keeps digits and intra-word apostrophes so dates ("2023") and
# possessives ("melanie's") survive tokenisation, since both carry retrieval signal here.
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9']*")

# Only the highest-frequency function words. Kept short on purpose -- aggressive stoplists
# discard words that are discriminative in short texts, and BM25's IDF term already damps
# common words automatically.
_STOPWORDS = frozenset("""
a an the and or but if then than that this these those of to in on at for with without
from by as is are was were be been being it its he she they them his her their we you i
do does did not no yes so such about into over under again further once here there all
any both each few more most other some only own same too very can will just should now
""".split())


def tokenize(text: str) -> List[str]:
    """Lowercase, split, drop stopwords and single characters."""
    return [t for t in _TOKEN_RE.findall((text or "").lower())
            if t not in _STOPWORDS and len(t) > 1]


def memory_text(memory: Dict) -> str:
    """Searchable surface of a memory.

    Includes `source_text` -- the utterance the memory was extracted from. key:value is a
    lossy compression, and the original sentence frequently contains the proper noun the
    question uses while the compressed value does not.
    """
    parts = [
        str(memory.get('key') or ''),
        str(memory.get('value') or ''),
        str(memory.get('speaker') or ''),
        str(memory.get('event_date') or ''),
        str(memory.get('source_text') or ''),
    ]
    return " ".join(p for p in parts if p)


class BM25Index:
    """Okapi BM25 over memory records.

    Rebuilt rather than incrementally maintained: corpus statistics (average document
    length, document frequencies) shift as memories are added, and for a few thousand
    documents a full rebuild is milliseconds. Correctness beats cleverness at this size.
    """

    __slots__ = ("k1", "b", "_ids", "_tf", "_df", "_len", "_avg_len", "_n")

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        # k1 controls term-frequency saturation, b the length normalisation. These are the
        # standard Okapi defaults and are not worth tuning without a labelled dev set.
        self.k1 = k1
        self.b = b
        self._ids: List[str] = []
        self._tf: List[Counter] = []
        self._df: Dict[str, int] = defaultdict(int)
        self._len: List[int] = []
        self._avg_len: float = 0.0
        self._n: int = 0

    def build(self, memories: Iterable[Dict]) -> "BM25Index":
        ids: List[str] = []
        tfs: List[Counter] = []
        lens: List[int] = []
        df: Dict[str, int] = defaultdict(int)

        for mem in memories:
            mem_id = mem.get('memory_id')
            if not mem_id:
                continue
            tokens = tokenize(memory_text(mem))
            if not tokens:
                # Still indexed: an empty document simply never matches, and dropping it
                # would silently desynchronise ids from the store.
                tokens = []
            tf = Counter(tokens)
            ids.append(mem_id)
            tfs.append(tf)
            lens.append(len(tokens))
            for term in tf:
                df[term] += 1

        self._ids, self._tf, self._len, self._df = ids, tfs, lens, df
        self._n = len(ids)
        self._avg_len = (sum(lens) / self._n) if self._n else 0.0
        return self

    def _idf(self, term: str) -> float:
        # Robertson/Sparck-Jones IDF with the +0.5 smoothing, floored at zero. Without the
        # floor, terms appearing in more than half the corpus get NEGATIVE weight and can
        # actively push relevant documents down.
        n_q = self._df.get(term, 0)
        if n_q == 0:
            return 0.0
        return max(0.0, math.log(1.0 + (self._n - n_q + 0.5) / (n_q + 0.5)))

    def search(self, query: str, limit: int = 100) -> List[Tuple[str, float]]:
        """(memory_id, score) for the best `limit` matches, descending."""
        if not self._n:
            return []
        q_terms = tokenize(query)
        if not q_terms:
            return []

        idf = {t: self._idf(t) for t in set(q_terms)}
        active = [t for t in set(q_terms) if idf[t] > 0.0]
        if not active:
            return []

        scores: List[Tuple[str, float]] = []
        for i, tf in enumerate(self._tf):
            doc_len = self._len[i]
            if not doc_len:
                continue
            norm = self.k1 * (1 - self.b + self.b * doc_len / self._avg_len) if self._avg_len else self.k1
            total = 0.0
            for term in active:
                freq = tf.get(term, 0)
                if freq:
                    total += idf[term] * (freq * (self.k1 + 1)) / (freq + norm)
            if total > 0.0:
                scores.append((self._ids[i], total))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:limit]

    def __len__(self) -> int:
        return self._n


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Tuple[Sequence[str], float]],
    k: int = 60,
) -> Dict[str, float]:
    """Fuse ranked ID lists into {id: fused_score}.

    `ranked_lists` is a sequence of (ids_in_rank_order, weight). Combining by rank rather
    than raw score is the point: cosine similarity and BM25 are on incomparable scales, and
    any normalisation between them needs constants that do not transfer across corpora.
    """
    fused: Dict[str, float] = defaultdict(float)
    for ids, weight in ranked_lists:
        for rank, mem_id in enumerate(ids, start=1):
            fused[mem_id] += weight / (k + rank)
    return dict(fused)


def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """Rescale to [0, 1] so fused scores can enter a weighted sum with other signals.

    RRF scores are tiny (order 1/k) and their absolute magnitude is meaningless -- only the
    ordering is. Rescaling by the observed max makes the top hit 1.0 so the downstream
    `semantic` weight means the same thing whether fusion ran or not.
    """
    if not scores:
        return {}
    top = max(scores.values())
    if top <= 0:
        return {mid: 0.0 for mid in scores}
    return {mid: val / top for mid, val in scores.items()}
