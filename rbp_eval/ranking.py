"""Port of librbp/qdocs.[ch]: per-query document lists and their orderings."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from enum import Enum, auto


class Ordering(Enum):
    OCCUR = auto()  # order of occurrence in the run file
    RANK = auto()   # by the rank field
    SCORE = auto()  # by the score field, as trec_eval does


# Order by similarity score, as with trec_eval. Ties are nominally ordered
# by docid, but in fact rbp handles ties explicitly (see rbp.py).
DEFAULT_ORDERING = Ordering.SCORE


@dataclass
class DocScore:
    docid: str
    occur: int
    rank: int
    score: float
    flags: int = 0


def _score_cmp(a: DocScore, b: DocScore) -> int:
    """Matches qdocs.c's _ds_cmp_score: descending score, ties broken by
    descending docid (following trec_eval)."""
    if a.score > b.score:
        return -1
    if b.score > a.score:
        return 1
    if b.docid > a.docid:
        return 1
    if b.docid < a.docid:
        return -1
    return 0


_SCORE_KEY = functools.cmp_to_key(_score_cmp)


class QueryDocs:
    """Equivalent of qdocs_t: the ranked list of documents for one query."""

    def __init__(self, qid: str):
        self.qid = qid
        self._scores: list[DocScore] = []
        self._cache: tuple[Ordering, list[DocScore]] | None = None

    def add_doc_score(self, docid: str, rank: int, score: float) -> None:
        occur = len(self._scores)
        self._scores.append(
            DocScore(docid=docid.lower(), occur=occur, rank=rank, score=score)
        )
        self._cache = None

    def num_scores(self) -> int:
        return len(self._scores)

    def get_scores(self, ordering: Ordering) -> list[DocScore]:
        if self._cache is not None and self._cache[0] is ordering:
            return self._cache[1]
        if ordering is Ordering.OCCUR:
            ordered = sorted(self._scores, key=lambda d: d.occur)
        elif ordering is Ordering.RANK:
            ordered = sorted(self._scores, key=lambda d: d.rank)
        elif ordering is Ordering.SCORE:
            ordered = sorted(self._scores, key=_SCORE_KEY)
        else:
            raise ValueError(ordering)
        self._cache = (ordering, ordered)
        return ordered
