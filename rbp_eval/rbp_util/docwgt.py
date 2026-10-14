"""Port of rbp_util/docwgt.[ch]: accumulates, per (qid, docid) pair, the
total RBP positional weight it earns across a pool of runs -- used to
identify which documents are most "valuable" to judge next.

The original caches a growable array of precomputed RBP weights to avoid
recomputing pow() per call; that's a micro-optimization with no
correctness implication, so this just calls rbp_point() directly per
occurrence instead of maintaining that cache.

get_entries(sort=True)'s tie order for equal-weight documents cannot be
made to match the original bit-for-bit: the C comparator returns 0 for
equal weights and is applied via qsort, which is not a stable sort and
documents no tie-break -- so the original's own tie order is
implementation-defined, not a reproducible target. (Confirmed by
example: in the test fixtures, docs x091/qid0 and x002/qid1 tie exactly
at weight 0.104751; minavgerr's default judging order swaps which one
comes first between the C binary and this port, shifting the "Unjudged"
judgment-count reported for whichever comes second by one. Every
downstream rbp/err number this feeds into still matches exactly -- see
tests/test_minavgerr_cli.py.)
"""

from __future__ import annotations

from dataclasses import dataclass

from ..rbp import rbp_point
from ..ranking import Ordering
from ..runs import Run
from .strid import Strid


@dataclass
class DocWgtElem:
    docid: str
    qidd: int
    wgt: float


class DocWgt:
    def __init__(self, persist: float):
        self.persist = persist
        self._wgts: dict[tuple[int, str], float] = {}

    def add_run(self, run: Run, qidid: Strid) -> None:
        for q in range(run.num_qdocs()):
            qdocs = run.get_qdocs_by_index(q)
            qidd = qidid.lookup_id(qdocs.qid)
            if qidd is None:
                continue
            scores = qdocs.get_scores(Ordering.SCORE)  # QDOCS_DEFAULT_ORDERING
            for rank, ds in enumerate(scores):
                self._add_doc(ds.docid, qidd, rank)

    def _add_doc(self, docid: str, qidd: int, rank: int) -> None:
        key = (qidd, docid)
        self._wgts[key] = self._wgts.get(key, 0.0) + rbp_point(self.persist, rank)

    def num_entries(self) -> int:
        return len(self._wgts)

    def get_entries(self, *, sort: bool = False) -> list[DocWgtElem]:
        entries = [
            DocWgtElem(docid=docid, qidd=qidd, wgt=wgt)
            for (qidd, docid), wgt in self._wgts.items()
        ]
        if sort:
            # Descending by weight, as the original's qsort comparator.
            entries.sort(key=lambda e: -e.wgt)
        return entries
