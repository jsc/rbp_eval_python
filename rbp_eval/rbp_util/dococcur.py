"""Port of rbp_util/dococcur.[ch]: records every (run, rank) occurrence of
each (qid, docid) pair across a pool of runs.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from ..ranking import Ordering
from ..runs import Run
from .strid import Strid


@dataclass
class DocOccurItem:
    rund: int
    rank: int
    data: Any = None


class DocOccur:
    def __init__(self):
        self._items: dict[tuple[int, str], list[DocOccurItem]] = defaultdict(list)

    def add(
        self, docid: str, qidd: int, rund: int, rank: int, data: Any = None
    ) -> None:
        self._items[(qidd, docid)].append(DocOccurItem(rund=rund, rank=rank, data=data))

    def add_run(self, run: Run, rund: int, qidid: Strid) -> None:
        for q in range(run.num_qdocs()):
            qdocs = run.get_qdocs_by_index(q)
            qidd = qidid.lookup_id(qdocs.qid)
            if qidd is None:
                continue
            scores = qdocs.get_scores(Ordering.SCORE)  # QDOCS_DEFAULT_ORDERING
            for rank, ds in enumerate(scores):
                self.add(ds.docid, qidd, rund, rank, ds)

    def get(self, docid: str, qidd: int) -> list[DocOccurItem] | None:
        return self._items.get((qidd, docid))
