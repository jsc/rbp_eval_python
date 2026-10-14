"""Port of librbp/run.[ch]: TREC run parsing."""

from __future__ import annotations

import math
from typing import TextIO

from .ranking import QueryDocs
from .warn import warn


class RunFormatError(ValueError):
    pass


class Run:
    """Equivalent of run_t."""

    def __init__(self):
        self._qdocs: dict[str, QueryDocs] = {}
        self._order: list[str] = []
        self.runid: str = ""

    def num_qdocs(self) -> int:
        return len(self._order)

    def get_qdocs_by_index(self, index: int) -> QueryDocs:
        return self._qdocs[self._order[index]]

    def get_qdocs_by_qid(self, qid: str) -> QueryDocs | None:
        return self._qdocs.get(qid.lower())

    def get_runid(self) -> str:
        return self.runid


def load_run(fp: TextIO, *, only_qid: str | None = None) -> Run:
    """Load a TREC run file.

    ``only_qid``, if given, restricts loading to a single query id
    (mirrors load_run_single_query).
    """
    run = Run()
    last_qid: str | None = None
    qd: QueryDocs | None = None
    runid: str | None = None
    warned_about_runid = False
    warned_about_commas = False

    for line_num, line in enumerate(fp, start=1):
        cols = line.split()
        if len(cols) != 6:
            raise RunFormatError(
                f"wrong number of fields on line {line_num} of run file"
            )
        qid, _iter, docid, rank_str, score_str, this_runid = cols
        qid = qid.lower()

        if only_qid is not None and only_qid != qid:
            continue

        if qid != last_qid:
            last_qid = qid
            qd = run._qdocs.get(qid)
            if qd is None:
                qd = QueryDocs(qid)
                run._qdocs[qid] = qd
                run._order.append(qid)

        try:
            score = float(score_str)
        except ValueError:
            raise RunFormatError(
                f"score '{score_str}' not a floating point number on line "
                f"{line_num} of run file"
            ) from None
        if math.isnan(score):
            # The odd run puts in a single document with a score of NaN as
            # a placeholder for a query it otherwise wouldn't have results
            # for.
            warn(
                f"score for qid {qid}, docid {docid}, on line {line_num} "
                f"of run file is NaN, converting to 0.0"
            )
            score = 0.0

        rank_field = rank_str
        if "," in rank_field:
            # Incredibly, some runs format ranks with commas.
            if not warned_about_commas:
                warn(f"comma(s) in rank on line {line_num} of run file, stripping")
                warned_about_commas = True
            rank_field = rank_field.replace(",", "")
        try:
            rank = int(rank_field)
        except ValueError:
            raise RunFormatError(
                f"rank not a non-negative integer on line {line_num} of run file"
            ) from None
        if rank < 0:
            raise RunFormatError(
                f"rank not a non-negative integer on line {line_num} of run file"
            )

        if runid is not None and this_runid != runid and not warned_about_runid:
            warn(
                f"runid changes on line {line_num} of run file: "
                f"was '{runid}', is now '{this_runid}'"
            )
            warned_about_runid = True
        elif runid is None:
            runid = this_runid

        assert qd is not None
        qd.add_doc_score(docid, rank, score)

    run.runid = runid or ""
    return run
