"""Port of librbp/qrels.[ch]: TREC qrels parsing and relevance lookup."""

from __future__ import annotations

import functools
from enum import Enum
from typing import TextIO

REL_UNJUDGED = -1.0
REL_INVALID_QID = -2.0


class RelType(Enum):
    BINARY = "binary"  # collapse relevance to binary at a threshold
    FRACT = "fract"    # allow fractional relevance scores, possibly scaled
    AUTO = "auto"       # binary at threshold 1 if all rels are integral,
                        # else fractional


class QrelsFormatError(ValueError):
    pass


def _qid_cmp(a: str, b: str) -> int:
    """Matches qrels.c's qid_cmp: numeric comparison if both qids are pure
    unsigned integers, else lexicographic."""
    if a.isdigit() and b.isdigit():
        ia, ib = int(a), int(b)
        return (ia > ib) - (ia < ib)
    return (a > b) - (a < b)


class QidQrels:
    """Equivalent of qid_qrels_t: judgments for a single query."""

    def __init__(self, qrels: "Qrels"):
        self._qrels = qrels
        self._raw: dict[str, float] = {}  # docid -> raw (unadjusted) rel

    def get_rel(self, docid: str) -> float:
        raw_rel = self._raw.get(docid.lower())
        if raw_rel is None:
            return REL_UNJUDGED
        return self._qrels._adj_rel(raw_rel)

    def get_num_rel(self) -> float:
        return sum(self._qrels._adj_rel(r) for r in self._raw.values())

    def __iter__(self):
        """Iterate (docid, adjusted_rel) pairs. Order is arbitrary, as in
        the C qrels_iterator_t."""
        for docid, raw_rel in self._raw.items():
            yield docid, self._qrels._adj_rel(raw_rel)


class Qrels:
    """Equivalent of qrels_t."""

    def __init__(self):
        self._qids: dict[str, QidQrels] = {}
        self.all_rels_are_integral = True
        self.reltype = RelType.AUTO
        self.reltype_arg = 1.0
        self.max_rel = 0.0

    def set_reltype(self, reltype: RelType, reltype_arg: float) -> None:
        self.reltype = reltype
        self.reltype_arg = reltype_arg

    def _adj_rel(self, raw_rel: float) -> float:
        reltype = self.reltype
        if reltype is RelType.AUTO:
            reltype = RelType.BINARY if self.all_rels_are_integral else RelType.FRACT
        if reltype is RelType.BINARY:
            return 1.0 if raw_rel >= self.reltype_arg else 0.0
        return raw_rel * self.reltype_arg

    def get_max_rel(self) -> float:
        return self._adj_rel(self.max_rel)

    def get_num_qids(self) -> int:
        return len(self._qids)

    def get_qids(self) -> list[str]:
        return sorted(self._qids.keys(), key=functools.cmp_to_key(_qid_cmp))

    def get_qid_qrels(self, qid: str) -> QidQrels | None:
        return self._qids.get(qid.lower())

    def get_rel(self, qid: str, docid: str) -> float:
        qq = self.get_qid_qrels(qid)
        if qq is None:
            return REL_INVALID_QID
        return qq.get_rel(docid)

    def get_num_rel(self, qid: str) -> float:
        """< 0.0 means there are no judgments for this query."""
        qq = self.get_qid_qrels(qid)
        if qq is None:
            return -1.0
        return qq.get_num_rel()


def load_qrels(fp: TextIO) -> Qrels:
    qrels = Qrels()
    last_qid: str | None = None
    qq: QidQrels | None = None

    for line_num, line in enumerate(fp, start=1):
        cols = line.split()
        if len(cols) != 4:
            raise QrelsFormatError(
                f"wrong number of fields on line {line_num} of qrels file"
            )
        qid, _iter, docid, rel_str = cols

        try:
            rel = float(rel_str)
        except ValueError:
            raise QrelsFormatError(
                f"rel not a non-negative float on line {line_num} of qrels file"
            ) from None
        if rel < 0.0:
            raise QrelsFormatError(
                f"rel not a non-negative float on line {line_num} of qrels file"
            )

        if rel > qrels.max_rel:
            qrels.max_rel = rel
        if qrels.all_rels_are_integral and "." in rel_str:
            qrels.all_rels_are_integral = False

        qid = qid.lower()
        docid = docid.lower()
        if qid != last_qid:
            last_qid = qid
            qq = qrels._qids.get(qid)
            if qq is None:
                qq = QidQrels(qrels)
                qrels._qids[qid] = qq

        assert qq is not None
        if docid in qq._raw:
            raise QrelsFormatError(
                f"duplicate relevance for qid '{qid}' and docid '{docid}' "
                f"on line '{line_num}'"
            )
        qq._raw[docid] = rel

    return qrels
