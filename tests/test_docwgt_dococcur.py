"""Unit tests replicating the assertions in the original C's DOCWGT_MAIN
and DOCOCCUR_MAIN test drivers (docwgt.c / dococcur.c) -- those were only
`check_PROGRAMS` unit tests, not standalone CLI tools, so there's no
compiled binary to differential-test against; these are direct
equivalents of the same assertions instead."""

from __future__ import annotations

import pytest

from rbp_eval.rbp_util.docwgt import DocWgt
from rbp_eval.rbp_util.dococcur import DocOccur


def test_docwgt_matches_c_main():
    dw = DocWgt(0.95)
    wgts = [(1 - 0.95) * (1.0 + 0.95), (1 - 0.95) * 1.0]

    dw._add_doc("d1", 3, 0)
    dw._add_doc("d2", 3, 0)
    dw._add_doc("d2", 4, 2)
    dw._add_doc("d1", 3, 1)

    assert dw.num_entries() == 3
    assert dw._wgts[(3, "d1")] == pytest.approx(wgts[0])
    assert dw._wgts[(3, "d2")] == pytest.approx(wgts[1])
    assert (4, "d2") in dw._wgts

    entries = dw.get_entries(sort=True)
    assert len(entries) == 3
    assert (entries[0].docid, entries[0].qidd) == ("d1", 3)
    assert (entries[1].docid, entries[1].qidd) == ("d2", 3)
    assert entries[2].qidd == 4
    assert entries[0].wgt == pytest.approx(wgts[0])
    assert entries[1].wgt == pytest.approx(wgts[1])


def test_dococcur_matches_c_main():
    dcr = DocOccur()
    dcr.add("d1", 4, 0, 2)
    dcr.add("d1", 4, 0, 1)
    dcr.add("d2", 2, 1, 4)
    dcr.add("d2", 2, 10, 8)
    dcr.add("d1", 2, 11, 3)

    da = dcr.get("d1", 4)
    assert len(da) == 2
    assert (da[0].rund, da[0].rank) == (0, 2)
    assert (da[1].rund, da[1].rank) == (0, 1)

    da = dcr.get("d1", 2)
    assert len(da) == 1
    assert (da[0].rund, da[0].rank) == (11, 3)

    da = dcr.get("d2", 2)
    assert len(da) == 2
    assert (da[0].rund, da[0].rank) == (1, 4)
    assert (da[1].rund, da[1].rank) == (10, 8)

    assert dcr.get("d3", 0) is None
    assert dcr.get("d1", 0) is None
