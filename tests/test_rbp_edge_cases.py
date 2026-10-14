"""Regression test for the bug fix documented in rbp_eval.rbp's module
docstring: requesting multiple depth cutoffs beyond a query's actual
number of retrieved documents must not crash or produce NaN."""

from __future__ import annotations

import io
import math

from rbp_eval.evaluate import evaluate
from rbp_eval.qrels import load_qrels
from rbp_eval.ranking import Ordering
from rbp_eval.runs import load_run

QRELS = "1 0 d1 1\n1 0 d2 1\n1 0 d3 0\n"
# Only 3 documents retrieved for qid 1, but depths go well past that.
RUN = "1 0 d1 1 3.0 sys\n1 0 d2 2 2.0 sys\n1 0 d3 3 1.0 sys\n"


def test_depths_beyond_num_retrieved_do_not_crash_or_nan():
    qrels = load_qrels(io.StringIO(QRELS))
    run = load_run(io.StringIO(RUN))

    res = evaluate(qrels, run, Ordering.SCORE, persist=[0.8], depths=[2, 5, 10])

    qres = res.qid_res[0]
    assert qres.num_ret == 3
    for dres in qres.depth_res:
        for pres in dres.persist_res:
            assert math.isfinite(pres.sum)
            assert math.isfinite(pres.err)

    # Depths 5 and 10 both exceed num_scores=3, so they should be
    # identical (nothing new to add past the end of the ranking) --
    # this is exactly the repeated-capped-depth case the fix targets.
    d5, d10 = qres.depth_res[1], qres.depth_res[2]
    assert d5.num_rel_ret == d10.num_rel_ret
    assert d5.persist_res[0].sum == d10.persist_res[0].sum
    assert d5.persist_res[0].err == d10.persist_res[0].err
