"""Port of rbp_util/minavgerr.c: choose documents to judge so as to
minimise average error.

Default (uniform run/query weighting): documents are judged strictly in
descending order of their aggregate RBP weight across the run pool (the
`docwgt` accumulator) -- the classic "judge the highest-value document
first" strategy.

Non-uniform weighting: rather than recomputing every remaining document's
weighted value after every single judgment (which the run/query weights
depend on), the original recomputes in batches of `5 * num_qids`
documents at a time, selecting that batch's top scorers via a
bounded min-heap with an early-termination prune (safe because
`docwgt_docs` is pre-sorted by raw, un-weighted RBP weight: once a
document's raw weight times the best possible run/query multiplier can no
longer beat the heap's current floor, no later, lower-raw-weight document
can either). This is an intentional approximation in the original, not a
simplification made here -- it does not exactly match an idealized
"always recompute after every judgment" strategy, so faithfully
reproducing this exact batching (not a cleaner alternative) is what
matches the original's real behaviour.

Python's heapq ties on the (score, doc_index) pair when scores are equal,
which is a fine, deterministic substitute for the original's
implementation-defined binary-heap tie order (the C dblheap doesn't
document or guarantee one either).
"""

from __future__ import annotations

import argparse
import heapq
import sys

from ..rbp import rbp_weights
from ..rbp_util.common import (
    add_common_arguments,
    build_common_options,
    init_runerr,
    load_report_depths,
    report_runerr_stats,
)
from ..rbp_util.docwgt import DocWgt
from ..rbp_util.runerr import QueryWeight, RunErrInitError, Weight

DEFAULT_ERR_LOG_INTERVAL = 1000
WEIGHTED_RECALC_INTERVAL_MULT = 5
MAX_DEPTH = 10000

_WEIGHT_NAME_MAP = {
    "quadratic": Weight.QUADRATIC,
    "linear": Weight.LINEAR,
    "uniform": Weight.UNIFORM,
    "projected": Weight.PROJECTED,
    "residual": Weight.RESIDUAL,
    "residual-and-midpoint": Weight.RESIDUAL_AND_MIDPOINT,
    "residual-and-midpoint-squared": Weight.RESIDUAL_AND_MIDPOINT_SQ,
    "residual-and-midpoint-cubed": Weight.RESIDUAL_AND_MIDPOINT_CB,
    "residual-and-projected": Weight.RESIDUAL_AND_PROJECTED,
    "residual-and-projected-squared": Weight.RESIDUAL_AND_PROJECTED_SQ,
    "midpoint-squared": Weight.MIDPOINT_SQ,
}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    progname = sys.argv[0] if len(sys.argv) > 0 else "minavgerr"

    parser = argparse.ArgumentParser(prog=progname)
    add_common_arguments(parser)
    parser.add_argument("-D", dest="report_depths_file", default=None)
    parser.add_argument("-w", dest="w_flag", action="store_true")
    parser.add_argument("-W", dest="weighting_name", choices=sorted(_WEIGHT_NAME_MAP), default=None)
    parser.add_argument(
        "-q", dest="qry_weighting_name", choices=["uniform", "linear"], default=None
    )
    parser.add_argument("run_files", nargs="*")
    args = parser.parse_args(argv)

    opts = build_common_options(args, parser)

    weighting = Weight.UNIFORM
    if args.w_flag:
        weighting = Weight.QUADRATIC
    if args.weighting_name is not None:
        weighting = _WEIGHT_NAME_MAP[args.weighting_name]

    qry_weighting = QueryWeight.UNIFORM
    if args.qry_weighting_name is not None:
        qry_weighting = QueryWeight(args.qry_weighting_name)

    report_depths: list[int] = []
    if args.report_depths_file is not None:
        try:
            report_depths = load_report_depths(args.report_depths_file)
        except RunErrInitError as e:
            sys.stderr.write(f"{e}\n")
            return 1

    if not args.run_files:
        sys.stderr.write(f"USAGE: {progname} <run>...\n")
        return 1

    docwgt = DocWgt(opts.persist)

    try:
        runerr = init_runerr(opts, args.run_files)
    except RunErrInitError as e:
        sys.stderr.write(f"Error initialising runerr: {e}\n")
        return 1

    runarr = runerr.get_runs()
    for ri in runarr:
        if ri.contributes_judgments:
            docwgt.add_run(ri.run, opts.qidid)

    num_qids = opts.qidid.num_ids()

    docwgt_docs = docwgt.get_entries(sort=True)
    num_ranked_docs = len(docwgt_docs)
    # Track per-entry "already judged" state alongside the sorted list
    # (equivalent to the original's docwgt_elem_t.flags field).
    judged_flags = [False] * num_ranked_docs

    weighted = weighting is not Weight.UNIFORM or qry_weighting is not QueryWeight.UNIFORM
    weighted_recalc_interval = 0
    wgt_heap: list[tuple[float, int]] = []
    wgt_list: list[int] = []
    if weighted:
        weighted_recalc_interval = max(2, WEIGHTED_RECALC_INTERVAL_MULT * num_qids)

    rbp_wgts = rbp_weights(opts.persist, MAX_DEPTH)

    prev_num_judged = 0
    prev_num_relevant = 0.0
    avg_err = 1.0
    finished = False
    rd = 0
    d = 0

    while (
        not finished
        and d < num_ranked_docs
        and (not report_depths or rd < len(report_depths))
        and (opts.min_avg_err == 0.0 or avg_err >= opts.min_avg_err)
    ):
        if (not report_depths and d % DEFAULT_ERR_LOG_INTERVAL == 0) or (
            report_depths and report_depths[rd] == d
        ):
            avg_err, prev_num_judged, prev_num_relevant = report_runerr_stats(
                sys.stdout, runerr, prev_num_judged, prev_num_relevant
            )
            rd += 1

        if weighted:
            if not wgt_list:
                max_prop_err, _max_run_d = runerr.max_prop_wgt(weighting)
                run_wgts: dict[int, float] = {}
                qry_wgts: dict[int, float] = {}
                threshold = -1.0

                for d2 in range(num_ranked_docs):
                    dw = docwgt_docs[d2]
                    if judged_flags[d2]:
                        continue
                    if dw.wgt * max_prop_err < threshold:
                        break
                    dcrarr = runerr.get_dococcur_items(dw.docid, dw.qidd)
                    err = 0.0
                    for item in dcrarr:
                        if runarr[item.rund].contributes_judgments:
                            if item.rund not in run_wgts:
                                run_wgts[item.rund] = runerr.prop_err_wgt_for_run(
                                    item.rund, weighting
                                )
                            if dw.qidd not in qry_wgts:
                                qry_wgts[dw.qidd] = runerr.wgt_for_qry(
                                    dw.qidd, qry_weighting
                                )
                            err_wgt = (
                                rbp_wgts[item.rank]
                                * run_wgts[item.rund]
                                * qry_wgts[dw.qidd]
                            )
                            err += err_wgt

                    if len(wgt_heap) >= weighted_recalc_interval:
                        if err > threshold:
                            heapq.heappushpop(wgt_heap, (err, d2))
                            threshold = wgt_heap[0][0]
                    else:
                        heapq.heappush(wgt_heap, (err, d2))

                # Drain the heap smallest-first, then consume from the
                # end of wgt_list (largest first) below.
                while wgt_heap:
                    _err, doc_idx = heapq.heappop(wgt_heap)
                    wgt_list.append(doc_idx)

            assert wgt_list
            judge_next = wgt_list.pop()
        else:
            judge_next = d

        dw = docwgt_docs[judge_next]
        judged_flags[judge_next] = True
        if not runerr.doc_judged(dw.docid, dw.qidd, num_qids):
            finished = True

        d += 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
