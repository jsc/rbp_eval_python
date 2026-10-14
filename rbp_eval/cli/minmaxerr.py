"""Port of rbp_util/minmaxerr.c: choose documents to judge so as to
minimise the maximum (weighted) error across runs.

Repeatedly picks whichever run currently has the largest weighted error,
judges the next not-yet-judged document at that run's current frontier
depth (advancing the frontier past any queries that turn out to already
be judged, e.g. via a shared document from another run), then re-picks.
"""

from __future__ import annotations

import argparse
import sys

from ..ranking import Ordering
from ..rbp_util.common import (
    add_common_arguments,
    build_common_options,
    init_runerr,
    load_report_depths,
    report_runerr_stats,
)
from ..rbp_util.runerr import RunErrInitError, Weight

DEFAULT_DEPTH = 1000


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    progname = sys.argv[0] if len(sys.argv) > 0 else "minmaxerr"

    parser = argparse.ArgumentParser(prog=progname)
    add_common_arguments(parser)
    parser.add_argument("-D", dest="report_depths_file", default=None)
    parser.add_argument("-w", dest="weighting", action="store_const", const=Weight.QUADRATIC)
    parser.add_argument(
        "-W",
        dest="weighting_name",
        choices=["quadratic", "linear", "uniform"],
        default=None,
    )
    parser.add_argument("run_files", nargs="*")
    args = parser.parse_args(argv)

    opts = build_common_options(args, parser)

    weighting = Weight.UNIFORM
    if args.weighting is not None:
        weighting = args.weighting
    if args.weighting_name is not None:
        weighting = Weight(args.weighting_name)

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

    try:
        runerr = init_runerr(opts, args.run_files)
    except RunErrInitError as e:
        sys.stderr.write(f"Error initialising runerr: {e}\n")
        return 1

    runarr = runerr.get_runs()
    num_qids = opts.qidid.num_ids()

    prev_num_judged = 0
    prev_num_relevant = 0.0
    avg_err = 1.0
    finished = False
    reached_depth = False
    short_run = False
    rd = 0
    d = 0

    while (
        not short_run
        and not reached_depth
        and (not report_depths or rd < len(report_depths))
        and (opts.min_avg_err == 0.0 or avg_err >= opts.min_avg_err)
    ):
        short_run = True
        _savg, _sdev, _smax, max_run_d, _nj, _nr, _nl = runerr.stats_wgt(weighting)
        ri = runarr[max_run_d]

        if (not report_depths and d % 1000 == 0) or (
            report_depths and report_depths[rd] == d
        ):
            avg_err, prev_num_judged, prev_num_relevant = report_runerr_stats(
                sys.stdout, runerr, prev_num_judged, prev_num_relevant
            )
            rd += 1

        num_qdocs = ri.run.num_qdocs()
        while True:
            if ri.judged_depth == DEFAULT_DEPTH:
                reached_depth = True
                sys.stderr.write(
                    f"Exhausted judgments for run {ri.run.get_runid()} at depth "
                    f"{DEFAULT_DEPTH}: {ri.rbp:.4f} +{ri.err:.4f}\n"
                )
                break

            judged_one = False
            q = 0
            while not finished and q < num_qdocs and not judged_one:
                qd = ri.run.get_qdocs_by_index(q)
                numranks = qd.num_scores()
                if ri.judged_depth < numranks:
                    short_run = False
                    ds = qd.get_scores(Ordering.SCORE)  # QDOCS_DEFAULT_ORDERING
                    if ds[ri.judged_depth].flags == 0:
                        judged_one = True
                        qid = qd.qid
                        qidd = opts.qidid.get_id(qid)
                        if not runerr.doc_judged(ds[ri.judged_depth].docid, qidd, num_qids):
                            finished = True
                        dococ = runerr.get_dococcur_items(
                            ds[ri.judged_depth].docid, qidd
                        )
                        for item in dococ:
                            item.data.flags = 1
                q += 1

            if not judged_one:
                ri.judged_depth += 1

            if finished or reached_depth or judged_one or short_run:
                break

        if short_run:
            sys.stderr.write(
                f"WARNING: finishing processing as run {ri.run.get_runid()} has "
                f"no more judgments at depth {ri.judged_depth}\n"
            )

        d += 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
