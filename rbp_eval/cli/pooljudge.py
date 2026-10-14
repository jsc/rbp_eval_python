"""Port of rbp_util/pooljudge.c: choose documents to judge using the
standard pooling technique (judge every document that appears in any
run's top-d ranking, for increasing d).

The original tracks a separate `judged_hash` of (docid, qidd) pairs
revealed this session purely to avoid calling runerr_doc_judged() twice
on the same pair (which would trip its internal assertion) when multiple
runs share a document at the current depth. RunErr already tracks exactly
this in `docs_judged`, updated in lockstep every time judged_hash would
have been, so this port just queries that directly instead of keeping a
second, redundant copy.
"""

from __future__ import annotations

import argparse
import sys

from ..ranking import Ordering
from ..rbp_util.common import (
    add_common_arguments,
    build_common_options,
    init_runerr,
    report_runerr_stats,
)
from ..rbp_util.runerr import RunErrInitError

DEFAULT_POOL_DEPTH = 1000
DEFAULT_ERR_LOG_INTERVAL = 1000


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    progname = sys.argv[0] if len(sys.argv) > 0 else "pooljudge"

    parser = argparse.ArgumentParser(prog=progname)
    add_common_arguments(parser)
    parser.add_argument("-d", dest="pool_depth", type=int, default=DEFAULT_POOL_DEPTH)
    parser.add_argument(
        "-I", dest="report_at_increment", action="store_false", default=True
    )
    parser.add_argument("run_files", nargs="*")
    args = parser.parse_args(argv)

    opts = build_common_options(args, parser)

    try:
        runerr = init_runerr(opts, args.run_files)
    except RunErrInitError as e:
        sys.stderr.write(f"Error initialising runerr: {e}\n")
        return 1

    num_qids = opts.qidid.num_ids()
    runarr = runerr.get_runs()

    prev_num_judged = 0
    prev_num_relevant = 0.0
    avg_err = 1.0
    finished = False

    d = 0
    while (
        not finished
        and d < args.pool_depth
        and (opts.min_avg_err == 0.0 or avg_err >= opts.min_avg_err)
    ):
        for ri in runarr:
            if finished:
                break
            if not ri.contributes_judgments:
                continue
            run = ri.run
            for q in range(run.num_qdocs()):
                if finished:
                    break
                qd = run.get_qdocs_by_index(q)
                if d >= qd.num_scores():
                    continue
                ds = qd.get_scores(Ordering.SCORE)  # QDOCS_DEFAULT_ORDERING
                docid = ds[d].docid
                qidd = opts.qidid.lookup_id(qd.qid)
                if qidd is None:
                    continue
                key = (qidd, docid)
                if key not in runerr.docs_judged:
                    keep_going = runerr.doc_judged(docid, qidd, num_qids)
                    if not args.report_at_increment:
                        if runerr.num_judged % DEFAULT_ERR_LOG_INTERVAL == 0:
                            avg_err, prev_num_judged, prev_num_relevant = (
                                report_runerr_stats(
                                    sys.stdout, runerr, prev_num_judged, prev_num_relevant
                                )
                            )
                    if not keep_going:
                        finished = True
        if args.report_at_increment:
            avg_err, prev_num_judged, prev_num_relevant = report_runerr_stats(
                sys.stdout, runerr, prev_num_judged, prev_num_relevant
            )
        d += 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
