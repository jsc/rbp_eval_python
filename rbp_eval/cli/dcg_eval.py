"""Port of dcg_eval/main.cpp.

One deliberate scoping decision: the original reads -b/-r values with
atof/atoi, which silently parse invalid input as 0.0/0 rather than
erroring. argparse's type=float/int is used here instead, which raises a
clear error on invalid input -- stricter than the original, not more
permissive, so it can't hide a typo'd flag value as a silent zero.
"""

from __future__ import annotations

import math
import sys

from ..dcg import DcgOptions, compute_dcg
from ..qrels import QrelsFormatError, RelType, load_qrels
from ..ranking import Ordering
from ..runs import RunFormatError, load_run

USAGE_FMT = "USAGE: %s [-R] [-b base] [-r depth] [-n|N] [-d|D] [-m|M] <qrels> <run>\n"

DEFAULT_DEPTH = 1000
DEFAULT_BASE = 2.0


def _usage_and_exit(progname: str, stream, code: int) -> None:
    stream.write(USAGE_FMT % progname)
    sys.exit(code)


def main(argv: list[str] | None = None) -> int:
    import argparse

    argv = sys.argv[1:] if argv is None else argv
    progname = sys.argv[0] if len(sys.argv) > 0 else "dcg_eval"

    parser = argparse.ArgumentParser(prog=progname, add_help=False)
    parser.set_defaults(
        rank_adjust=0, base=DEFAULT_BASE, depth=DEFAULT_DEPTH,
        normalised=True, discounted=True, reltype=RelType.AUTO,
    )
    parser.add_argument("-h", dest="help", action="store_true")
    parser.add_argument("-R", dest="rank_adjust", action="store_const", const=1)
    parser.add_argument("-b", dest="base", type=float)
    parser.add_argument("-r", dest="depth", type=int)
    parser.add_argument("-n", dest="normalised", action="store_true")
    parser.add_argument("-N", dest="normalised", action="store_false")
    parser.add_argument("-d", dest="discounted", action="store_true")
    parser.add_argument("-D", dest="discounted", action="store_false")
    parser.add_argument("-m", dest="reltype", action="store_const", const=RelType.FRACT)
    parser.add_argument("-M", dest="reltype", action="store_const", const=RelType.BINARY)
    parser.add_argument("files", nargs="*")

    try:
        args = parser.parse_args(argv)
    except SystemExit:
        _usage_and_exit(progname, sys.stderr, 1)
        return 1

    if args.help:
        _usage_and_exit(progname, sys.stdout, 0)

    if len(args.files) != 2:
        _usage_and_exit(progname, sys.stderr, 1)
    qrels_fname, run_fname = args.files

    try:
        with open(run_fname) as fp:
            run = load_run(fp)
    except OSError:
        sys.stderr.write(f"Unable to open run file '{run_fname}' for reading\n")
        _usage_and_exit(progname, sys.stderr, 1)
        return 1
    except RunFormatError as e:
        sys.stderr.write(f"Error loading run file '{run_fname}': {e}\n")
        _usage_and_exit(progname, sys.stderr, 1)
        return 1

    try:
        with open(qrels_fname) as fp:
            qrels = load_qrels(fp)
    except OSError:
        sys.stderr.write(f"Unable to open qrels file '{qrels_fname}' for reading\n")
        _usage_and_exit(progname, sys.stderr, 1)
        return 1
    except QrelsFormatError as e:
        sys.stderr.write(f"Error loading qrels file '{qrels_fname}': {e}\n")
        _usage_and_exit(progname, sys.stderr, 1)
        return 1

    qrels.set_reltype(args.reltype, 1.0)

    opts = DcgOptions(
        base=args.base,
        depth=args.depth,
        rank_adjust=args.rank_adjust,
        discounted=args.discounted,
        normalised=args.normalised,
    )

    tot_score = 0.0
    num_queries = 0
    for q in range(run.num_qdocs()):
        qdocs = run.get_qdocs_by_index(q)
        doc_scores = qdocs.get_scores(Ordering.SCORE)
        score = compute_dcg(qrels, qdocs.qid, doc_scores, opts)
        if score is None:
            sys.stderr.write(f"No judgments for query {qdocs.qid}; ignoring\n")
            continue
        num_queries += 1
        tot_score += score
        print(f"{qdocs.qid} {score:g}")

    avg = tot_score / num_queries if num_queries > 0 else math.nan
    print(f"all {avg:g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
