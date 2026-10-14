"""Port of rbp_eval/{main.c,opt.c,help.c,desc_fmt.c}: the rbp_eval CLI.

Two deliberate scoping decisions, relative to the original opt.c/getopt
parser:

  * Usage errors (bad flag combos, missing files) are validated with
    idiomatic argparse machinery (mutually exclusive groups, etc.) rather
    than a byte-for-byte port of opt.c's hand-rolled getopt loop. The
    *validation semantics* match (e.g. -r/-s/-o are still mutually
    exclusive), but exact error wording for CLI-usage mistakes does not.
    Numeric/output fidelity is unaffected -- nothing here touches how
    qrels/runs are parsed or how RBP is computed.

  * opt.c's -T handling has an operator-precedence bug
    (`!opt->details & DETAILS_AVERAGES`, due to `!` binding tighter than
    `&`) that made its "already suppressed" duplicate-flag check into a
    no-op, and would flip averages back *on* if -T were given twice. -T is
    a plain idempotent store_true flag here, which sidesteps the bug
    rather than reproducing it.

trec_fmt.c (an alternate tabular output format) is compiled into the
original binary but never wired to any CLI option or called from main.c
-- it is dead code and is not ported.
"""

from __future__ import annotations

import argparse
import getpass
import math
import os
import socket
import sys
import time
import warnings

from ..evaluate import QidRes, Results, evaluate
from ..qrels import QrelsFormatError, RelType, load_qrels
from ..ranking import Ordering
from ..runs import RunFormatError, load_run
from ..specs import DEPTH_FULL, DepthSpecError, PersistSpecError, parse_depth, parse_persist
from ..warn import RbpEvalWarning, warn

VERSION = "0.2"

DEFAULT_DEPTH = "0"
DEFAULT_PERSIST = "0.5,0.8,0.95"

USAGE_FMT = "USAGE: %s [options] <qrels-file> <run-file>\n"
OPTIONS_STR = """\
options:
   -d DEPTH_SPEC    ranking depths to calculate rbp to.  A comma-separated
                      list of positive integers, or 0 to indicate to
                      calculate for all documents in the run.
   -p PERSIST_SPEC  user persistences to calculate rbp with.  A 
                      comma-separated list of floats in range [0.0,1.0]
   -q               print rbp values for each query (default is only give
                      the overall averages)
   -T               do not print overall averages ('T'otals)
   -o               rank documents by order of occurence in the run file.
   -r               rank documents by the rankings given in column 4 of
                      the run file.  Two documents with the same ranking
                      will be considered equal for rbp calculation
   -s               rank documents by scores given in column 5 of the run
                      file.  Two document with the same score will be 
                      considered equal for rbp calculation.  This is the
                      default ranking.
   -b THRESHOLD     convert relevance judgments in qrels file to binary
                      at threshold THRESHOLD.
   -B               convert relevance judgments in qrels file to binary
                      at threshold 1.
   -f SCALE         retain fractional relevance judgments in qrels file,
                      multiplying all relevance judgments by SCALE.
   -F               retain fractional relevance judgments in qrels file,
                      without scaling.
   -a               automatically determine relevance treatment based on
                      contents of qrels file.  If it contains only integral
                      relevances, treat as binary at threshold 1
                      (as with the -B option); if it contains
                      fractional relevances, treat as fractional without
                      scaling (as with the -F option).
   -H               do not add header comment to output.
   -W               suppress warning messages.
   -h               this help message
"""


def print_help(progname: str, stream) -> None:
    stream.write(USAGE_FMT % progname)
    stream.write(OPTIONS_STR)


class _HelpAction(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        kwargs.setdefault("nargs", 0)
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        print_help(parser.prog, sys.stdout)
        parser.exit(0)


class _ArgParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write(f"ERROR: {message}\n")
        print_help(self.prog, sys.stderr)
        self.exit(1)


def _build_parser(progname: str) -> _ArgParser:
    parser = _ArgParser(prog=progname, add_help=False)
    parser.add_argument("-h", action=_HelpAction)
    parser.add_argument("-d", dest="depth_spec", default=DEFAULT_DEPTH)
    parser.add_argument("-p", dest="persist_spec", default=DEFAULT_PERSIST)
    parser.add_argument("-q", dest="per_query", action="store_true")
    parser.add_argument("-T", dest="no_averages", action="store_true")
    parser.add_argument("-H", dest="no_header", action="store_true")
    parser.add_argument("-W", dest="no_warnings", action="store_true")

    ord_group = parser.add_mutually_exclusive_group()
    ord_group.add_argument(
        "-o", dest="ordering", action="store_const", const=Ordering.OCCUR
    )
    ord_group.add_argument(
        "-r", dest="ordering", action="store_const", const=Ordering.RANK
    )
    ord_group.add_argument(
        "-s", dest="ordering", action="store_const", const=Ordering.SCORE
    )

    rel_group = parser.add_mutually_exclusive_group()
    rel_group.add_argument("-b", dest="b_threshold", type=float, metavar="THRESHOLD")
    rel_group.add_argument("-B", dest="B", action="store_true")
    rel_group.add_argument("-f", dest="f_scale", type=float, metavar="SCALE")
    rel_group.add_argument("-F", dest="F", action="store_true")
    rel_group.add_argument("-a", dest="a", action="store_true")

    parser.add_argument("files", nargs="*", metavar="<qrels-file> <run-file>")
    return parser


def _resolve_reltype(args: argparse.Namespace, parser: _ArgParser) -> tuple[RelType, float]:
    if args.b_threshold is not None:
        if args.b_threshold < 0.0:
            parser.error("invalid argument to -b option")
        return RelType.BINARY, args.b_threshold
    if args.B:
        return RelType.BINARY, 1.0
    if args.f_scale is not None:
        if args.f_scale <= 0.0:
            parser.error("invalid argument to -f option")
        return RelType.FRACT, args.f_scale
    if args.F:
        return RelType.FRACT, 1.0
    return RelType.AUTO, 1.0


def _desc_header(argv: list[str], run_fname: str, fp) -> None:
    now = time.time()
    fp.write("# rbp_eval output\n")
    fp.write("#\n")
    fp.write(f"# {time.ctime(now)}\n")
    fp.write("# command:" + "".join(f" {a}" for a in argv) + "\n")
    fp.write(f"# hostname: {socket.gethostname()}\n")
    fp.write(f"# rbp_eval version: {VERSION}\n")
    fp.write(f"# current working directory: {os.getcwd()}\n")
    try:
        fp.write(f"# user: {getpass.getuser()}\n")
    except Exception:
        pass
    try:
        mtime = os.stat(run_fname).st_mtime
        fp.write(f"# {run_fname} modification time: {time.ctime(mtime)}\n")
    except OSError:
        pass


def _desc_fmt_qid(qres: QidRes, depths: list[float], persist: list[float], fp) -> None:
    for d, depth in enumerate(depths):
        depth_str = "full" if depth == DEPTH_FULL else str(int(depth))
        dres = qres.depth_res[d]
        for p, pv in enumerate(persist):
            pres = dres.persist_res[p]
            fp.write(
                f"p= {pv:.2f} q= {qres.qid:>4} d= {depth_str:>4} "
                f"rbp= {pres.sum:.4f} +{pres.err:.4f}\n"
            )


def desc_fmt(
    res: Results,
    *,
    argv: list[str],
    run_fname: str,
    per_query: bool,
    show_averages: bool,
    no_header: bool,
    fp,
) -> None:
    if not no_header:
        _desc_header(argv, run_fname, fp)

    for qres in res.qid_res:
        if qres.num_rel < 0:
            warn(f"there are no judgments for query id {qres.qid}")
            continue
        if per_query:
            _desc_fmt_qid(qres, res.depths, res.persist, fp)

    if show_averages:
        _desc_fmt_qid(res.ave_res, res.depths, res.persist, fp)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    progname = sys.argv[0] if len(sys.argv) > 0 else "rbp_eval"
    parser = _build_parser(progname)
    args = parser.parse_args(argv)

    try:
        depths = parse_depth(args.depth_spec)
        if not depths:
            parser.error("empty depth specification")
    except DepthSpecError as e:
        parser.error(str(e))
    try:
        persist = parse_persist(args.persist_spec)
        if not persist:
            parser.error("empty persist specification")
    except PersistSpecError as e:
        parser.error(str(e))

    ordering = args.ordering if args.ordering is not None else Ordering.SCORE
    reltype, reltype_arg = _resolve_reltype(args, parser)

    if len(args.files) < 2:
        parser.error("both qrels and run files must be specified")
    if len(args.files) > 2:
        parser.error("trailing arguments")
    qrels_fname, run_fname = args.files

    if args.no_warnings:
        warnings.simplefilter("ignore", RbpEvalWarning)

    if not os.path.isfile(qrels_fname):
        sys.stderr.write(f"Unable to open qrels file {qrels_fname} for reading\n")
        print_help(progname, sys.stderr)
        return 1
    if not os.path.isfile(run_fname):
        sys.stderr.write(f"Unable to open run file {run_fname} for reading\n")
        print_help(progname, sys.stderr)
        return 1

    try:
        with open(run_fname) as fp:
            run = load_run(fp)
    except RunFormatError as e:
        sys.stderr.write(f"Error loading run file: {e}\n")
        print_help(progname, sys.stderr)
        return 1

    try:
        with open(qrels_fname) as fp:
            qrels = load_qrels(fp)
    except QrelsFormatError as e:
        sys.stderr.write(f"Error loading qrels: {e}\n")
        print_help(progname, sys.stderr)
        return 1

    qrels.set_reltype(reltype, reltype_arg)

    max_rel = qrels.get_max_rel()
    if max_rel > 1.0:
        warn(f"maximum effective relevance of {max_rel:.2f} exceeds 1.0")

    res = evaluate(qrels, run, ordering, persist, depths)

    desc_fmt(
        res,
        argv=[progname, *argv],
        run_fname=run_fname,
        per_query=args.per_query,
        show_averages=not args.no_averages,
        no_header=args.no_header,
        fp=sys.stdout,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
