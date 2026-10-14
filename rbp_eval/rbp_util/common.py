"""Port of rbp_util/common.[ch]: shared CLI option handling and runerr
bootstrapping used by minavgerr/minmaxerr/pooljudge.

COMMON_OPTS's getopt dispatch protocol (0=unhandled/1=handled/-1=error,
so each tool's own flags and the shared ones can share one getopt loop)
has no equivalent need in Python: each CLI's argparse parser just
registers the common arguments (via add_common_arguments) alongside its
own, in one parser, so there's nothing to dispatch between.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass, field

from ..qrels import QrelsFormatError, load_qrels
from ..stats import (
    paired_bootstrap_test_p,
    paired_sign_test_p,
    paired_t_test_p,
    paired_wilcoxon_test_p,
)
from .runerr import RunErr, RunErrInitError, SignifMode, _safe_div
from .strid import Strid

MAX_DEPTH = 10000
DEFAULT_PERSIST = 0.95
DEFAULT_SCORE_LOG_INTERVAL = 100
DEFAULT_SIGNIFICANCE_LOG_INTERVAL = 100
DEFAULT_SIGNIFICANCE_MODE = SignifMode.PESSIMAL
DEFAULT_SIGNIFICANCE_PROPORTION = 1.0
DEFAULT_SIGNIFICANCE_P_THRESHOLD = 0.05

DEFAULT_CHART_DEPTH = 100
DEFAULT_CHART_JUDGMENTS = 500

SIGNIF_FUNCTIONS = {
    "wilcoxon": paired_wilcoxon_test_p,
    "sign": paired_sign_test_p,
    "t": paired_t_test_p,
    "bootstrap": paired_bootstrap_test_p,
}
DEFAULT_SIGNIFICANCE_FUNCTION_NAME = "wilcoxon"

_ATOI_RE = re.compile(r"\s*([+-]?\d+)")


def _atoi(s: str) -> int:
    """Mimics C's atoi(): leading whitespace/sign/digits, trailing
    garbage ignored, 0 if nothing parses -- never raises."""
    m = _ATOI_RE.match(s)
    return int(m.group(1)) if m else 0


def load_report_depths(fname: str) -> list[int]:
    try:
        fp = open(fname)
    except OSError:
        raise RunErrInitError(
            f"Unable to open reporting depths specifier file '{fname}' for reading"
        ) from None
    depths: list[int] = []
    with fp:
        last_depth = -1
        for line in fp:
            depth = _atoi(line)
            if depth <= last_depth:
                raise RunErrInitError("Report depths must be in strictly increasing order")
            depths.append(depth)
            last_depth = depth
    return depths


@dataclass
class CommonOptions:
    persist: float = DEFAULT_PERSIST
    min_avg_err: float = 0.0
    chart_dir: str | None = None
    score_log_fp = None
    score_log_interval: int = DEFAULT_SCORE_LOG_INTERVAL
    judgment_log_fp = None
    qrels_fname: str | None = None
    signif_log_fp = None
    lacking_log_fp = None
    signif_fn_name: str = DEFAULT_SIGNIFICANCE_FUNCTION_NAME
    signif_log_interval: int = DEFAULT_SIGNIFICANCE_LOG_INTERVAL
    signif_mode: SignifMode = DEFAULT_SIGNIFICANCE_MODE
    signif_proportion: float = DEFAULT_SIGNIFICANCE_PROPORTION
    max_depth: int = MAX_DEPTH
    max_num_judgments: float = math.inf
    rel_if_unjudged: float = 0.0
    qidid: Strid = field(default_factory=Strid)
    non_contrib_runs: list[str] = field(default_factory=list)
    stop_when_top_run_found: bool = False


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-p", dest="persist", type=float, default=DEFAULT_PERSIST)
    parser.add_argument("-C", dest="chart_dir", default=None)
    parser.add_argument("-s", dest="score_log_interval", type=int, default=None)
    parser.add_argument("-S", dest="score_log_file", default=None)
    parser.add_argument("-Q", dest="qrels_fname", default=None)
    parser.add_argument("-E", dest="min_avg_err", type=float, default=0.0)
    parser.add_argument("-j", dest="judgment_log_file", default=None)
    parser.add_argument("-Z", dest="signif_log_file", default=None)
    parser.add_argument(
        "-z", dest="signif_log_interval", type=int, default=DEFAULT_SIGNIFICANCE_LOG_INTERVAL
    )
    parser.add_argument(
        "-P", dest="signif_fn_name", choices=sorted(SIGNIF_FUNCTIONS), default=None
    )
    parser.add_argument(
        "-m",
        dest="signif_mode_name",
        choices=["base", "conservative", "pessimal", "projected"],
        default=None,
    )
    parser.add_argument("-J", dest="max_num_judgments", type=int, default=None)
    parser.add_argument("-N", dest="non_contrib_runs", action="append", default=[])
    parser.add_argument("-T", dest="stop_when_top_run_found", action="store_true")
    parser.add_argument(
        "-G", dest="signif_proportion", type=float, default=DEFAULT_SIGNIFICANCE_PROPORTION
    )
    parser.add_argument("-L", dest="lacking_log_file", default=None)
    parser.add_argument("-U", dest="rel_if_unjudged", type=float, default=0.0)


def build_common_options(args: argparse.Namespace, parser: argparse.ArgumentParser) -> CommonOptions:
    opts = CommonOptions()
    opts.persist = args.persist
    if not (0.0 < opts.persist < 1.0):
        parser.error("Persist (-p) must be in range (0.0, 1.0)")
    opts.chart_dir = args.chart_dir
    opts.qrels_fname = args.qrels_fname
    opts.min_avg_err = args.min_avg_err
    opts.signif_log_interval = args.signif_log_interval
    opts.signif_proportion = args.signif_proportion
    if not (0.0 <= opts.signif_proportion <= 1.0):
        parser.error("Significance proportion (-G) must be between 0.0 and 1.0")
    if args.signif_fn_name is not None:
        opts.signif_fn_name = args.signif_fn_name
    if args.signif_mode_name is not None:
        opts.signif_mode = SignifMode(args.signif_mode_name)
    opts.max_num_judgments = args.max_num_judgments if args.max_num_judgments is not None else math.inf
    opts.non_contrib_runs = list(args.non_contrib_runs)
    opts.stop_when_top_run_found = args.stop_when_top_run_found
    opts.rel_if_unjudged = args.rel_if_unjudged

    opts.score_log_interval = (
        args.score_log_interval if args.score_log_interval is not None else DEFAULT_SCORE_LOG_INTERVAL
    )
    if args.score_log_file is not None:
        try:
            opts.score_log_fp = open(args.score_log_file, "w")
        except OSError as e:
            parser.error(
                f"Unable to open score log file '{args.score_log_file}' for writing: {e.strerror}"
            )
    elif args.score_log_interval is not None:
        opts.score_log_fp = sys.stdout

    if args.judgment_log_file is not None:
        try:
            opts.judgment_log_fp = open(args.judgment_log_file, "w")
        except OSError as e:
            parser.error(
                f"Unable to open judgment log file '{args.judgment_log_file}' for writing: {e.strerror}"
            )

    if args.signif_log_file is not None:
        try:
            opts.signif_log_fp = open(args.signif_log_file, "w")
        except OSError as e:
            parser.error(
                f"Unable to open significance log file '{args.signif_log_file}' for writing: {e.strerror}"
            )

    if args.lacking_log_file is not None:
        try:
            opts.lacking_log_fp = open(args.lacking_log_file, "w")
        except OSError as e:
            parser.error(
                f"Unable to open lacking log file '{args.lacking_log_file}' for writing: {e.strerror}"
            )

    return opts


def init_runerr(opts: CommonOptions, run_fnames: list[str]) -> RunErr:
    qrels = None
    if opts.qrels_fname is not None:
        try:
            with open(opts.qrels_fname) as fp:
                qrels = load_qrels(fp)
        except OSError as e:
            raise RunErrInitError(
                f"Unable to open qrels file '{opts.qrels_fname}' for reading: {e.strerror}"
            ) from None
        except QrelsFormatError as e:
            raise RunErrInitError(
                f"Error loading qrels file '{opts.qrels_fname}': {e}"
            ) from None

    # NOTE: the original calls qrels_get_qids(qrels, ...) here
    # unconditionally, even when qrels is still NULL (-Q not given) -- a
    # guaranteed NULL-pointer dereference (qrels->tbl on a NULL qrels).
    # Verified: minavgerr/minmaxerr/pooljudge all segfault immediately,
    # with zero output, if run without -Q. This is enforced as a clean
    # error here instead of reproducing the crash.
    if qrels is None:
        raise RunErrInitError(
            "no qrels file given (-Q) -- the original C tool segfaults "
            "here (qrels_get_qids() on a NULL qrels); -Q is a required "
            "option in practice, so this is a clean error instead of a "
            "crash"
        )

    for qid in qrels.get_qids():
        opts.qidid.get_id(qid)

    runerr = RunErr(opts.persist, opts.qidid, opts.max_depth)
    runerr.set_max_num_judgments(opts.max_num_judgments)
    runerr.set_qrels(qrels)
    runerr.set_rel_if_unjudged(opts.rel_if_unjudged)

    if opts.score_log_fp is not None:
        runerr.log_scores(opts.score_log_fp, opts.score_log_interval)
    if opts.judgment_log_fp is not None:
        runerr.log_judgments(opts.judgment_log_fp)
    if opts.lacking_log_fp is not None:
        runerr.log_lacking_documents(opts.lacking_log_fp)
    if opts.chart_dir is not None:
        if not runerr.chart(opts.chart_dir, DEFAULT_CHART_DEPTH, DEFAULT_CHART_JUDGMENTS):
            raise RunErrInitError(f"Unable to chart to directory '{opts.chart_dir}'")
    if opts.signif_log_fp is not None:
        runerr.log_significance(
            opts.signif_log_fp,
            SIGNIF_FUNCTIONS[opts.signif_fn_name],
            opts.signif_mode,
            DEFAULT_SIGNIFICANCE_P_THRESHOLD,
            opts.signif_log_interval,
            opts.signif_proportion,
        )

    err = runerr.load_runs_from_fnames(run_fnames)
    if err is not None:
        raise RunErrInitError(f"Error loading runs: {err}")

    for runid in opts.non_contrib_runs:
        if not runerr.set_run_noncontributing(runid):
            raise RunErrInitError(f"Unknown run '{runid}' set non-contributing")

    if opts.stop_when_top_run_found:
        runerr.enable_stop_when_top_run_found()

    return runerr


def report_runerr_stats(
    fp, runerr: RunErr, prev_num_judged: int, prev_num_relevant: float
) -> tuple[float, int, float]:
    """Returns (avg_err, new_num_judged, new_num_relevant) -- the caller
    threads the latter two back in as prev_* on the next call, mirroring
    the original's in/out pointer parameters."""
    (
        savg,
        _sdev,
        _smax,
        max_run_d,
        new_num_judged,
        new_num_relevant,
        num_lacking_judgments,
    ) = runerr.stats()
    assert new_num_relevant >= prev_num_relevant
    assert new_num_judged >= prev_num_judged
    precision = _safe_div(new_num_relevant - prev_num_relevant, new_num_judged - prev_num_judged)
    fp.write(
        f"{new_num_judged} {savg:.4f} {_sdev:.4f} {_smax:.4f} {precision:.4f} "
        f"{max_run_d} {num_lacking_judgments}\n"
    )
    return savg, new_num_judged, new_num_relevant
