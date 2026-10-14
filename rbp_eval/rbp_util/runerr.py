"""Port of rbp_util/runerr.[ch]: incremental residual-error tracking for a
pool of runs as documents are judged one at a time.

This is the engine behind minavgerr/minmaxerr/pooljudge: it maintains,
per run, a running RBP estimate and a shrinking residual-error bound as
more (query, document) judgments are revealed, plus optional logging
(scores/judgments/lacking-documents/significance) and HTML charting.

Two deliberate deviations from a literal port, both documented in detail
at their call sites below, not silently:

1. `RunErrInitError` is raised where the original would follow a NULL
   qrels pointer through qrels_get_qids() -- confirmed to segfault the
   compiled binaries of minavgerr/minmaxerr/pooljudge immediately (no
   output at all) if run without -Q. Reproducing a NULL-pointer
   dereference isn't meaningfully "porting" it in a memory-safe language,
   so this raises a clear error instead, consistent with how the
   `reltrans` port handled its own NULL-dereference bug.

2. `wgt_err`'s WGT_RESIDUAL_AND_PROJECTED[_SQ] cases use the *unguarded*
   PROJECTED_RBP macro (unlike WGT_PROJECTED, which guards err==1.0).
   Every run starts at (rbp=0, err=1.0) before its first judgment, so
   selecting this weighting produces 0/0 = NaN on the very first call to
   runerr_max_prop_wgt() -- and since NaN comparisons are always false,
   `max_run_d` in the C version is left as uninitialized stack memory.
   This port reproduces the NaN (same IEEE 754 propagation) but can't
   meaningfully reproduce "uninitialized memory", so `max_run_d` just
   keeps a well-defined (if equally nonsensical) default of 0 -- see
   `max_prop_wgt`.

The paired significance test functions here take a plain (dist_x, dist_y)
two-sequence signature, dropping the original C typedef's redundant
`dist_size` (== len(dist_x)) and always-None `data` parameter.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, TextIO

from ..qrels import REL_UNJUDGED, Qrels
from ..ranking import Ordering
from ..rbp import rbp_weights
from ..runs import Run, RunFormatError, load_run
from .dococcur import DocOccur
from .strid import Strid

SigFn = Callable[[list[float], list[float]], float]


class Weight(Enum):
    UNIFORM = "uniform"
    LINEAR = "linear"
    QUADRATIC = "quadratic"
    PROJECTED = "projected"
    RESIDUAL = "residual"
    RESIDUAL_AND_MIDPOINT = "residual-and-midpoint"
    RESIDUAL_AND_MIDPOINT_SQ = "residual-and-midpoint-squared"
    RESIDUAL_AND_MIDPOINT_CB = "residual-and-midpoint-cubed"
    RESIDUAL_AND_PROJECTED = "residual-and-projected"
    RESIDUAL_AND_PROJECTED_SQ = "residual-and-projected-squared"
    MIDPOINT_SQ = "midpoint-squared"


class QueryWeight(Enum):
    UNIFORM = "uniform"
    LINEAR = "linear"


class SignifMode(Enum):
    BASE = "base"
    CONSERVATIVE = "conservative"
    PESSIMAL = "pessimal"
    PROJECTED = "projected"


WGT_ERR_PROJ_MIN = 0.01
WGT_QUERY_MIN = 0.01


def _safe_div(numerator: float, denominator: float) -> float:
    """Python raises ZeroDivisionError for float division by zero; C
    (IEEE 754) silently produces +-inf or nan. Several original formulas
    rely on that silent behaviour (see wgt_err, wgt_for_qry)."""
    if denominator != 0.0:
        return numerator / denominator
    if numerator == 0.0:
        return math.nan
    return math.copysign(math.inf, numerator)


def _projected_rbp(base: float, residual: float) -> float:
    """Guarded projection (matches the C `projected_rbp()` wrapper)."""
    if residual == 1.0:
        return 1.0
    return base + _safe_div(base * residual, 1.0 - residual)


def _projected_rbp_unsafe(base: float, residual: float) -> float:
    """Unguarded projection (matches the raw PROJECTED_RBP macro used
    directly by wgt_err's RESIDUAL_AND_PROJECTED[_SQ] cases) -- see the
    module docstring for the resulting NaN-at-(0, 1.0) bug this
    reproduces."""
    return base + _safe_div(base * residual, 1.0 - residual)


def wgt_err(rbp: float, err: float, weighting: Weight) -> float:
    if weighting is Weight.UNIFORM:
        return err
    if weighting is Weight.LINEAR:
        return 0.5 * ((rbp + err) ** 2 - rbp**2)
    if weighting is Weight.QUADRATIC:
        return (1.0 / 3.0) * ((rbp + err) ** 3 - rbp**3)
    if weighting is Weight.PROJECTED:
        if err == 1.0:
            return err
        wgt = (rbp + _safe_div(rbp * err, 1.0 - err)) * err
        if _safe_div(wgt, err) < WGT_ERR_PROJ_MIN:
            return WGT_ERR_PROJ_MIN
        return wgt
    if weighting is Weight.RESIDUAL:
        return err * err
    if weighting is Weight.RESIDUAL_AND_MIDPOINT:
        return (rbp + err / 2) * err * err
    if weighting is Weight.RESIDUAL_AND_MIDPOINT_SQ:
        return (rbp + err / 2) ** 2 * err * err
    if weighting is Weight.RESIDUAL_AND_MIDPOINT_CB:
        return (rbp + err / 2) ** 3 * err * err
    if weighting is Weight.RESIDUAL_AND_PROJECTED:
        return _projected_rbp_unsafe(rbp, err) * err * err
    if weighting is Weight.RESIDUAL_AND_PROJECTED_SQ:
        return _projected_rbp_unsafe(rbp, err) ** 2 * err * err
    if weighting is Weight.MIDPOINT_SQ:
        return (rbp + err / 2) ** 2 * err
    raise ValueError(weighting)


@dataclass
class RunInfo:
    run: Run
    rund: int
    rbp: float = 0.0
    err: float = 1.0
    rbps: list[float] = field(default_factory=list)
    errs: list[float] = field(default_factory=list)
    judged_depth: int = 0  # used by minmaxerr
    contributes_judgments: bool = True


@dataclass
class QryInfo:
    qid: str
    qidd: int
    judged: int = 0
    rel: float = 0.0


class RunErrInitError(Exception):
    pass


class RunErr:
    def __init__(self, persist: float, qidid: Strid, depth: int):
        self.persist = persist
        self.qidid = qidid
        self.depth = depth
        self.max_num_judgments: float = math.inf
        self.lacking_judgments = 0
        self.rbp_wgts = rbp_weights(persist, depth)
        self.num_relevant = 0.0
        self.num_judged = 0
        self.rel_if_unjudged = 0.0

        self.chart_dir: str | None = None
        self.chart_depth = 0
        self.chart_judgments = 0

        self.qrels: Qrels | None = None

        self.score_log_fp: TextIO | None = None
        self.score_log_interval = 0

        self.judgment_log_fp: TextIO | None = None
        self.lacking_judgments_log_fp: TextIO | None = None

        self.signif_log_fp: TextIO | None = None
        self.signif_fn: SigFn | None = None
        self.signif_mode: SignifMode | None = None
        self.signif_p_threshold = 0.0
        self.signif_log_interval = 0
        self.signif_proportion = 0.0

        self.stop_when_top_run_found = False

        self.dcr = DocOccur()
        self.runs: list[RunInfo] = []
        num_qids = qidid.num_ids()
        assert num_qids > 0
        self.queries: list[QryInfo] = [
            QryInfo(qid=qidid.get_str(q), qidd=q) for q in range(num_qids)
        ]
        self.docs_judged: dict[tuple[int, str], float] = {}

    # -- setup --

    def set_qrels(self, qrels: Qrels) -> None:
        self.qrels = qrels

    def set_rel_if_unjudged(self, rel: float) -> None:
        self.rel_if_unjudged = rel

    def set_max_num_judgments(self, n: float) -> None:
        self.max_num_judgments = n

    def log_scores(self, fp: TextIO, interval: int) -> None:
        self.score_log_fp = fp
        self.score_log_interval = interval

    def log_judgments(self, fp: TextIO) -> None:
        self.judgment_log_fp = fp

    def log_lacking_documents(self, fp: TextIO) -> None:
        self.lacking_judgments_log_fp = fp

    def enable_stop_when_top_run_found(self) -> None:
        self.stop_when_top_run_found = True

    def log_significance(
        self,
        fp: TextIO,
        signif_fn: SigFn,
        signif_mode: SignifMode,
        p_threshold: float,
        interval: int,
        proportion: float,
    ) -> None:
        self.signif_log_fp = fp
        self.signif_fn = signif_fn
        self.signif_mode = signif_mode
        self.signif_p_threshold = p_threshold
        self.signif_log_interval = interval
        self.signif_proportion = proportion

    def chart(self, chart_dir: str, chart_depth: int, chart_judgments: int) -> bool:
        if not os.path.isdir(chart_dir):
            return False
        self.chart_dir = chart_dir
        self.chart_depth = chart_depth
        self.chart_judgments = chart_judgments
        return True

    def load_runs_from_fnames(self, fnames: list[str]) -> str | None:
        """Returns an error message on failure, else None."""
        for fname in fnames:
            try:
                with open(fname) as fp:
                    run = load_run(fp)
            except OSError as e:
                return f"Unable to open run file '{fname}' for reading: {e.strerror}"
            except RunFormatError as e:
                return f"Error loading run file '{fname}': {e}"
            self._add_run(run)
        return None

    def _add_run(self, run: Run) -> None:
        num_qids = len(self.queries)
        ri = RunInfo(
            run=run,
            rund=len(self.runs),
            rbps=[0.0] * num_qids,
            errs=[1.0] * num_qids,
        )
        self.dcr.add_run(run, ri.rund, self.qidid)
        self.runs.append(ri)

    def set_run_noncontributing(self, runid: str) -> bool:
        for ri in self.runs:
            if ri.run.get_runid() == runid:
                ri.contributes_judgments = False
                return True
        return False

    # -- introspection --

    def get_dococcur_items(self, docid: str, qidd: int):
        return self.dcr.get(docid, qidd)

    def get_runs(self) -> list[RunInfo]:
        return self.runs

    # -- the core incremental step --

    def doc_judged(self, docid: str, qidd: int, num_qids: int) -> bool:
        """Reveal the judgment for one (docid, qidd) pair, updating every
        run's running rbp/err estimate. Returns True if the caller should
        keep going, False if a stopping condition has been reached
        (mirrors runerr_doc_judged's `!finished` return)."""
        dococ = self.dcr.get(docid, qidd)
        assert dococ is not None
        self.queries[qidd].judged += 1

        rel = REL_UNJUDGED
        is_judged = False
        qid = self.qidid.get_str(qidd)

        if self.qrels is not None:
            rel = self.qrels.get_rel(qid, docid)
            if rel < 0.0:
                self.lacking_judgments += 1
                if self.lacking_judgments_log_fp is not None:
                    fp = self.lacking_judgments_log_fp
                    fp.write(f"{docid} {qidd}:")
                    for item in dococ:
                        ri = self.runs[item.rund]
                        fp.write(f" ({ri.run.get_runid()}, {item.rank})")
                    fp.write("\n")
                else:
                    sys.stderr.write(f"Unjudged: {docid} {qidd}: {self.num_judged}\n")
                rel = self.rel_if_unjudged
            else:
                is_judged = True
            self.queries[qidd].rel += rel
            if rel > 0.0:
                self.num_relevant += rel

        if self.judgment_log_fp is not None:
            rep_rel = rel if is_judged else -1.0
            self.judgment_log_fp.write(f"{qid} {docid} {rep_rel:.2f}\n")

        key = (qidd, docid)
        assert key not in self.docs_judged
        self.docs_judged[key] = rel

        for item in dococ:
            ds = item.data
            ds.flags = 1
            assert item.rank < self.depth
            ri = self.runs[item.rund]
            wgt = self.rbp_wgts[item.rank]
            ri.err -= wgt / num_qids
            ri.errs[qidd] -= wgt
            if rel > 0.0:
                ri.rbp += (wgt * rel) / num_qids
                ri.rbps[qidd] += rel * wgt

        if self.chart_dir is not None and self.num_judged < self.chart_judgments:
            self._chart(docid, qidd, num_qids)

        self.num_judged += 1

        if (
            self.score_log_fp is not None
            and self.num_judged % self.score_log_interval == 0
        ):
            self._log_scores()

        if (
            self.signif_log_fp is not None
            and self.num_judged % self.signif_log_interval == 0
        ):
            signif_pc = self.significance(
                self.signif_p_threshold,
                self.signif_proportion,
                self.signif_fn,
                self.signif_mode,
            )
            self.signif_log_fp.write(f"{self.num_judged} {signif_pc:.4f}\n")

        finished = False
        if self.stop_when_top_run_found:
            max_base = 0.0
            max_base_pos = 0
            for r, ri in enumerate(self.runs):
                if ri.rbp > max_base:
                    max_base = ri.rbp
                    max_base_pos = r
            distinct_top = True
            for r, ri in enumerate(self.runs):
                if not distinct_top:
                    break
                if r == max_base_pos:
                    continue
                if ri.rbp + ri.err >= max_base:
                    distinct_top = False
            if distinct_top:
                finished = True
                if self.score_log_fp is not None:
                    self._log_scores()
                else:
                    print(f"# Top run found after {self.num_judged} judgments")

        if self.num_judged >= self.max_num_judgments:
            finished = True

        return not finished

    # -- weighted stats --

    def stats(self):
        return self.stats_wgt(Weight.UNIFORM)

    def stats_wgt(self, weighting: Weight):
        """Returns (avg_err, stddev, max_err, max_run_d, num_judged,
        num_relevant, num_lacking_judgments)."""
        max_err = 0.0
        max_run_d = 0
        tot_err = 0.0
        for r, ri in enumerate(self.runs):
            err = wgt_err(ri.rbp, ri.err, weighting)
            tot_err += err
            if err > max_err:
                max_err = err
                max_run_d = r
        avg_err = tot_err / len(self.runs)
        var = 0.0
        for ri in self.runs:
            err = wgt_err(ri.rbp, ri.err, weighting)
            v = avg_err - err
            var += v * v
        var /= len(self.runs)
        stddev = math.sqrt(var)
        return (
            avg_err,
            stddev,
            max_err,
            max_run_d,
            self.num_judged,
            self.num_relevant,
            self.lacking_judgments,
        )

    def prop_err_wgt_for_run(self, rund: int, weighting: Weight) -> float:
        ri = self.runs[rund]
        return _safe_div(wgt_err(ri.rbp, ri.err, weighting), ri.err)

    def wgt_for_qry(self, qidd: int, weighting: QueryWeight) -> float:
        qi = self.queries[qidd]
        if weighting is QueryWeight.UNIFORM:
            return 1.0
        if weighting is QueryWeight.LINEAR:
            return WGT_QUERY_MIN + _safe_div(qi.rel, qi.judged)
        raise ValueError(weighting)

    def max_prop_wgt(self, weighting: Weight) -> tuple[float, int]:
        """Returns (max_proportional_weight, run_index). See the module
        docstring for the NaN/uninitialized-index bug this reproduces for
        WGT_RESIDUAL_AND_PROJECTED[_SQ] before any judgments are made."""
        max_val = 0.0
        max_run_d = 0
        for r, ri in enumerate(self.runs):
            err = _safe_div(wgt_err(ri.rbp, ri.err, weighting), ri.err)
            if err > max_val:
                max_val = err
                max_run_d = r
        return max_val, max_run_d

    # -- significance --

    def significance(
        self,
        p_threshold: float,
        top_proportion: float,
        sig_fn: SigFn,
        signif_mode: SignifMode,
    ) -> float:
        assert 0.0 < top_proportion <= 1.0
        num_runs = len(self.runs)
        num_qids = len(self.queries)

        num_top_runs = int(num_runs * top_proportion + 0.5)
        if num_top_runs > num_runs:
            num_top_runs = num_runs

        order = sorted(range(num_runs), key=lambda r: -self.runs[r].rbp)

        runs_b: list[RunInfo] | None = None
        if signif_mode is SignifMode.PESSIMAL:
            runs_b = [self.runs[order[r]] for r in range(num_top_runs)]

        pairs = 0
        asl = 0
        for i in range(num_top_runs):
            if signif_mode is SignifMode.PESSIMAL:
                num_runs_b = num_top_runs - i - 1
                pairs += num_runs_b
                asl += self._pessimal_significance_pairs(
                    p_threshold,
                    self.runs[order[i]],
                    runs_b[i + 1 :],
                    num_qids,
                    sig_fn,
                )
            else:
                for j in range(i + 1, num_top_runs):
                    ra = self.runs[order[i]]
                    rb = self.runs[order[j]]
                    assert ra.rbp >= rb.rbp
                    pairs += 1
                    if signif_mode is SignifMode.CONSERVATIVE:
                        dist_b = [rb.rbps[s] + rb.errs[s] for s in range(num_qids)]
                    elif signif_mode is SignifMode.PROJECTED:
                        dist_b = [
                            _projected_rbp(rb.rbps[s], rb.errs[s])
                            for s in range(num_qids)
                        ]
                    else:
                        dist_b = rb.rbps
                    p = sig_fn(ra.rbps, dist_b)
                    if p < p_threshold:
                        asl += 1
        return asl / pairs

    def _pessimal_significance_pairs(
        self,
        p_threshold: float,
        run_a: RunInfo,
        runs_b: list[RunInfo],
        num_qids: int,
        sig_fn: SigFn,
    ) -> int:
        num_runs_b = len(runs_b)
        rund_to_b_ind = {rb.rund: b_ind for b_ind, rb in enumerate(runs_b)}

        rbp_a = [[-1.0] * num_qids for _ in range(num_runs_b)]
        rbp_b = [[-1.0] * num_qids for _ in range(num_runs_b)]
        err_b = [[-1.0] * num_qids for _ in range(num_runs_b)]
        for b_ind, rb in enumerate(runs_b):
            for q in range(num_qids):
                assert run_a.rbps[q] >= -0.001
                assert rb.rbps[q] >= -0.001
                assert rb.errs[q] >= -0.001
                rbp_a[b_ind][q] = run_a.rbps[q]
                rbp_b[b_ind][q] = rb.rbps[q]
                err_b[b_ind][q] = rb.errs[q]

        for q in range(num_qids):
            qid = self.qidid.get_str(q)
            qdocs = run_a.run.get_qdocs_by_qid(qid)
            assert qdocs is not None
            num_scores = min(qdocs.num_scores(), self.depth)
            scores = qdocs.get_scores(Ordering.SCORE)  # QDOCS_DEFAULT_ORDERING
            judged_rbp = 0.0

            for d in range(num_scores):
                docid = scores[d].docid
                key = (q, docid)
                if key not in self.docs_judged:
                    a_wgt = self.rbp_wgts[d]
                    dococ = self.dcr.get(docid, q)
                    assert dococ is not None
                    seen_here = False
                    for item in dococ:
                        if item.rund == run_a.rund:
                            assert not seen_here
                            seen_here = True
                        b_ind = rund_to_b_ind.get(item.rund)
                        if b_ind is not None:
                            b_wgt = self.rbp_wgts[item.rank]
                            err_b[b_ind][q] -= b_wgt
                            assert err_b[b_ind][q] >= -0.001
                            if a_wgt < b_wgt:
                                assert d > item.rank
                                rbp_b[b_ind][q] += b_wgt
                                rbp_a[b_ind][q] += a_wgt
                                assert rbp_b[b_ind][q] <= 1.001
                                assert rbp_a[b_ind][q] <= 1.001
                    assert seen_here
                else:
                    rel = self.docs_judged[key]
                    if rel > 0.0:
                        judged_rbp += self.rbp_wgts[d] * rel

            assert abs(judged_rbp - run_a.rbps[q]) < 0.001
            for b_ind in range(num_runs_b):
                rbp_b[b_ind][q] += err_b[b_ind][q]
                assert rbp_b[b_ind][q] <= 1.001

        for b_ind, rb in enumerate(runs_b):
            for q in range(num_qids):
                assert rbp_a[b_ind][q] >= run_a.rbps[q] - 0.001
                assert rbp_a[b_ind][q] <= 1.001
                assert rbp_b[b_ind][q] >= rb.rbps[q] - 0.001
                assert rbp_b[b_ind][q] <= 1.001
                assert err_b[b_ind][q] >= -0.001
                assert err_b[b_ind][q] <= rb.errs[q] + 0.001
                assert rbp_a[b_ind][q] <= run_a.rbps[q] + run_a.errs[q] + 0.001
                assert rbp_b[b_ind][q] <= rb.rbps[q] + rb.errs[q] + 0.001
                assert rbp_a[b_ind][q] - run_a.rbps[q] < rbp_b[b_ind][q] - rb.rbps[q] + 0.001

        asl = 0
        for b_ind in range(num_runs_b):
            p = sig_fn(rbp_a[b_ind], rbp_b[b_ind])
            if p < p_threshold:
                asl += 1
        return asl

    def num_judged_count(self) -> int:
        return self.num_judged

    # -- logging / charting --

    def _log_scores(self) -> None:
        if self.score_log_fp is not None:
            fp = self.score_log_fp
            fp.write(f"# {self.num_judged} judged\n")
            for ri in self.runs:
                fp.write(f"{ri.run.get_runid()} {ri.rbp:.4f} +{ri.err:.4f}\n")
            fp.write("\n")

    def _chart(self, docid: str, qidd: int, num_qids: int) -> None:
        assert self.chart_dir is not None
        chart_fname = os.path.join(self.chart_dir, f"{self.num_judged}.html")
        try:
            fp = open(chart_fname, "w")
        except OSError:
            sys.stderr.write(
                f"ERROR: unable to open chart file '{chart_fname}'; "
                f"turning charting off\n"
            )
            self.chart_dir = None
            return
        with fp:
            fp.write(f"<html><head><title>Judgement Chart: {self.num_judged}</title>")
            fp.write("<link rel='stylesheet' href='judgments.css' type='text/css'>\n")
            fp.write("</head>\n")
            fp.write(f"<body><h1>Judgement Chart: {self.num_judged}</h1>\n")
            # `num_judged - 1` is an unsigned int in C: at num_judged==0
            # (the very first chart page) it wraps around to UINT_MAX
            # rather than going negative. The "prev" link 404s either way
            # (neither page exists) but this matches the exact filename.
            prev_page = (self.num_judged - 1) % (2**32)
            fp.write(
                f"<div class='nav'><a href='{prev_page}.html'>prev</a> | "
                f"<a href='{self.num_judged + 1}.html'>next</a></div>\n"
            )
            fp.write("<div class='nav'>\n")
            i = 0
            while i * 20 < self.chart_judgments:
                fp.write(f"<a href='{i * 20}.html'>{i * 20}</a> | \n")
                i += 1
            fp.write("</div>\n")
            fp.write(f"<div class='info'>Judged: docid {docid}, query {qidd}</div>\n")
            for q in range(num_qids):
                fp.write("<table>\n")
                for ri in self.runs:
                    qd = ri.run.get_qdocs_by_index(q)
                    fp.write("<tr>\n")
                    fp.write(f"<td class='err'>{ri.err:.4f}</td>\n")
                    for d in range(self.chart_depth):
                        numranks = qd.num_scores()
                        if d < numranks:
                            ds2 = qd.get_scores(Ordering.SCORE)
                            if q == qidd and docid == ds2[d].docid:
                                fp.write("<td class='justjudged'>")
                            elif ds2[d].flags == 1:
                                fp.write("<td class='judged'>")
                            else:
                                fp.write("<td class='unjudged'>")
                        else:
                            fp.write("<td class='unranked'>")
                        fp.write("</td>")
                    fp.write("</tr>\n")
                fp.write("</table>\n")
            fp.write("</html>\n")
