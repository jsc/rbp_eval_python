# rbp-eval (Python port)

A Python port of the original C/C++ `rbp_eval` toolkit: Rank-Biased
Precision evaluation for IR experiments, plus DCG, significance tests,
and the adaptive/minimal-test-collection pooling tools from `rbp_util`.

The Rank-Biased Precision metric itself is due to Moffat, A. and Zobel,
J., 2008, "Rank-biased precision for measurement of retrieval
effectiveness," ACM TOIS 27(1). The original C/C++ implementation being
ported here -- `rbp_eval`, `dcg_eval`, `reltrans`, and the `rbp_util`
pooling tools -- was written by **William Webber**. This package is a
Python port of that implementation, not an original design.

This is a from-scratch reimplementation, not a wrapper around the C code.
It's been validated directly against the compiled C/C++ binaries (see
`tests/`) rather than just unit-tested in isolation, and along the way it
found and fixed six real bugs in the original (crashes, silent numeric
overflow, and undefined behaviour) -- these are documented in the
docstrings of the modules where they were found, not just here.

## Install

Requires Python 3.10+.

```sh
conda create -n "RBP" python=3.10
conda activate RBP
pip install -e .
```

This installs the `rbp_eval` package and six console scripts: `rbp_eval`,
`dcg_eval`, `reltrans`, `minavgerr`, `minmaxerr`, `pooljudge`.

Dependencies are `numpy` and `scipy` (used for the binomial/sign/t-test
significance tests and vectorized bootstrap resampling).

## CLI tools

### `rbp_eval` -- the core RBP evaluator

```sh
rbp_eval [options] <qrels-file> <run-file>
```

Key options: `-d DEPTH_SPEC` (comma-separated cutoff depths, `0` = full
ranking), `-p PERSIST_SPEC` (comma-separated persistence values), `-q`
(per-query output), `-T` (suppress averages), `-r`/`-s`/`-o` (rank by
rank column / score column / run-file order), `-b`/`-B`/`-f`/`-F`/`-a`
(relevance handling: binary at a threshold, binary at 1, fractional with
a scale, fractional unscaled, or auto-detect), `-H` (no header), `-W`
(suppress warnings). Run `rbp_eval -h` for the full list.

```sh
rbp_eval -q -d 5,10,20 -p 0.5,0.8,0.95 qrels.txt run.txt
```

### `dcg_eval` -- (normalised) Discounted Cumulative Gain

```sh
dcg_eval [-R] [-b base] [-r depth] [-n|N] [-d|D] [-m|M] <qrels> <run>
```

`-R` shifts to the "MS"/non-stupid DCG variant (discounts rank 1 too),
`-b` sets the log base (default 2), `-r` sets the depth cutoff (default
1000), `-n`/`-N` toggle normalisation (nDCG vs. raw DCG, default on),
`-d`/`-D` toggle discounting, `-m`/`-M` select fractional vs. binary
relevance handling.

### `reltrans` -- translate run files into per-query relevance listings

```sh
reltrans [-d <output-dir>] <qrels> <run> ...
```

For each run, writes `<output-dir>/<runid>/<qid>` files listing one
relevance value per ranked document (`NA` for unjudged).

### `minavgerr`, `minmaxerr`, `pooljudge` -- adaptive pooling / minimal test collections

These simulate incrementally revealing judgments from a reference qrels
file to a pool of competing runs, tracking each run's estimated RBP and
shrinking error bound as more documents are judged -- used for
minimal-judging-effort test collection research.

```sh
pooljudge  -Q <qrels> [options] <run> ...   # standard depth-k pooling
minavgerr  -Q <qrels> [options] <run> ...   # judge to minimise average error
minmaxerr  -Q <qrels> [options] <run> ...   # judge to minimise the max error
```

`-Q <qrels>` is required in practice (see "Known deviations" below).
Common options shared by all three: `-p PERSIST` (default 0.95), `-E
MIN_AVG_ERR` (stop early once average error drops below this), `-C
CHART_DIR` (dump an HTML judgment chart per document judged), `-s`/`-S`
(periodic / continuous score logging), `-j` (judgment log), `-L` (log of
documents lacking a reference judgment), `-Z`/`-z` (significance
logging + interval), `-P {wilcoxon,sign,t,bootstrap}` (which paired
significance test to use), `-m {base,conservative,pessimal,projected}`
(significance comparison mode), `-G PROPORTION` (fraction of top runs to
compare), `-N RUNID` (mark a run as non-contributing to judgments,
repeatable), `-T` (stop once the top run is unambiguously decided), `-U
REL` (fallback relevance for documents the reference qrels doesn't
cover), `-J MAX_JUDGMENTS` (judgment budget).

`minavgerr` additionally supports `-W {uniform,linear,quadratic,
projected,residual,residual-and-midpoint[-squared|-cubed],
residual-and-projected[-squared],midpoint-squared}` (run error-weighting
scheme) and `-q {uniform,linear}` (query weighting), plus `-D
depths-file` (report at specific judgment-count checkpoints instead of
every 1000). `minmaxerr` supports the same `-W`/`-D` (restricted to
uniform/linear/quadratic weighting).

## Library usage

Every CLI is a thin wrapper over a plain library API:

```python
from rbp_eval import load_qrels, load_run, evaluate, Ordering

with open("qrels.txt") as f:
    qrels = load_qrels(f)
with open("run.txt") as f:
    run = load_run(f)

res = evaluate(qrels, run, Ordering.SCORE, persist=[0.8], depths=[10, 20])
print(res.ave_res.depth_res[0].persist_res[0].sum)   # average RBP@10, p=0.8
```

Significance tests and rank correlation live in `rbp_eval.stats`:

```python
from rbp_eval.stats import paired_wilcoxon_test_p, kendall_tau, tau_ap
```

The pooling framework lives in `rbp_eval.rbp_util` (`RunErr`, `DocWgt`,
`DocOccur`, `common.init_runerr`) if you want to script custom pooling
strategies rather than using the CLIs directly.

## Tests

```sh
pip install -e ".[dev]"   # or just: pip install pytest
python -m pytest tests/
```

Most tests are differential: they run the compiled C/C++ binaries
(built from the repo root's autotools build) alongside this port on the
same inputs and assert matching output, rather than only checking this
port's output against hand-written expectations. Those tests
auto-skip if the corresponding C binary hasn't been built.

## Known deviations from the original

Documented in detail in the relevant module docstrings; summarized here:

- **Bugs fixed, not reproduced** (all confirmed against the compiled C
  binaries): `binomial.c`'s overflow-avoidance recurrence actually
  overflows to `inf` around ~1100 trials, not "~10,000" as its own
  comment claims (`rbp_eval/stats/binomial.py`); `reltrans` and
  `minavgerr`/`minmaxerr`/`pooljudge` (without `-Q`) segfault on a
  NULL-pointer dereference (`rbp_eval/cli/reltrans.py`,
  `rbp_eval/rbp_util/common.py`); a depth-cutoff restart bug in the core
  RBP tie-handling could crash or emit NaN when a query returns fewer
  documents than a later requested depth (`rbp_eval/rbp.py`); an
  operator-precedence bug made `rbp_eval`'s `-T` flag's duplicate-check
  a no-op (sidestepped by just using a plain idempotent flag,
  `rbp_eval/cli/rbp_eval.py`); `minavgerr -W residual-and-projected`
  divides by zero on its very first run-selection call
  (`rbp_eval/rbp_util/runerr.py`).
- **Upgraded, not ported**: the paired t-test's p-value was a coarse
  8-bucket lookup table in the original; this uses
  `scipy.stats.ttest_rel` for a real continuous p-value instead
  (`rbp_eval/stats/t.py`).
- **Written fresh, not ported**: `tau_ap` (AP correlation) -- the
  original C++ never actually assembled the coefficient, only a
  leftover counting helper (`rbp_eval/stats/tau_ap.py`).
- **Inherently unreproducible**: tie-break order among exactly-equal-
  weight documents in `minavgerr`'s document ordering, since it depends
  on the original's use of an unstable C `qsort` with no tie-break
  comparator. Verified this never affects the actual RBP/error numbers,
  only the order in which tied documents are judged
  (`rbp_eval/rbp_util/docwgt.py`).
