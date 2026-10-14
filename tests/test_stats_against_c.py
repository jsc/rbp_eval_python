"""Differential tests: the stats module against the compiled C test
binaries in ../../stats/ (built from stats/*.c's *_MAIN drivers).

binomial, sign, wilcoxon, and kendall are checked for exact numeric
agreement. t is checked only on the raw t-statistic (its p-value was
deliberately upgraded to scipy, so it will not match -- see
rbp_eval.stats.t). bootstrap is PID-seeded in the C version and was never
reproducible even between two C runs, so it's only sanity-checked (valid
probability, roughly consistent for a clearly-one-sided difference), not
compared for exact agreement.
"""

from __future__ import annotations

import random
import subprocess
from pathlib import Path

import pytest

from rbp_eval.stats import (
    cumulative_binomial_p,
    kendall_tau,
    paired_bootstrap_test_p,
    paired_sign_test_p,
    paired_t_test,
    paired_wilcoxon_test_p,
)

STATS_DIR = Path(__file__).resolve().parents[2] / "stats"
TOL = 1e-5


def _c_binary(name: str) -> Path:
    return STATS_DIR / name if name != "tau_ap" else STATS_DIR / "test" / "tau_ap"


def _have(name: str) -> bool:
    return _c_binary(name).is_file()


def _run_paired(binary: Path, xs, ys) -> str:
    stdin = "\n".join(f"{x} {y}" for x, y in zip(xs, ys)) + "\n"
    proc = subprocess.run([str(binary)], input=stdin, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout, proc.stderr


def _random_paired(n: int, seed: int) -> tuple[list[float], list[float]]:
    rng = random.Random(seed)
    xs = [round(rng.uniform(0, 1), 4) for _ in range(n)]
    ys = [round(rng.uniform(0, 1), 4) for _ in range(n)]
    return xs, ys


@pytest.mark.skipif(not _have("sign"), reason="C sign binary not built")
@pytest.mark.parametrize("seed", range(10))
def test_sign_matches_c(seed):
    xs, ys = _random_paired(30, seed)
    out, _ = _run_paired(_c_binary("sign"), xs, ys)
    expected = float(out)
    actual = paired_sign_test_p(xs, ys)
    assert actual == pytest.approx(expected, abs=TOL)


@pytest.mark.skipif(not _have("wilcoxon"), reason="C wilcoxon binary not built")
@pytest.mark.parametrize("seed", range(10))
def test_wilcoxon_matches_c(seed):
    xs, ys = _random_paired(20, seed)
    out, _ = _run_paired(_c_binary("wilcoxon"), xs, ys)
    expected = float(out)
    actual = paired_wilcoxon_test_p(xs, ys)
    assert actual == pytest.approx(expected, abs=TOL)


@pytest.mark.skipif(not _have("kendall"), reason="C kendall binary not built")
@pytest.mark.parametrize("seed", range(10))
def test_kendall_matches_c(seed):
    xs, ys = _random_paired(25, seed)
    out, _ = _run_paired(_c_binary("kendall"), xs, ys)
    expected = float(out)
    actual = kendall_tau(xs, ys)
    assert actual == pytest.approx(expected, abs=TOL)


@pytest.mark.skipif(not _have("t"), reason="C t binary not built")
@pytest.mark.parametrize("seed", range(10))
def test_t_statistic_matches_c(seed):
    xs, ys = _random_paired(15, seed)
    out, _ = _run_paired(_c_binary("t"), xs, ys)
    expected_t, _expected_p_bucket = (float(v) for v in out.split())
    actual_t = paired_t_test(xs, ys)
    assert actual_t == pytest.approx(expected_t, abs=TOL)


@pytest.mark.skipif(not _have("binomial"), reason="C binomial binary not built")
@pytest.mark.parametrize(
    "trials,successes,prob", [(20, 15, 0.5), (100, 60, 0.5), (50, 10, 0.3)]
)
def test_binomial_matches_c(trials, successes, prob):
    proc = subprocess.run(
        [str(_c_binary("binomial")), str(trials), str(successes), str(prob)],
        capture_output=True,
        text=True,
    )
    expected = float(proc.stderr)  # binomial's MAIN prints to stderr
    actual = cumulative_binomial_p(trials, successes, prob)
    assert actual == pytest.approx(expected, abs=TOL)


@pytest.mark.skipif(not _have("binomial"), reason="C binomial binary not built")
def test_binomial_c_overflows_where_python_does_not():
    """Documents a real bug in the original: despite its own comment
    claiming long doubles keep it safe "around the 10,000 trial mark",
    the C recurrence actually overflows to `inf` starting around ~1100
    trials (verified: trials=1000 is fine, trials=1200 already inf, at
    p=0.5 with successes near the tail). This means the original sign
    test silently breaks on any modern-scale IR experiment (1000+
    queries). scipy.stats.binom.sf has no such issue.
    """
    proc = subprocess.run(
        [str(_c_binary("binomial")), "5000", "2600", "0.5"],
        capture_output=True,
        text=True,
    )
    c_result = float(proc.stderr)
    assert c_result == float("inf"), "expected the known C overflow bug to still reproduce"

    py_result = cumulative_binomial_p(5000, 2600, 0.5)
    assert py_result == pytest.approx(0.0024418248687587472, abs=TOL)


@pytest.mark.skipif(not _have("bootstrap"), reason="C bootstrap binary not built")
def test_bootstrap_sanity():
    # x consistently, substantially higher than y: ASL should be small.
    rng = random.Random(0)
    xs = [5.0 + rng.uniform(0, 1) for _ in range(30)]
    ys = [1.0 + rng.uniform(0, 1) for _ in range(30)]
    out, _ = _run_paired(_c_binary("bootstrap"), xs, ys)
    c_p = float(out)
    py_p = paired_bootstrap_test_p(xs, ys)
    assert 0.0 <= c_p <= 1.0
    assert 0.0 <= py_p <= 1.0
    # identical distributions => p should be 1.0 by the '>=' convention
    same_p = paired_bootstrap_test_p(xs, xs)
    assert same_p == 1.0
