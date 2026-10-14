"""Port of stats/t.c: one-tailed paired t-test.

Deliberate upgrade (per user decision), not a faithful port: the original
did not compute a real p-value at all -- it looked the t statistic up
against a hardcoded table of only 8 critical-value buckets (p = 0.1 down
to 0.00005), i.e. it returned a coarse step function, not a continuous
p-value. paired_t_test_p here uses scipy.stats.ttest_rel for a proper
continuous p-value; it will not numerically match the original tool's
bucketed output.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from scipy import stats as _stats


def paired_t_test(dist_x: Sequence[float], dist_y: Sequence[float]) -> float:
    """The raw one-tailed paired t statistic (same definition as the
    original: t = mean(x - y) * sqrt(n) / sd(x - y))."""
    diffs = [x - y for x, y in zip(dist_x, dist_y)]
    n = len(diffs)
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    sd = math.sqrt(var)
    return mean * math.sqrt(n) / sd


def paired_t_test_p(dist_x: Sequence[float], dist_y: Sequence[float]) -> float:
    """One-tailed paired t-test p-value for H1: mean(dist_x) > mean(dist_y)."""
    result = _stats.ttest_rel(dist_x, dist_y, alternative="greater")
    return float(result.pvalue)
