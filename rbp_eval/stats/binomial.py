"""Port of stats/binomial.c.

The original computes the upper-tail binomial probability via a
recurrence intended to avoid the overflow of naive double-precision
sums. In practice it overflows to `inf` far earlier than its own comment
claims ("doubles start overflowing around the 10,000 trial mark") --
verified overflowing by ~1100-1200 trials at p=0.5, e.g.
`cumulative_binomial_p(5000, 2600, 0.5)` returns `inf` from the C tool.
That silently breaks the sign test on any IR experiment with 1000+
queries, which is routine at modern scale. scipy.stats.binom.sf has no
such issue, so this is a straight substitution -- and a correctness fix,
not just a convenience one.
"""

from __future__ import annotations

from scipy import stats as _stats


def cumulative_binomial_p(trials: int, successes: int, probability: float) -> float:
    """P(X >= successes) for X ~ Binomial(trials, probability)."""
    assert 0 <= successes <= trials
    assert 0.0 <= probability <= 1.0
    return float(_stats.binom.sf(successes - 1, trials, probability))
