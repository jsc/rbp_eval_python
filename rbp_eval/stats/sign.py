"""Port of stats/sign.c: one-tailed paired sign test."""

from __future__ import annotations

from collections.abc import Sequence

from .binomial import cumulative_binomial_p


def paired_sign_test_p(dist_x: Sequence[float], dist_y: Sequence[float]) -> float:
    pairs_greater = 0
    pairs_not_equal = len(dist_x)
    for x, y in zip(dist_x, dist_y):
        if x > y:
            pairs_greater += 1
        elif x == y:
            pairs_not_equal -= 1
    return cumulative_binomial_p(pairs_not_equal, pairs_greater, 0.5)
