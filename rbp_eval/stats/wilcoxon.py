"""Port of stats/wilcoxon.[ch]: exact one-tailed paired Wilcoxon signed-rank
test, for any sample size.

Not a substitution for scipy.stats.wilcoxon: scipy's default falls back to
a normal approximation once n is large (its "exact" method only applies
below its own size threshold), whereas this computes the exact null
distribution of W+ via a subset-sum DP for arbitrary n, matching the
original tool's behaviour.

Ties are handled by averaging ranks across a tied block, as usual -- but
when that produces a non-integer W+, the original rounds *up* to the next
whole outcome rather than interpolating between outcomes. The original's
own comment flags this as not fully confidence-inspiring ("The
description of the Wilcoxon test are unclear on what should be done in
the case of half-values"); it is preserved here for fidelity rather than
"fixed", since there is no clearly-more-correct alternative on offer.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def generate_wilcoxon_outcomes(dist_size: int) -> list[int]:
    """outcomes[k] = number of subsets of {1, ..., dist_size} that sum to
    k, for every achievable W+ value k in [0, dist_size*(dist_size+1)/2].
    This is the null distribution of the Wilcoxon W+ statistic under
    random sign assignment (2**dist_size equally likely sign patterns).
    """
    num_outcomes = dist_size * (dist_size + 1) // 2 + 1
    outcomes = [0] * num_outcomes
    outcomes[0] = 1
    total = 0
    for v in range(1, dist_size + 1):
        total += v
        for k in range(total, v - 1, -1):
            outcomes[k] += outcomes[k - v]
    return outcomes


def paired_wilcoxon_test_p(dist_x: Sequence[float], dist_y: Sequence[float]) -> float:
    diffs = [x - y for x, y in zip(dist_x, dist_y) if x != y]
    n = len(diffs)
    if n == 0:
        return 1.0

    outcomes = generate_wilcoxon_outcomes(n)
    diffs.sort(key=abs)

    wplus = 0.0
    i = 0
    while i < n:
        abs_diff = abs(diffs[i])
        j = i + 1
        while j < n and abs(diffs[j]) == abs_diff:
            j += 1
        mean_rank = (i + j + 1) / 2.0
        for k in range(i, j):
            if diffs[k] > 0.0:
                wplus += mean_rank
        i = j

    threshold = math.ceil(wplus)
    critical_region_size = sum(outcomes[threshold:])
    return critical_region_size / (2.0**n)
