"""Port of stats/kendall.c.

Not a substitution for scipy.stats.kendalltau: this is *not* tau-b. Ties
are scored as fully concordant if both sequences tie at that pair, else
fully discordant, and the denominator is always n(n-1)/2 (no tie
correction). scipy's default ('b' variant) disagrees whenever ties are
present, so it cannot be swapped in without changing results.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def kendall_tau(dat1: Sequence[float], dat2: Sequence[float]) -> float:
    n = len(dat1)
    if n < 2:
        return 1.0

    x = np.asarray(dat1, dtype=float)
    y = np.asarray(dat2, dtype=float)
    iu = np.triu_indices(n, k=1)
    dx = (x[:, None] - x[None, :])[iu]
    dy = (y[:, None] - y[None, :])[iu]

    # Concordant iff the two pairwise differences agree in sign --
    # including the tie/tie case (both signs 0). Any other combination
    # (a strict difference against a tie, or opposite strict directions)
    # is discordant. This matches the original's three-way branch on
    # dat1[i] vs dat1[j] exactly.
    concordant = int(np.count_nonzero(np.sign(dx) == np.sign(dy)))
    discordant = dx.size - concordant

    tau = (concordant - discordant) / (n * (n - 1) / 2)
    assert -1.0 <= tau <= 1.0
    return tau
