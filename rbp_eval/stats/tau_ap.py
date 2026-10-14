"""AP correlation coefficient (Yilmaz, Aslam & Robertson, "A New Rank
Correlation Coefficient for Information Retrieval", SIGIR 2008).

NOT a port: the original stats/tau_ap.cpp only ever implemented a
binary-search-tree helper (CtBNode) intended for efficiently counting
concordant pairs -- the actual tau_ap coefficient was never assembled on
top of it, and its own "test" driver just printed a subtree count. This is
written fresh from the published formula.

tau_ap weights agreement near the top of the ranking more heavily than
agreement at the bottom (unlike Kendall's tau, which weights all
positions equally), which is why it's commonly used to compare system
rankings produced by different IR effectiveness measures.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def tau_ap(dat1: Sequence[float], dat2: Sequence[float]) -> float:
    """dat1 and dat2 give paired values (e.g. two effectiveness scores)
    for the same n items, in matching order.

    dat1 defines the reference order: items are considered from highest
    to lowest dat1 value (ties broken by input order). For each item at
    reference position i (2-indexed from 1), concordance is checked
    against dat2. The coefficient is *not* symmetric in (dat1, dat2) --
    that's the paper's definition, not a bug -- so swap the arguments to
    get the other direction.

    Ties in dat2 are not part of the formal definition (the paper assumes
    strict rankings); here, a tie in dat2 for a given pair is counted as
    neither concordant nor discordant.
    """
    n = len(dat1)
    if n < 2:
        return 1.0

    order = np.argsort(-np.asarray(dat1, dtype=float), kind="stable")
    ranked2 = np.asarray(dat2, dtype=float)[order]

    total = 0.0
    for i in range(1, n):
        c_i = int(np.count_nonzero(ranked2[:i] > ranked2[i]))
        total += c_i / i

    return -1.0 + (2.0 / (n - 1)) * total
