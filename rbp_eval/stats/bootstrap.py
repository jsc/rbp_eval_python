"""Port of stats/bootstrap.c: paired bootstrap ASL (achieved significance
level) test, per Efron & Tibshirani.

The original seeds its RNG from the process id (`srandom(getpid())`), so
it was never reproducible even between two runs of the same C binary --
bit-for-bit fidelity to a specific random stream was never a property of
the original, so there's nothing to match here. This vectorizes the
resampling with numpy instead of a per-trial loop.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

NUM_TRIALS = 1000


def paired_bootstrap_test_p(
    dist_x: Sequence[float],
    dist_y: Sequence[float],
    *,
    num_trials: int = NUM_TRIALS,
    rng: np.random.Generator | None = None,
) -> float:
    diff = np.asarray(dist_x, dtype=float) - np.asarray(dist_y, dtype=float)
    n = diff.size
    mean_diff = diff.mean()
    w = diff - mean_diff  # shift to zero mean, i.e. impose the null hypothesis

    if rng is None:
        rng = np.random.default_rng()
    samples = rng.choice(w, size=(num_trials, n), replace=True)
    sample_means = samples.mean(axis=1)

    # '>=' rather than '>': gives p = 1.0 (rather than 0.0) when dist_x
    # and dist_y are identical, matching the original's explicit choice.
    asl_count = int(np.count_nonzero(sample_means >= mean_diff))
    return asl_count / num_trials
