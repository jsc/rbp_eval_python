"""Unit tests for stats functions that have no C counterpart to diff
against directly (tau_ap is greenfield) or need a basic sanity check on
their own combinatorics (wilcoxon's outcome table)."""

from __future__ import annotations

import pytest

from rbp_eval.stats import tau_ap
from rbp_eval.stats.wilcoxon import generate_wilcoxon_outcomes


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 12])
def test_wilcoxon_outcomes_sum_to_2_pow_n(n):
    outcomes = generate_wilcoxon_outcomes(n)
    assert len(outcomes) == n * (n + 1) // 2 + 1
    assert sum(outcomes) == 2**n


def test_tau_ap_identical_rankings_is_one():
    dat1 = [5.0, 4.0, 3.0, 2.0, 1.0]
    assert tau_ap(dat1, dat1) == pytest.approx(1.0)


def test_tau_ap_weights_top_more_than_kendall_would():
    # Swapping the top two items should hurt tau_ap more than swapping
    # the bottom two, since it weights agreement near the top more
    # heavily.
    reference = [5.0, 4.0, 3.0, 2.0, 1.0]
    swap_top = [4.0, 5.0, 3.0, 2.0, 1.0]
    swap_bottom = [5.0, 4.0, 3.0, 1.0, 2.0]

    top_penalty = tau_ap(reference, swap_top)
    bottom_penalty = tau_ap(reference, swap_bottom)
    assert top_penalty < bottom_penalty


def test_tau_ap_single_item_or_empty_is_one():
    assert tau_ap([1.0], [1.0]) == 1.0
    assert tau_ap([], []) == 1.0
