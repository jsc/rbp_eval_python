"""Port of dcg_eval/main.cpp's per-query (n)DCG computation.

Discounting matches the classic formula: rank 1 gets full credit
undiscounted, and from rank `base` onward (default base=2, so from rank 2)
relevance is divided by log_base(rank). Passing rank_adjust=1 (the CLI's
-R, "MS"/'non-stupid' DCG) shifts this so every rank, including rank 1,
is discounted by log_base(rank+1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .qrels import Qrels
from .ranking import DocScore


@dataclass
class DcgOptions:
    base: float = 2.0
    depth: int = 1000
    rank_adjust: int = 0  # -R
    discounted: bool = True
    normalised: bool = True


def _discounted_rel(rel: float, rank0: int, opts: DcgOptions, log_base: float) -> float:
    """rank0 is the 0-indexed rank."""
    rank1 = rank0 + 1 + opts.rank_adjust
    if opts.discounted and rank1 >= opts.base:
        return rel / (math.log(rank1) / log_base)
    return rel


def compute_dcg(
    qrels: Qrels, qid: str, doc_scores: list[DocScore], opts: DcgOptions
) -> float | None:
    """doc_scores must already be ordered by score (descending), i.e.
    qdocs.get_scores(Ordering.SCORE) -- dcg_eval always uses score
    ordering, unlike rbp_eval which lets you choose.

    Returns None if there are no qrels judgments at all for this query
    (caller should skip/warn, matching main.cpp).
    """
    qq = qrels.get_qid_qrels(qid)
    if qq is None:
        return None

    log_base = math.log(opts.base)
    run_length = min(len(doc_scores), opts.depth)

    score = 0.0
    for r in range(run_length):
        rel = qq.get_rel(doc_scores[r].docid)
        if rel > 0.0:
            score += _discounted_rel(rel, r, opts, log_base)

    if opts.normalised:
        optimal_rels = sorted((rel for _docid, rel in qq if rel > 0.0), reverse=True)
        optimal_score = sum(
            _discounted_rel(rel, r, opts, log_base) for r, rel in enumerate(optimal_rels)
        )
        if optimal_score > 0:
            score /= optimal_score

    return score
