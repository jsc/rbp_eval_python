"""Port of librbp/rbp.[ch]: the Rank-Biased Precision calculation itself.

This is more than the textbook RBP formula because it has to support:

  1. tied rankings,
  2. evaluation cutoff at arbitrary depths,
  3. tied rankings that span an evaluation depth cutoff, and
  4. multiple evaluation depth cutoffs, with calculation of each cutoff
     continuing from where the previous one left off.

Tied rankings are handled by assigning to each element of the tie the
average weight for the positions over which the tie extends. Cutoffs in
the middle of a tie are handled by including all tied elements before the
cutoff, but weighting each by the proportion of the tie that fits into the
cutoff. Restartable multi-depth evaluations mean that the span and weights
of ties hanging over from the previous depth cutoff have to be
incorporated into the following one. In degenerate cases, a tie may start
before the continued-from cutoff, stretch across the span of the
continued evaluation, and extend beyond the continuation's end cutoff.

Consider an evaluation made to depths 2, 5, and 10, on a ranking where
five documents are tied at rank 2. Let w(n) be the weighting given to a
document at rank n. Every document at rank 2 gets a base weighting of
w' = (w(2)+w(3)+w(4)+w(5)+w(6)) / 5. For the evaluation to depth 2,
one-fifth of the tie at positions 2-6 fits into that segment, so the five
documents each contribute w'/5. Continuing to depth 5, three-fifths of the
tie fits into the next segment, so each contributes 3w'/5. Continuing to
depth 10, the final one-fifth contributes w'/5 more. Note that at each
depth cutoff d, the residual uncertainty is always p^d, even if a tie
straddles the boundary.

Deviation from the original C (see Rbp._reached_depth below): the C
version re-derives its "has anything changed since last call" check from
the *raw* depth argument of the previous call, not from the depth actually
reached (capped at the number of retrieved documents). If a query returns
fewer documents than a later requested cutoff, after an earlier cutoff was
already capped, the C code re-enters the tie-scan at an already-exhausted
position, triggering `assert(t > pos)` (aborts, in an assert-enabled
build) or a division by a zero-length tie (NaN, in an NDEBUG build). This
port tracks the capped depth instead, which is a strict bug fix: it is
behaviourally identical whenever a query's retrieved-document count is at
least as large as every requested depth (true of every existing test
fixture), and only changes behaviour in the previously-broken case.
"""

from __future__ import annotations

from dataclasses import dataclass

from .qrels import Qrels, REL_UNJUDGED
from .ranking import DocScore, Ordering, QueryDocs


@dataclass
class RbpVal:
    """Per-persistence-value running state and result. Equivalent of
    rbp_val_t."""

    sum: float = 0.0
    cumerr: float = 0.0
    wgt: float = 0.0
    tie_wgt: float = 0.0
    err: float = 0.0


def rbp_point(persist: float, rank: int) -> float:
    """The (uncapped) RBP weight contributed by a single document at
    0-indexed position `rank`, for a given persistence value."""
    return (1 - persist) * persist**rank


def rbp_weights(persist: float, depth: int) -> list[float]:
    """RBP weights at each position up to (not including) `depth`."""
    wgt = 1 - persist
    out = []
    for _ in range(depth):
        out.append(wgt)
        wgt *= persist
    return out


class Rbp:
    """Restartable, tie-aware RBP calculator for a single query at a time.
    Port of librbp/rbp.c's rbp_t / rbp_calc_to_depth."""

    def __init__(self, qrels: Qrels, ordering: Ordering, persist: list[float]):
        self.qrels = qrels
        self.ordering = ordering
        self.persist = persist
        self.qdocs: QueryDocs | None = None
        self.vals: list[RbpVal] = []
        self.num_rel_ret = 0.0
        self.tie_len = 0
        self.tie_pos = 0
        # Depth actually reached so far, capped at the query's
        # retrieved-document count. See module docstring.
        self._reached_depth = 0
        # Last *requested* depth, purely to enforce the "depth must
        # increase on each call" contract (mirrors the C assert).
        self._last_requested_depth = 0

    def init(self, qdocs: QueryDocs) -> None:
        """(Re-)initialise for processing of a ranking for a particular
        query. Raises ValueError if there are no qrels judgments at all
        for this query (mirrors rbp_init's -1 return)."""
        if self.qrels.get_qid_qrels(qdocs.qid) is None:
            raise ValueError(f"no qrels judgments for qid {qdocs.qid!r}")
        self.qdocs = qdocs
        self.vals = [RbpVal(wgt=1 - p) for p in self.persist]
        self.num_rel_ret = 0.0
        self.tie_len = 0
        self.tie_pos = 0
        self._reached_depth = 0
        self._last_requested_depth = 0

    def _tie_key(self, ds: DocScore):
        if self.ordering is Ordering.RANK:
            return ds.rank
        if self.ordering is Ordering.SCORE:
            return ds.score
        return ds.occur

    def calc_to_depth(self, depth: float) -> tuple[list[RbpVal], float]:
        """Extend the calculation up to `depth` (which must be strictly
        larger than the depth passed to any previous call for this query).
        Returns (per-persistence RbpVal list, cumulative num_rel_ret).

        The returned list is reused/mutated on the next call -- copy it if
        you need to keep a snapshot.
        """
        assert self.qdocs is not None, "call init() first"
        assert depth > self._last_requested_depth
        self._last_requested_depth = depth

        qdocs = self.qdocs
        num_scores = qdocs.num_scores()
        doc_scores = qdocs.get_scores(self.ordering)
        qq = self.qrels.get_qid_qrels(qdocs.qid)
        assert qq is not None

        actual_depth = min(num_scores, depth)
        if actual_depth == self._reached_depth:
            return self.vals, self.num_rel_ret

        while True:
            if self.tie_len == 0:
                pos = self.tie_pos
                for v in self.vals:
                    v.tie_wgt = 0.0
                # Search forward to see how many documents share this
                # rank/score/occurrence. Starting t at pos guarantees a
                # tie length of at least 1.
                t = pos
                while t < num_scores and self._tie_key(doc_scores[pos]) == self._tie_key(
                    doc_scores[t]
                ):
                    for v, p in zip(self.vals, self.persist):
                        v.tie_wgt += v.wgt
                        v.wgt *= p
                    t += 1
                assert t > pos
                self.tie_len = t - pos
                for v in self.vals:
                    v.tie_wgt /= self.tie_len

            tie_frac = (
                min(self.tie_pos + self.tie_len, depth)
                - max(self.tie_pos, self._reached_depth)
            ) / self.tie_len

            end_pos = self.tie_pos + self.tie_len
            for d in range(self.tie_pos, end_pos):
                rel = qq.get_rel(doc_scores[d].docid)
                if rel != REL_UNJUDGED:
                    self.num_rel_ret += rel * tie_frac
                for v in self.vals:
                    if rel == REL_UNJUDGED:
                        v.cumerr += v.tie_wgt * tie_frac
                    else:
                        v.sum += v.tie_wgt * rel * tie_frac

            if end_pos <= actual_depth:
                self.tie_pos = end_pos
                self.tie_len = 0
            if end_pos >= actual_depth:
                break

        for v, p in zip(self.vals, self.persist):
            v.err = v.cumerr + p**actual_depth
        self._reached_depth = actual_depth

        return self.vals, self.num_rel_ret
