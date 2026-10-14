"""Port of librbp/res.[ch]: ties qrels + run + ordering + persist + depths
together into per-query and averaged RBP results."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .qrels import Qrels
from .ranking import Ordering
from .rbp import Rbp
from .runs import Run


@dataclass
class PersistRes:
    sum: float = 0.0
    err: float = 0.0


@dataclass
class DepthRes:
    num_rel_ret: float = 0.0
    persist_res: list[PersistRes] = field(default_factory=list)


@dataclass
class QidRes:
    qid: str | None = None
    num_ret: int = 0
    num_rel: float = 0.0  # < 0.0 means "no judgments"
    depth_res: list[DepthRes] = field(default_factory=list)


@dataclass
class Results:
    depths: list[float]
    persist: list[float]
    qid_res: list[QidRes]
    ave_res: QidRes


def _new_qid_res(qid: str | None, depths: list[float], persist: list[float]) -> QidRes:
    return QidRes(
        qid=qid,
        depth_res=[
            DepthRes(persist_res=[PersistRes() for _ in persist]) for _ in depths
        ],
    )


def evaluate(
    qrels: Qrels, run: Run, ordering: Ordering, persist: list[float], depths: list[float]
) -> Results:
    ave = _new_qid_res("all", depths, persist)
    qid_res = [_new_qid_res(None, depths, persist) for _ in range(run.num_qdocs())]

    rbp = Rbp(qrels, ordering, persist)
    num_judged_queries = 0

    for q in range(run.num_qdocs()):
        qd = run.get_qdocs_by_index(q)
        qres = qid_res[q]
        qres.qid = qd.qid
        qres.num_ret = qd.num_scores()
        qres.num_rel = qrels.get_num_rel(qd.qid)
        if qres.num_rel < 0.0:
            continue
        num_judged_queries += 1
        ave.num_rel += qres.num_rel
        ave.num_ret += qres.num_ret

        rbp.init(qd)
        for d, depth in enumerate(depths):
            rbpvals, num_rel_ret = rbp.calc_to_depth(depth)
            dres = qres.depth_res[d]
            ave_dres = ave.depth_res[d]
            dres.num_rel_ret = num_rel_ret
            ave_dres.num_rel_ret += num_rel_ret
            for p in range(len(persist)):
                dres.persist_res[p].sum = rbpvals[p].sum
                dres.persist_res[p].err = rbpvals[p].err
                ave_dres.persist_res[p].sum += rbpvals[p].sum
                ave_dres.persist_res[p].err += rbpvals[p].err

    for d in range(len(depths)):
        for p in range(len(persist)):
            pres = ave.depth_res[d].persist_res[p]
            if num_judged_queries > 0:
                pres.sum /= num_judged_queries
                pres.err /= num_judged_queries
            else:
                pres.sum = math.nan
                pres.err = math.nan

    return Results(depths=depths, persist=persist, qid_res=qid_res, ave_res=ave)
