"""Differential test: validates the Python RBP port against the existing
hand-verified `.spec` fixtures in ../../rbp_eval/test/data/ (from the
original C project's `rbp_tester` test harness -- see tester.c and
test/data/README.txt in the C tree).

Each .spec file's header line is a subset of rbp_eval's CLI options
followed by a qrels filename and a run filename; the body then lists
expected num_rel/num_ret/num_rel_ret/sum/err values per query and
averaged, to 4 decimal places.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rbp_eval.evaluate import Results, evaluate
from rbp_eval.qrels import RelType, load_qrels
from rbp_eval.ranking import Ordering
from rbp_eval.runs import load_run
from rbp_eval.specs import parse_depth, parse_persist

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "rbp_eval" / "test" / "data"
MARGIN_OF_ERROR = 0.0001


def _spec_files() -> list[Path]:
    if not FIXTURE_DIR.is_dir():
        return []
    return sorted(FIXTURE_DIR.glob("*.spec"))


class _LineReader:
    """Yields non-comment lines one at a time, like tester.c's GETLINE
    (lines starting with '#' are discarded)."""

    def __init__(self, text: str):
        self._lines = iter(text.splitlines())

    def next(self) -> str:
        for line in self._lines:
            if line.startswith("#"):
                continue
            return line
        raise EOFError("premature end of spec file")


def _parse_header(header: str):
    """Parse the subset of rbp_eval's CLI options exercised by the test
    fixtures: -d, -p, -r/-s/-o, -b/-B/-f/-F/-a, followed by qrels and run
    filenames."""
    tokens = header.split()
    qrels_fname, run_fname = tokens[-2], tokens[-1]
    opts = tokens[:-2]

    depths: list[float] | None = None
    persist: list[float] | None = None
    ordering = Ordering.SCORE
    reltype = RelType.AUTO
    reltype_arg = 1.0

    i = 0
    while i < len(opts):
        tok = opts[i]
        if tok == "-d":
            depths = parse_depth(opts[i + 1])
            i += 2
        elif tok == "-p":
            persist = parse_persist(opts[i + 1])
            i += 2
        elif tok == "-r":
            ordering = Ordering.RANK
            i += 1
        elif tok == "-s":
            ordering = Ordering.SCORE
            i += 1
        elif tok == "-o":
            ordering = Ordering.OCCUR
            i += 1
        elif tok == "-b":
            reltype = RelType.BINARY
            reltype_arg = float(opts[i + 1])
            i += 2
        elif tok == "-B":
            reltype = RelType.BINARY
            reltype_arg = 1.0
            i += 1
        elif tok == "-f":
            reltype = RelType.FRACT
            reltype_arg = float(opts[i + 1])
            i += 2
        elif tok == "-F":
            reltype = RelType.FRACT
            i += 1
        elif tok == "-a":
            reltype = RelType.AUTO
            i += 1
        else:
            raise ValueError(f"unsupported test-spec option {tok!r}")

    assert depths is not None and persist is not None
    return qrels_fname, run_fname, depths, persist, ordering, reltype, reltype_arg


def _check_float(reader: _LineReader, actual: float, what: str) -> None:
    expected = float(reader.next())
    assert abs(actual - expected) <= MARGIN_OF_ERROR, (
        f"{what}: got {actual}, expected {expected}"
    )


def _check_against_spec(res: Results, reader: _LineReader) -> None:
    blocks = list(res.qid_res) + [res.ave_res]
    for qres in blocks:
        _check_float(reader, qres.num_rel, f"num_rel for qid={qres.qid}")
        _check_float(reader, qres.num_ret, f"num_ret for qid={qres.qid}")
        for d, dres in enumerate(qres.depth_res):
            _check_float(reader, dres.num_rel_ret, f"num_rel_ret qid={qres.qid} d={d}")
            for p, pres in enumerate(dres.persist_res):
                sum_str, err_str = reader.next().split()
                assert abs(pres.sum - float(sum_str)) <= MARGIN_OF_ERROR, (
                    f"sum qid={qres.qid} d={d} p={p}: got {pres.sum}, "
                    f"expected {sum_str}"
                )
                assert abs(pres.err - float(err_str)) <= MARGIN_OF_ERROR, (
                    f"err qid={qres.qid} d={d} p={p}: got {pres.err}, "
                    f"expected {err_str}"
                )


@pytest.mark.parametrize(
    "spec_path", _spec_files(), ids=lambda p: p.name if _spec_files() else "no-fixtures"
)
def test_matches_c_spec(spec_path: Path):
    reader = _LineReader(spec_path.read_text())
    header = reader.next()
    qrels_fname, run_fname, depths, persist, ordering, reltype, reltype_arg = (
        _parse_header(header)
    )

    with open(spec_path.parent / qrels_fname) as fp:
        qrels = load_qrels(fp)
    qrels.set_reltype(reltype, reltype_arg)
    with open(spec_path.parent / run_fname) as fp:
        run = load_run(fp)

    res = evaluate(qrels, run, ordering, persist, depths)
    _check_against_spec(res, reader)


def test_fixtures_found():
    assert _spec_files(), f"no .spec fixtures found under {FIXTURE_DIR}"
