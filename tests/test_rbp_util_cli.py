"""Differential tests: pooljudge/minavgerr/minmaxerr against the compiled
C binaries, across a representative sweep of the shared common.c options
plus each tool's own flags (weighting schemes, significance modes/tests,
logging, charting, report-depths files, non-contributing runs, stop-at-
top-run).

Two known, accepted divergence classes (documented at the point they're
exercised below, not glossed over):

  * Tie-break order among equal-weight documents (in docwgt's sort, and
    in minavgerr's weighted-recalculation heap) is implementation-defined
    in the original -- its qsort comparator returns 0 for ties, and qsort
    itself is not stable. Verified: the *final* aggregate stats
    (report_runerr_stats' numeric line) match exactly in every case
    tested; only the diagnostic "Unjudged: ... <count>" line-by-line
    messages can shift by small amounts when two documents tie exactly.
  * Running any of these three tools without -Q segfaults the compiled C
    binary (a real NULL-pointer-dereference bug in init_runerr, described
    in rbp_util/common.py) -- this port raises a clear error instead.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "rbp_eval" / "test" / "data"
RBP_UTIL_DIR = Path(__file__).resolve().parents[2] / "rbp_util"


def _c_binary(name: str) -> Path:
    return RBP_UTIL_DIR / name


def _run_c(name: str, args: list[str]):
    return subprocess.run(
        [str(_c_binary(name)), *args], cwd=FIXTURE_DIR, capture_output=True, text=True
    )


def _run_py(module: str, args: list[str]):
    return subprocess.run(
        [sys.executable, "-m", f"rbp_eval.cli.{module}", *args],
        cwd=FIXTURE_DIR,
        capture_output=True,
        text=True,
    )


def _skip_if_missing(name: str):
    return pytest.mark.skipif(
        not _c_binary(name).is_file(), reason=f"C {name} binary not built"
    )


EXACT_CASES = [
    ("pooljudge", "pooljudge", ["-Q", "basic.qrels", "-d", "3", "noties.run", "ties.run"]),
    (
        "pooljudge",
        "pooljudge",
        ["-Q", "basic.qrels", "-N", "test", "-d", "3", "noties.run", "ties.run"],
    ),
    (
        "pooljudge",
        "pooljudge",
        ["-Q", "basic.qrels", "-T", "noties.run", "ties.run", "manyties.run"],
    ),
    (
        "minmaxerr",
        "minmaxerr",
        ["-Q", "basic.qrels", "noties.run", "ties.run", "manyties.run"],
    ),
    (
        "minmaxerr",
        "minmaxerr",
        ["-Q", "basic.qrels", "-W", "quadratic", "noties.run", "ties.run", "manyties.run"],
    ),
    (
        "minmaxerr",
        "minmaxerr",
        [
            "-Q", "basic.qrels", "-P", "sign", "-m", "conservative", "-G", "0.5",
            "noties.run", "ties.run", "manyties.run",
        ],
    ),
    (
        "minmaxerr",
        "minmaxerr",
        ["-Q", "basic.qrels", "-P", "t", "noties.run", "ties.run", "manyties.run"],
    ),
    (
        "minmaxerr",
        "minmaxerr",
        [
            "-Q", "basic.qrels", "-P", "wilcoxon", "-m", "projected", "-G", "0.6",
            "noties.run", "ties.run", "manyties.run",
        ],
    ),
    (
        "minmaxerr",
        "minmaxerr",
        [
            "-Q", "basic.qrels", "-P", "sign", "-m", "pessimal", "-G", "1.0",
            "noties.run", "ties.run", "manyties.run",
        ],
    ),
    (
        "minmaxerr",
        "minmaxerr",
        ["-Q", "basic.qrels", "-U", "0.3", "noties.run", "ties.run", "manyties.run"],
    ),
    (
        "minavgerr",
        "minavgerr",
        ["-Q", "basic.qrels", "-W", "quadratic", "noties.run", "ties.run", "manyties.run"],
    ),
]


@pytest.mark.parametrize(
    "c_name,py_module,args", EXACT_CASES, ids=lambda v: v if isinstance(v, str) else " ".join(v)
)
def test_exact_match(c_name, py_module, args):
    if not _c_binary(c_name).is_file():
        pytest.skip(f"C {c_name} binary not built")
    c_proc = _run_c(c_name, args)
    py_proc = _run_py(py_module, args)
    assert c_proc.stdout == py_proc.stdout
    assert c_proc.stderr == py_proc.stderr


@_skip_if_missing("minavgerr")
@pytest.mark.parametrize(
    "args",
    [
        ["-Q", "basic.qrels", "noties.run", "ties.run", "manyties.run"],
        ["-Q", "basic.qrels", "-W", "linear", "-q", "linear", "noties.run", "ties.run", "manyties.run"],
        ["-Q", "basic.qrels", "-E", "0.5", "noties.run", "ties.run", "manyties.run"],
    ],
    ids=["uniform", "linear-both", "min-avg-err"],
)
def test_minavgerr_final_stats_match_despite_tie_order(args):
    """Documents the tie-order divergence: only the final
    report_runerr_stats line is compared here, not the per-document
    "Unjudged" ordering (which can differ when two documents tie
    exactly -- see module docstring)."""
    c_proc = _run_c("minavgerr", args)
    py_proc = _run_py("minavgerr", args)
    c_last_line = c_proc.stdout.splitlines()[-1]
    py_last_line = py_proc.stdout.splitlines()[-1]
    assert c_last_line == py_last_line

    c_unjudged = sorted(
        line.split(":")[0] for line in c_proc.stderr.splitlines() if line.startswith("Unjudged")
    )
    py_unjudged = sorted(
        line.split(":")[0] for line in py_proc.stderr.splitlines() if line.startswith("Unjudged")
    )
    assert c_unjudged == py_unjudged


@_skip_if_missing("pooljudge")
def test_missing_qrels_crashes_c_but_not_python():
    c_proc = _run_c("pooljudge", ["noties.run"])
    assert c_proc.returncode == -11, "expected the known C segfault to still reproduce"

    py_proc = _run_py("pooljudge", ["noties.run"])
    assert py_proc.returncode == 1
    assert "no qrels file given" in py_proc.stderr


@_skip_if_missing("pooljudge")
def test_judgment_and_lacking_logs_match(tmp_path):
    c_jlog, py_jlog = tmp_path / "c_j.txt", tmp_path / "py_j.txt"
    c_llog, py_llog = tmp_path / "c_l.txt", tmp_path / "py_l.txt"
    args_c = ["-Q", "basic.qrels", "-j", str(c_jlog), "-L", str(c_llog), "-d", "3",
              "noties.run", "ties.run"]
    args_py = ["-Q", "basic.qrels", "-j", str(py_jlog), "-L", str(py_llog), "-d", "3",
               "noties.run", "ties.run"]

    c_proc = _run_c("pooljudge", args_c)
    py_proc = _run_py("pooljudge", args_py)
    assert c_proc.stdout == py_proc.stdout
    assert c_jlog.read_text() == py_jlog.read_text()
    assert c_llog.read_text() == py_llog.read_text()


@_skip_if_missing("minmaxerr")
def test_score_and_significance_logs_match(tmp_path):
    c_slog, py_slog = tmp_path / "c_s.txt", tmp_path / "py_s.txt"
    c_zlog, py_zlog = tmp_path / "c_z.txt", tmp_path / "py_z.txt"
    common = ["-Q", "basic.qrels", "-z", "5", "-P", "sign", "noties.run", "ties.run", "manyties.run"]

    c_proc = _run_c("minmaxerr", ["-S", str(c_slog), "-Z", str(c_zlog), *common])
    py_proc = _run_py("minmaxerr", ["-S", str(py_slog), "-Z", str(py_zlog), *common])
    assert c_proc.stdout == py_proc.stdout
    assert c_slog.read_text() == py_slog.read_text()
    assert c_zlog.read_text() == py_zlog.read_text()


@_skip_if_missing("minmaxerr")
def test_report_depths_file_matches(tmp_path):
    depths_file = tmp_path / "depths.txt"
    depths_file.write_text("3\n7\n")
    args = ["-Q", "basic.qrels", "-D", str(depths_file), "noties.run", "ties.run", "manyties.run"]

    c_proc = _run_c("minmaxerr", args)
    py_proc = _run_py("minmaxerr", args)
    assert c_proc.stdout == py_proc.stdout
    assert c_proc.stderr == py_proc.stderr


@_skip_if_missing("pooljudge")
def test_chart_output_matches(tmp_path):
    c_dir, py_dir = tmp_path / "c_chart", tmp_path / "py_chart"
    c_dir.mkdir()
    py_dir.mkdir()
    args_c = ["-Q", "basic.qrels", "-C", str(c_dir), "-d", "2", "noties.run", "ties.run"]
    args_py = ["-Q", "basic.qrels", "-C", str(py_dir), "-d", "2", "noties.run", "ties.run"]

    _run_c("pooljudge", args_c)
    _run_py("pooljudge", args_py)

    c_files = sorted(p.name for p in c_dir.iterdir())
    py_files = sorted(p.name for p in py_dir.iterdir())
    assert c_files == py_files
    for name in c_files:
        assert (c_dir / name).read_text() == (py_dir / name).read_text()
