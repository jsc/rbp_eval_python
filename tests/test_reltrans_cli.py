"""Differential test: reltrans CLI against the compiled C binary.

Includes a regression test for a real bug found in the C original:
do_trans() segfaults (NULL-pointer dereference in qrels_delete) if the
qrels file can't be opened. This port fixes that -- verified separately
that the crash still reproduces on the compiled C binary, then asserts
the Python port exits cleanly instead.
"""

from __future__ import annotations

import filecmp
import subprocess
import sys
from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "rbp_eval" / "test" / "data"
C_BINARY = Path(__file__).resolve().parents[2] / "rbp_util" / "reltrans"


def _dirs_equal(a: Path, b: Path) -> bool:
    cmp = filecmp.dircmp(a, b)
    if cmp.left_only or cmp.right_only or cmp.diff_files or cmp.funny_files:
        return False
    for sub in cmp.common_dirs:
        if not _dirs_equal(a / sub, b / sub):
            return False
    return True


@pytest.mark.skipif(not C_BINARY.is_file(), reason="C reltrans binary not built")
def test_reltrans_matches_c_binary(tmp_path):
    c_out = tmp_path / "c_out"
    py_out = tmp_path / "py_out"
    c_out.mkdir()
    py_out.mkdir()

    subprocess.run(
        [str(C_BINARY), "-d", str(c_out), "basic.qrels", "noties.run"],
        cwd=FIXTURE_DIR,
        capture_output=True,
        text=True,
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "rbp_eval.cli.reltrans", "-d", str(py_out),
         "basic.qrels", "noties.run"],
        cwd=FIXTURE_DIR,
        capture_output=True,
        text=True,
        check=True,
    )

    assert _dirs_equal(c_out, py_out)


@pytest.mark.skipif(not C_BINARY.is_file(), reason="C reltrans binary not built")
def test_missing_qrels_crashes_c_but_not_python(tmp_path):
    c_proc = subprocess.run(
        [str(C_BINARY), "-d", str(tmp_path), "nonexistent.qrels", "noties.run"],
        cwd=FIXTURE_DIR,
        capture_output=True,
        text=True,
    )
    # subprocess reports a signal-terminated process as -signum (here SIGSEGV,
    # -11); a shell's $? would show the 128+signum convention (139) instead.
    assert c_proc.returncode == -11, "expected the known C segfault to still reproduce"

    py_proc = subprocess.run(
        [sys.executable, "-m", "rbp_eval.cli.reltrans", "-d", str(tmp_path),
         "nonexistent.qrels", "noties.run"],
        cwd=FIXTURE_DIR,
        capture_output=True,
        text=True,
    )
    assert py_proc.returncode == 0
