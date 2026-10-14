"""Differential test: dcg_eval CLI against the compiled C++ binary."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "rbp_eval" / "test" / "data"
CXX_BINARY = Path(__file__).resolve().parents[2] / "dcg_eval" / "dcg_eval"

ARG_COMBOS = [
    ["basic.qrels", "noties.run"],
    ["-R", "-b", "10", "-N", "-D", "basic.qrels", "noties.run"],
    ["-m", "frac.qrels", "noties.run"],
    ["-M", "-r", "5", "basic.qrels", "ties.run"],
    ["-N", "-D", "mult.qrels", "noties.run"],
]


@pytest.mark.skipif(not CXX_BINARY.is_file(), reason="C++ dcg_eval binary not built")
@pytest.mark.parametrize("args", ARG_COMBOS, ids=lambda a: " ".join(a))
def test_dcg_cli_matches_cxx_binary(args):
    cxx_proc = subprocess.run(
        [str(CXX_BINARY), *args], cwd=FIXTURE_DIR, capture_output=True, text=True
    )
    py_proc = subprocess.run(
        [sys.executable, "-m", "rbp_eval.cli.dcg_eval", *args],
        cwd=FIXTURE_DIR,
        capture_output=True,
        text=True,
    )
    assert cxx_proc.stdout == py_proc.stdout
    assert cxx_proc.stderr == py_proc.stderr
    assert cxx_proc.returncode == py_proc.returncode
