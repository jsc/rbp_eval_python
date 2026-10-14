"""Differential test: runs the Python CLI against the actual compiled C
rbp_eval binary (if available) over every .spec fixture's flag
combination, and checks output matches byte-for-byte apart from the
`# command:` line (which necessarily differs -- it echoes each program's
own invocation)."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "rbp_eval" / "test" / "data"
C_BINARY = Path(__file__).resolve().parents[2] / "rbp_eval" / "rbp_eval"

_CTIME_RE = re.compile(r"^# \w{3} \w{3} .* \d{4}$")


def _spec_files() -> list[Path]:
    if not FIXTURE_DIR.is_dir():
        return []
    return sorted(FIXTURE_DIR.glob("*.spec"))


def _strip_nondeterministic_lines(output: str) -> str:
    # "# command:" necessarily differs (each program echoes its own
    # invocation); the ctime line can legitimately differ by a second
    # between the two subprocess invocations.
    return "\n".join(
        line
        for line in output.splitlines()
        if not line.startswith("# command:") and not _CTIME_RE.match(line)
    )


@pytest.mark.skipif(not C_BINARY.is_file(), reason="C rbp_eval binary not built")
@pytest.mark.parametrize("spec_path", _spec_files(), ids=lambda p: p.name)
def test_cli_matches_c_binary(spec_path: Path):
    header = next(
        line
        for line in spec_path.read_text().splitlines()
        if not line.startswith("#")
    )
    args = header.split()

    c_proc = subprocess.run(
        [str(C_BINARY), "-q", *args],
        cwd=spec_path.parent,
        capture_output=True,
        text=True,
    )
    py_proc = subprocess.run(
        [sys.executable, "-m", "rbp_eval.cli.rbp_eval", "-q", *args],
        cwd=spec_path.parent,
        capture_output=True,
        text=True,
    )

    assert _strip_nondeterministic_lines(c_proc.stdout) == _strip_nondeterministic_lines(
        py_proc.stdout
    )
    assert c_proc.returncode == 0
    assert py_proc.returncode == 0
