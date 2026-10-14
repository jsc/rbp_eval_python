"""Port of rbp_util/reltrans.c: translate run files into relevance
listings.

For each run, creates a subdirectory under the output dir named after the
run's runid; for each query in that run, writes a file (named after the
qid) listing one relevance value per ranked document ("NA" for unjudged).

Preserved quirks from the original (not fixed, since they're relatively
harmless -- wrong exit code / noisy-but-non-fatal warnings, not wrong
*data*):
  * Unreadable qrels/run files only print a warning during the upfront
    check; they don't set the error flag that would abort before
    attempting (and then properly failing) to open them.
  * The process always exits 0, even if translation fails partway
    through -- the original's `do_trans()` return value is never checked
    by main().

One bug that IS fixed rather than preserved: in the original, if the
qrels file can't be opened, `do_trans()` jumps to its cleanup label
before `qrels` is ever assigned, then calls `qrels_delete(&qrels)` on
that still-NULL pointer -- a guaranteed NULL-pointer dereference.
Verified: `reltrans -d out nonexistent.qrels some.run` segfaults
(exit code 139) on the compiled C binary. This port simply returns
cleanly in that case, same as any other reported error.
"""

from __future__ import annotations

import argparse
import errno
import os
import sys

from ..qrels import QrelsFormatError, load_qrels
from ..ranking import DEFAULT_ORDERING
from ..runs import RunFormatError, load_run
from ..warn import warn

USAGE = "%s [-d <output-dir>] <qrels> <run> ...\n"


def _writeable_dir_error(path: str) -> str | None:
    """Returns None if path is a writeable directory, else a
    strerror-style message describing why not."""
    try:
        st_isdir = os.path.isdir(path)
    except OSError as e:
        return e.strerror
    if not os.path.exists(path):
        return os.strerror(errno.ENOENT)
    if not st_isdir:
        return os.strerror(errno.EINVAL)
    if not os.access(path, os.W_OK):
        return os.strerror(errno.EACCES)
    return None


def _readable_file_error(path: str) -> str | None:
    """Returns None if path is a readable regular file, else a
    strerror-style message describing why not."""
    if not os.path.exists(path):
        return os.strerror(errno.ENOENT)
    if os.path.isdir(path):
        return os.strerror(errno.EISDIR)
    if not os.access(path, os.R_OK):
        return os.strerror(errno.EACCES)
    return None


def _trans_query(fp, qdocs, qid_qrels) -> None:
    for ds in qdocs.get_scores(DEFAULT_ORDERING):
        rel = qid_qrels.get_rel(ds.docid)
        if rel < 0.0:
            fp.write("NA\n")
        else:
            fp.write(f"{rel:.2f}\n")


def _trans_run(output_dir: str, qrels, run) -> int:
    sdir = os.path.join(output_dir, run.get_runid())
    try:
        os.mkdir(sdir, 0o777)
    except OSError as e:
        sys.stderr.write(f"Unable to create run output dir '{sdir}': {e.strerror}\n")
        return -1

    for q in range(run.num_qdocs()):
        qdocs = run.get_qdocs_by_index(q)
        qid = qdocs.qid
        qid_qrels = qrels.get_qid_qrels(qid)
        if qid_qrels is None:
            warn(f"unjudged qid '{qid}' in run '{run.get_runid()}'")
            continue
        fname = os.path.join(sdir, qid)
        try:
            with open(fname, "w") as fp:
                _trans_query(fp, qdocs, qid_qrels)
        except OSError as e:
            sys.stderr.write(
                f"Unable to open output file '{fname}' for writing: {e.strerror}\n"
            )
            return -1
    return 0


def _do_trans(output_dir: str, qrels_fname: str, run_fnames: list[str]) -> int:
    try:
        with open(qrels_fname) as fp:
            qrels = load_qrels(fp)
    except OSError as e:
        sys.stderr.write(
            f"Unable to open qrels file '{qrels_fname}' for reading: {e.strerror}\n"
        )
        return -1
    except QrelsFormatError as e:
        sys.stderr.write(f"Error loading qrels file '{qrels_fname}': {e}\n")
        return -1

    for run_fname in run_fnames:
        try:
            with open(run_fname) as fp:
                run = load_run(fp)
        except OSError as e:
            sys.stderr.write(
                f"Unable to open run file '{run_fname}' for reading: {e.strerror}\n"
            )
            return -1
        except RunFormatError as e:
            sys.stderr.write(f"Error loading run file '{run_fname}': {e}\n")
            return -1

        if _trans_run(output_dir, qrels, run) < 0:
            return -1

    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    progname = sys.argv[0] if len(sys.argv) > 0 else "reltrans"

    parser = argparse.ArgumentParser(prog=progname, add_help=False)
    parser.add_argument("-d", dest="output_dir", default=".")
    parser.add_argument("files", nargs="*")

    try:
        args = parser.parse_args(argv)
    except SystemExit:
        sys.stderr.write(USAGE % progname)
        return 1

    error = False
    dir_err = _writeable_dir_error(args.output_dir)
    if dir_err is not None:
        sys.stderr.write(f"{args.output_dir} not a writeable valid directory: {dir_err}\n")
        error = True

    if len(args.files) < 2:
        error = True
    else:
        qrels_fname = args.files[0]
        run_fnames = args.files[1:]
        qrels_err = _readable_file_error(qrels_fname)
        if qrels_err is not None:
            sys.stderr.write(f"Qrels file {qrels_fname} is not readable: {qrels_err}\n")
        for run_fname in run_fnames:
            run_err = _readable_file_error(run_fname)
            if run_err is not None:
                sys.stderr.write(f"Run file {run_fname} is not readable: {run_err}\n")

    if error:
        sys.stderr.write(USAGE % progname)
        return 1

    _do_trans(args.output_dir, qrels_fname, run_fnames)
    return 0


if __name__ == "__main__":
    sys.exit(main())
