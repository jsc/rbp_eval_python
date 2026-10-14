"""Warning plumbing, mirroring librbp/error.c's warning()/warning_set_stream().

Uses the stdlib warnings module instead of a stream pointer: callers who
want librbp's "-W" (no-warnings) behaviour call
``warnings.simplefilter("ignore", RbpEvalWarning)``.
"""

from __future__ import annotations

import warnings


class RbpEvalWarning(UserWarning):
    pass


def warn(msg: str) -> None:
    warnings.warn(msg, RbpEvalWarning, stacklevel=2)
