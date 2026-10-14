"""Port of librbp/depth.[ch] and librbp/persist.[ch]: '<v1>,<v2>,...' spec
parsing for the -d and -p command line options.

The fixed-size DEPTH_MAX_NUM/PERSIST_MAX_NUM element-count limits from the
C arrays are not reproduced here -- they were a C array-sizing artifact,
not part of the actual spec grammar.
"""

from __future__ import annotations

import math

DEPTH_FULL = math.inf


class DepthSpecError(ValueError):
    pass


class PersistSpecError(ValueError):
    pass


def _tokenize_spec(spec: str) -> list[str]:
    if spec == "":
        return []
    tokens = spec.split(",")
    if tokens[-1] == "":
        # A single trailing comma is allowed and simply ends the spec.
        tokens.pop()
    return tokens


def parse_depth(spec: str) -> list[float]:
    """Parse a depth spec '<d1>,<d2>,...'.

    A depth of 0 means "unlimited" and is represented as DEPTH_FULL
    (math.inf). Depths must be in strictly ascending order.
    """
    depths: list[float] = []
    for i, tok in enumerate(_tokenize_spec(spec)):
        try:
            dval = int(tok)
        except ValueError:
            raise DepthSpecError(
                f"invalid format for element {i} of depth specification "
                f"'{spec}'.  Must be in form '<d1>,<d2>...'"
            ) from None
        if dval < 0:
            raise DepthSpecError(
                f"element {i} of depth specification '{spec}' with value "
                f"{dval} not in range [1, LONG_MAX)"
            )
        d = DEPTH_FULL if dval == 0 else float(dval)
        if depths and d <= depths[-1]:
            raise DepthSpecError(
                f"depths in specification '{spec}' not in strictly "
                f"ascending order"
            )
        depths.append(d)
    return depths


def parse_persist(spec: str) -> list[float]:
    """Parse a persistence spec '<p1>,<p2>,...', each in [0.0, 1.0]."""
    persists: list[float] = []
    for i, tok in enumerate(_tokenize_spec(spec)):
        try:
            val = float(tok)
        except ValueError:
            raise PersistSpecError(
                f"invalid format for element {i} of persistence "
                f"specification '{spec}'.  Must be in form '<p1>,<p2>...'"
            ) from None
        if val < 0.0 or val > 1.0:
            raise PersistSpecError(
                f"element {i} of persistence specification '{spec}' with "
                f"value {val} not in range [0.0, 1.0]"
            )
        persists.append(val)
    return persists
