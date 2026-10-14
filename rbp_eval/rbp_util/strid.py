"""Port of librbp/strid.[ch]: a bidirectional string<->sequential-integer-id
map. Ids are assigned in the order strings are first seen, starting at 0.

The original backs this with a full custom hash table; a dict + list is
the direct Python equivalent, so there is nothing to "port" algorithmically
here.
"""

from __future__ import annotations


class Strid:
    def __init__(self):
        self._ids: dict[str, int] = {}
        self._strs: list[str] = []

    def get_id(self, s: str) -> int:
        """Get the id of a string, assigning a new one if not yet seen."""
        idx = self._ids.get(s)
        if idx is None:
            idx = len(self._strs)
            self._ids[s] = idx
            self._strs.append(s)
        return idx

    def lookup_id(self, s: str) -> int | None:
        """Look up the id of a string. None if not yet assigned (the C
        original returns UINT_MAX; None is the Pythonic equivalent)."""
        return self._ids.get(s)

    def get_str(self, id_: int) -> str:
        return self._strs[id_]

    def num_ids(self) -> int:
        return len(self._strs)
