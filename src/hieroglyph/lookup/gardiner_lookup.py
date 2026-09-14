"""Sign -> meaning lookup, backed by data_tables/gardiner_signs.csv.

The CSV was hand-compiled from Gardiner's 1957 sign list (Egyptian Grammar,
3rd ed.), cross-checked against exactly our 171 trained classes. A few
notes on its content, since this data is inherently imperfect:
- `transliteration` is genuinely empty for many signs -- determinatives
  (signs that convey meaning but no sound) don't have one, and we didn't
  fabricate one where a source didn't give it.
- 5 codes (D156, M195, O11, P13, P98) either weren't in Gardiner's original
  list at all, or needed a second source (O11 was resolved via the Unicode
  17.0 standard; the other 4 remain unresolved with an explicit note in
  their `notes` field, rather than a guessed meaning).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CSV_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data_tables" / "gardiner_signs.csv"


@dataclass
class GardinerEntry:
    gardiner_code: str
    category: str
    transliteration: str  # "" if none exists/is known
    gloss: str  # "" only for the handful of genuinely unresolved codes
    notes: str  # "" unless something about this entry needs flagging


class GardinerLookup:
    def __init__(self, csv_path: Path = DEFAULT_CSV_PATH) -> None:
        self._entries: dict[str, GardinerEntry] = {}
        with open(csv_path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                entry = GardinerEntry(
                    gardiner_code=row["gardiner_code"],
                    category=row["category"],
                    transliteration=row["transliteration"],
                    gloss=row["gloss"],
                    notes=row["notes"],
                )
                self._entries[entry.gardiner_code] = entry

    def __len__(self) -> int:
        return len(self._entries)

    def lookup(self, gardiner_code: str) -> GardinerEntry:
        """Look up a sign by its Gardiner code.

        Raises KeyError for a code truly not in the table (e.g. a typo or a
        model output that doesn't correspond to any trained class) -- this
        is meant to surface as a loud failure during development, not be
        silently swallowed, since it would indicate a real mismatch between
        the model's class list and this lookup table.
        """
        if gardiner_code not in self._entries:
            raise KeyError(f"No lookup entry for Gardiner code {gardiner_code!r}")
        return self._entries[gardiner_code]
