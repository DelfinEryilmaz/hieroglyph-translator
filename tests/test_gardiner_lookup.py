from pathlib import Path

import pytest

from hieroglyph.lookup.gardiner_lookup import GardinerLookup

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "gardiner_signs_fixture.csv"


def test_lookup_returns_correct_entry():
    lookup = GardinerLookup(FIXTURE_PATH)

    entry = lookup.lookup("D21")

    assert entry.gloss == "Mouth"
    assert entry.transliteration == "r, jw"
    assert entry.category == "D. Parts Of The Human Body"


def test_lookup_handles_missing_transliteration_as_empty_not_none():
    lookup = GardinerLookup(FIXTURE_PATH)

    entry = lookup.lookup("A55")

    assert entry.transliteration == ""
    assert entry.gloss == "Mummy on bed"


def test_lookup_unresolved_entry_has_empty_gloss_and_a_note():
    lookup = GardinerLookup(FIXTURE_PATH)

    entry = lookup.lookup("D156")

    assert entry.gloss == ""
    assert entry.notes != ""


def test_lookup_unknown_code_raises_keyerror():
    lookup = GardinerLookup(FIXTURE_PATH)

    with pytest.raises(KeyError):
        lookup.lookup("NOT_A_REAL_CODE")


def test_full_table_loads_and_covers_every_trained_class():
    # Uses the real project CSV (not the fixture) -- this is the "does our
    # lookup table actually cover all 171 trained classes" integration check.
    lookup = GardinerLookup()
    assert len(lookup) == 171
