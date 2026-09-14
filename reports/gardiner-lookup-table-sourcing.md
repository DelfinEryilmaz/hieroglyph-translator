# Gardiner sign lookup table — sourcing notes

`data_tables/gardiner_signs.csv` was compiled for exactly our 171 trained
classes (checked against `data/raw`'s folder names), from:

- **Primary source:** Alan Gardiner's *Egyptian Grammar* (3rd ed., 1957)
  sign list, via a machine-readable XLSX transcription:
  https://github.com/omnika-datastore/alan-gardiner-list-of-hieroglyphic-signs
  — covered 166/171 codes directly (category + gloss from its "Description"
  column).
- **Transliteration** (82/171 codes have one — many signs, especially
  determinatives, genuinely have none) came from Wikipedia's
  ["List of Egyptian hieroglyphs"](https://en.wikipedia.org/wiki/List_of_Egyptian_hieroglyphs),
  fetched in batches (the page is too long for one fetch to cover in full —
  only reached categories A through N before truncating).
- **O11** ("palace", *ꜥḥ*) isn't in Gardiner's original 1957 list at all —
  resolved via the Unicode 17.0 Egyptian Hieroglyphs code chart, where it
  exists as codepoint name "EGYPTIAN HIEROGLYPH O011".
- **D156, M195, P13, P98** — not found in Gardiner (1957), Unicode 17.0, or
  Wikipedia after real research effort. These exceed both Gardiner's
  original numbering and current Unicode's coverage for their categories
  (Unicode's Egyptian Hieroglyphs block tops out at D67, M44, P11). They're
  likely dataset-specific extended codes (possibly from JSesh/MdC's own
  extended catalog, unconfirmed). Left with empty gloss/transliteration and
  an explicit note in the CSV rather than a fabricated meaning.

## A real bug found and fixed during sourcing
The XLSX source also contains shape-based cross-reference appendix tables
("Tall Narrow Signs", "Low Narrow Signs", "Low Broad Signs" — from the back
of Gardiner's book) that **reuse Gardiner codes to mean something entirely
different** (pointers to visually-similar signs, not the sign itself). A
naive `{code: row for row in all_rows}` dict silently let these appendix
rows overwrite the correct main-list entries for any code that happened to
also appear there — caught via D21 (correct: "Mouth") getting overwritten
with garbage ("T30") from the appendix. Fixed by excluding those 3
appendix categories entirely during parsing. Also fixed: the "X. Bread,
Cakes, Etc." category header row was missed during parsing, mislabeling
X1/X6/X8 under category "W" (their actual gloss text was correct, only the
category field was wrong).

## If picking this up later
To improve transliteration coverage further (currently blank for most O–Z
categories, ~89 codes), Wikipedia's page would need fetching in more
targeted chunks, or a different structured source found — the omnika XLSX's
"Details" column has transliteration mixed into free-text usage notes for
every sign, which could be parsed with more effort than we spent here.
