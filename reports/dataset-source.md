# Dataset source decision

**Chosen:** Kaggle `ayatollahelkolally/hieroglyphs-dataset`
https://www.kaggle.com/datasets/ayatollahelkolally/hieroglyphs-dataset

## Why
This matches the canonical academic dataset from:
> M. Franken and J. van Gemert, "Automatic Egyptian Hieroglyph Recognition by
> Retrieving Images as Texts," ACM Multimedia 2013.

- 4,032 grayscale images
- 171 unique hieroglyphic sign classes, labeled via Gardiner Sign List codes
- Source: photographs taken inside the Pyramid of Unas
- Fixed resolution 75×50 px, single glyph per image

Other Kaggle mirrors found during research (`waleedumer/egyptian-hieroglyphics-datasets`,
`alexandrepetit881234/egyptian-hieroglyphs`, `ahmedsamir1598/glyphdataset`) are
likely re-uploads of the same underlying data but weren't independently
verified — going with the one whose description explicitly matches the
original paper's spec.

## Related work noted during research
- arXiv 2512.03817, "HieroGlyphTranslator: Automatic Recognition and
  Translation of Egyptian Hieroglyphs to English" — recent paper (Dec 2025)
  doing a similar recognition+translation pipeline. Uses their own
  stock-photo-sourced dataset (Adobe Stock / Alamy), not the Pyramid of Unas
  set. Code: https://github.com/mariamsk10/HieroGlyphTranslator — worth a
  look later for implementation ideas, not a dataset source for us.
- arXiv 2512.24197, "The OCR-PT-CT Project: Semi-Automatic Recognition of
  Ancient Egyptian Hieroglyphs Based on Metric Learning" — noted, not
  reviewed in depth.

## Open item
Still need to find/confirm a source for the transliteration/gloss side of
the lookup table (Gardiner sign list meanings) — separate from the image
dataset. To be researched in Phase 9.
