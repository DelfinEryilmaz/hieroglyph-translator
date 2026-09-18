# Real-papyrus qualitative eval photo: source

**Chosen:** "Papyrus of Ani BM Sheet 12.jpg" on Wikimedia Commons
https://commons.wikimedia.org/wiki/File:Papyrus_of_Ani_BM_Sheet_12.jpg
(direct file: https://upload.wikimedia.org/wikipedia/commons/c/c9/Papyrus_of_Ani_BM_Sheet_12.jpg,
downloaded via `scripts/download_papyrus_eval_photo.py`)

## Why
`notebooks/05_evaluate_segmenter.ipynb`'s existing qualitative check
(Section 3) uses the Roboflow "egyptian-hieroglyphs" set -- real photos,
but none of them are actual papyrus columns of dense cursive text, which is
exactly the case that exposed the sim2real gap this effort addresses (see
`docs/superpowers/specs/2026-09-17-dense-text-detection-design.md`). This
photo is one sheet of the real Papyrus of Ani (Book of the Dead, ~1250 BCE,
British Museum EA10470), a photographic reproduction of a dense
hieroglyphic-text column -- exactly the layout `synthesize_composite`'s
sparse scatter never trained the detector on.

- 2,414x1,498px -- large enough to actually exercise
  `TiledYoloSegmenter`'s tiling (a single 640x640 tile could not cover it).
- Public domain: Wikimedia Commons' file-info page states it is "public
  domain in its country of origin and other countries and areas where the
  copyright term is the author's life plus 100 years or fewer" (a ~1250 BCE
  papyrus, photographed for the British Museum's own reproduction of a
  reproduction already over a century old).

## License note
Public domain, per Wikimedia Commons' license tag on the file page linked
above. We keep this file as the paper trail for that provenance, matching
this repo's convention (see `reports/dataset-source.md`,
`reports/2026-09-15-segmentation-detector-sourcing.md`).
