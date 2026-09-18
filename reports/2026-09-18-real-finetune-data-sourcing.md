# Real-photo fine-tune data: what's committed and why

**Committed:** `data/real_eval_photos/train/` (83 image+label pairs) and
`data/real_eval_photos/valid/` (7 pairs), from the Roboflow
"egyptian-hieroglyphs" dataset
(https://universe.roboflow.com/custom-yolov8-ljpde/egyptian-hieroglyphs),
already referenced in
`reports/2026-09-15-segmentation-detector-sourcing.md` and downloaded per
`notebooks/05_evaluate_segmenter.ipynb` section 3's manual step.

## Why committed now (reversing Phase 12's original decision)
Phase 12's design spec
(`docs/superpowers/specs/2026-09-15-glyph-segmentation-detector-design.md`)
judged this set "too small to train on" and used it only for a qualitative
box-count eyeball check. That's the right call for training *from
scratch*, but not for a short **fine-tuning** stage after a large
synthetic pretrain -- see
`docs/superpowers/specs/2026-09-18-real-data-finetune-and-hard-negatives-design.md`
for the reasoning. At ~2.3MB total, committing these two splits is small
enough to keep in the repo, which also means `notebooks/04_train_segmenter.ipynb`'s
existing `git clone` step on Colab picks them up automatically -- no new
manual upload cell needed.

## What's NOT committed
- `data/real_eval_photos/test/` -- an inconsistent leftover from the
  original export (loose label files directly under `test/` plus a
  near-empty `images/`/`labels/` subdirectory pair, likely a partial
  re-export). Not used by anything; left as a future cleanup.
- `data/real_eval_photos/papyrus/` -- the Papyrus of Ani qualitative-eval
  photo, re-downloadable via `scripts/download_papyrus_eval_photo.py`; no
  need to duplicate it into git history.

## License
CC BY 4.0, per the Roboflow dataset page linked above --
attribution: "egyptian-hieroglyphs Dataset" by custom-yolov8-ljpde, made
available on Roboflow Universe under CC BY 4.0.
