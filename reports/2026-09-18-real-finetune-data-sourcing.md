# Real-photo fine-tune data: what's committed and why

**Committed:** `data/real_eval_photos/train/` (83 image+label pairs) and
`data/real_eval_photos/valid/` (7 pairs), from the Roboflow
"egyptian-hieroglyphs" dataset
(https://universe.roboflow.com/custom-yolov8-ljpde/egyptian-hieroglyphs),
already referenced in
`reports/2026-09-15-segmentation-detector-sourcing.md` and downloaded per
`notebooks/05_evaluate_segmenter.ipynb` section 3's manual step.

## What this data actually is (measured)
Measured directly from the committed label files
(`data/real_eval_photos/{train,valid}/labels/*.txt`, YOLO
`<class> cx cy w h`, normalized, on 640x640 images):

| Measurement | Value |
| --- | --- |
| Images / label files | 90 (83 train + 7 valid) |
| Boxes total | 90 — **exactly one box per image** |
| Median box area | **39.5% of the frame** (train 41.1%, valid 20.1%) |
| Boxes covering >50% of the frame | 36 of 90 (**40%**) |
| Box area range | 12.8% – 99.2% |

These are **single-glyph crops, not multi-sign scene photos** — one sign
per image, filling a large fraction of the frame, many visibly upscaled
from small sources (blocky) and carrying black letterboxing corners from
Roboflow's rotation augmentation.

That matters for how the fine-tune stage should be understood: it exposes
the model to **real photographic texture and noise on individual signs**,
*not* to **realistic real-world sign density or layout**. Dense, realistic
layout is what the dense column composites and hard-negative composites
(`src/hieroglyph/segmentation/synthesize.py`) are for. Because fine-tuning
15 epochs on one-large-sign-per-frame data could plausibly cost
dense-text performance, `notebooks/04_train_segmenter.ipynb` step 9
deliberately downloads *both* the synthetic-only and the real-fine-tuned
checkpoint and asks the user to evaluate both (especially
`notebooks/05_evaluate_segmenter.ipynb` Section 5's real-papyrus
qualitative check) before choosing one as `models/yolo_seg.pt`.

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
