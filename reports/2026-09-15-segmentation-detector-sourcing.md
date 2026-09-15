# Segmentation detector: pretrained checkpoint source

**Chosen starting point:** `best.pt` from
https://github.com/EngAdhamTamer/hieroglyph-detection
(release: https://github.com/EngAdhamTamer/hieroglyph-detection/releases/tag/v1.0.0)

## Why
Research during Phase 12 planning found no bulk downloadable real-photo
dataset with glyph bounding-box/segmentation annotations (the closest
candidate, this same repo's own 235,436-instance training set, is
explicitly **not** distributed — confirmed by reading its
`dataset/data.yaml`, which says the images/labels aren't included "due to
size constraints"). Its trained checkpoint *is* distributed, though, and:

- MIT licensed
- YOLOv8-segmentation, single class ("hieroglyph") — matches our need
  exactly, since sign classification is already a separate downstream
  stage (the ResNet18 classifier) in our pipeline
- Reports 70.9% mAP@0.5 on real archaeological photos (temple walls,
  obelisks, papyrus, stelae)
- 6.8MB, downloaded via `scripts/download_pretrained_yolo.py`

We fine-tune this checkpoint on our own synthesized composite images
(`hieroglyph.segmentation.synthesize`, built from `data/raw/` single-glyph
crops) rather than training from scratch, so the model starts from weights
already exposed to real photos instead of only ever seeing our synthetic
compositing style.

## Other candidates considered, not used
- [Roboflow "egyptian-hieroglyphs"](https://universe.roboflow.com/custom-yolov8-ljpde/egyptian-hieroglyphs)
  — real photos, 40 images, 19 Gardiner-coded classes, CC BY 4.0. Too
  small to train on; used instead as a real-photo qualitative eval set in
  `notebooks/05_evaluate_segmenter.ipynb`.
- Roboflow "Hieroglyphic_3" (nadine-mostafa) — found during search, not
  independently verified (couldn't confirm image count/license without a
  Roboflow account); not pursued.

## License note
`best.pt`'s MIT license permits this reuse (including fine-tuning and
redistributing derivative weights). We keep this file as the paper trail
for that provenance, matching this repo's convention (see
`reports/dataset-source.md`, `reports/gardiner-lookup-table-sourcing.md`).
