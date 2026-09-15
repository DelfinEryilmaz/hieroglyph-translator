"""Download the pretrained YOLOv8-segmentation checkpoint we fine-tune for
glyph detection (see reports/2026-09-15-segmentation-detector-sourcing.md
for where this comes from and why).

No API token needed -- it's a public GitHub release asset.
"""

import argparse
import sys
import urllib.request
from pathlib import Path

CHECKPOINT_URL = (
    "https://github.com/EngAdhamTamer/hieroglyph-detection/releases/download/v1.0.0/best.pt"
)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", default=str(PROJECT_ROOT / "models" / "yolo_seg_pretrained.pt"))
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {CHECKPOINT_URL} -> {dest} ...")
    try:
        urllib.request.urlretrieve(CHECKPOINT_URL, dest)
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        print(f"You can also download it manually from {CHECKPOINT_URL}", file=sys.stderr)
        return 1

    print(f"Done. Saved to {dest} ({dest.stat().st_size} bytes).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
