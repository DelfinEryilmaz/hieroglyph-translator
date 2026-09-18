"""Download a real, densely-packed papyrus photo for qualitative
segmentation eval (see reports/2026-09-17-papyrus-eval-photo-sourcing.md
for what this image is and why).

No API token needed -- it's a public-domain file hosted on Wikimedia
Commons. Wikimedia's servers reject requests with no User-Agent header
(HTTP 403) -- plain urllib.request.urlretrieve sends none, so this uses an
explicit Request with a descriptive User-Agent instead, per Wikimedia's
own request policy (https://meta.wikimedia.org/wiki/User-Agent_policy).
"""

import argparse
import sys
import urllib.request
from pathlib import Path

IMAGE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/c/c9/Papyrus_of_Ani_BM_Sheet_12.jpg"
)
USER_AGENT = (
    "hieroglyph-translator-research-script/1.0 "
    "(https://github.com/DelfinEryilmaz/hieroglyph-translator)"
)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        default=str(PROJECT_ROOT / "data" / "real_eval_photos" / "papyrus" / "papyrus_of_ani_bm_sheet_12.jpg"),
    )
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {IMAGE_URL} -> {dest} ...")
    try:
        request = urllib.request.Request(IMAGE_URL, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request) as response, open(dest, "wb") as f:
            f.write(response.read())
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        print(f"You can also download it manually from {IMAGE_URL}", file=sys.stderr)
        return 1

    print(f"Done. Saved to {dest} ({dest.stat().st_size} bytes).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
