"""Download the hieroglyph classification dataset from Kaggle into data/raw/.

Requires a Kaggle API token at ~/.kaggle/kaggle.json (or KAGGLE_USERNAME /
KAGGLE_KEY env vars). Get a token from https://www.kaggle.com/settings ->
"Create New Token", which downloads kaggle.json for you to place at:
  Windows: C:\\Users\\<you>\\.kaggle\\kaggle.json

If you'd rather not set up API access, download the dataset manually from
https://www.kaggle.com/datasets/ayatollahelkolally/hieroglyphs-dataset
and unzip it into data/raw/ yourself, then skip this script.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_DATASET = "ayatollahelkolally/hieroglyphs-dataset"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def kaggle_credentials_available() -> bool:
    token_path = Path.home() / ".kaggle" / "kaggle.json"
    return token_path.exists()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dest", default=str(PROJECT_ROOT / "data" / "raw"))
    args = parser.parse_args()

    if shutil.which("kaggle") is None:
        print("The 'kaggle' CLI isn't on PATH. Install it with:", file=sys.stderr)
        print("  pip install kaggle", file=sys.stderr)
        return 1

    if not kaggle_credentials_available():
        print(
            "No Kaggle API token found at ~/.kaggle/kaggle.json.\n"
            "Get one from https://www.kaggle.com/settings -> 'Create New Token',\n"
            "then place the downloaded kaggle.json there.\n"
            "(Or download the dataset manually and unzip into data/raw/ instead.)",
            file=sys.stderr,
        )
        return 1

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.dataset} into {dest} ...")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", args.dataset, "-p", str(dest), "--unzip"],
    )
    if result.returncode != 0:
        print("Download failed.", file=sys.stderr)
        return result.returncode

    class_dirs = [p for p in dest.iterdir() if p.is_dir()]
    image_count = sum(1 for _ in dest.rglob("*") if _.is_file())
    print(f"Done. {len(class_dirs)} class folders, {image_count} files in {dest}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
