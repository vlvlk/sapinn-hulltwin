"""Download the Shifts power-consumption upload from Zenodo.

``train.csv`` (~109 MB) exceeds GitHub's 100 MB hard limit, so it is NOT
committed: the repo tracks only the licence, the dataset description and the
two dev splits (~7.5 MB total). This script fetches the full Zenodo archive
(record 7057666, CC BY-NC-SA 4.0) and unpacks it, restoring/refreshing
``data/external/power_consumption_upload/``.

Run: ``uv run python scripts/download_shifts.py``

NOTE (2025-09, during MVP development): zenodo.org had a full outage —
every endpoint (records page, API, file download) answered HTTP 504
Gateway Time-out for hours. The download() helper below therefore streams
the body and retries transient 5xx with exponential backoff; if the outage
returns, wait it out or restore the directory from a local backup archive.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ZENODO_URL = "https://zenodo.org/records/7684813/files/power_consumption_upload.zip?download=1"
TARGET_DIR = Path("data/external/power_consumption_upload")

#: Zenodo's gateway (502/503/504) is notoriously flaky for large files:
#: stream the body with a UA header and retry transient failures.
CHUNK_SIZE = 1 << 20
RETRIES = 5
TIMEOUT_S = 60.0
_USER_AGENT = "hulltwin/0.1 (+dataset fetch: Shifts power consumption, Zenodo 7057666)"


def _stream_once(url: str, dest: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})  # noqa: S310
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as resp, dest.open("wb") as out:  # noqa: S310
        while chunk := resp.read(CHUNK_SIZE):
            out.write(chunk)


def download(url: str, dest: Path) -> None:
    print(f"downloading {url} -> {dest} ...")
    for attempt in range(1, RETRIES + 1):
        try:
            _stream_once(url, dest)
            return
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            if attempt == RETRIES:
                raise
            wait = min(2.0**attempt * 5, 60.0)  # 10, 20, 40, 60, 60 s
            print(f"  attempt {attempt}/{RETRIES} failed ({exc}); retrying in {wait:.0f}s ...")
            time.sleep(wait)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Shifts power-consumption (Zenodo 7057666)")
    parser.add_argument(
        "--target",
        type=Path,
        default=TARGET_DIR,
        help="destination directory (default: %(default)s)",
    )
    args = parser.parse_args()

    if (args.target / "synthetic_data" / "train.csv").exists():
        print(f"train.csv already present under {args.target} — nothing to do")
        return

    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "power_consumption_upload.zip"
        download(ZENODO_URL, archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(tmp)  # noqa: S202 (archive from pinned Zenodo record)

        extracted = next(
            (p for p in Path(tmp).iterdir() if p.is_dir() and p.name == "power_consumption_upload"),
            None,
        )
        if extracted is None:  # zip may unpack without a wrapper directory
            extracted = Path(tmp)
        args.target.parent.mkdir(parents=True, exist_ok=True)
        if args.target.exists():
            shutil.rmtree(args.target)
        shutil.copytree(extracted, args.target)

    print(f"Shifts data unpacked to {args.target}")


if __name__ == "__main__":
    main()
