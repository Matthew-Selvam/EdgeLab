"""Acquire the ten pinned public-domain results JSONs; never download odds.

Run once from the repository root: python scripts/fetch_openfootball.py
Existing matching files are reused; differing local files are never overwritten.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    manifest = json.loads((ROOT / "docs/OPENFOOTBALL_SOURCES.json").read_text())
    destination = ROOT / "data/raw/openfootball"
    pending = []
    for item in manifest["files"]:
        path = destination / f"{item['season']}.json"
        if path.exists():
            data = path.read_bytes()
        else:
            with urlopen(item["url"], timeout=60) as response:
                data = response.read()
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError(f"Source hash mismatch: {item['season']}; no files written")
        if not path.exists():
            pending.append((path, data))
    destination.mkdir(parents=True, exist_ok=True)
    for path, data in pending:
        # Exclusive creation prevents replacing an input that appeared meanwhile.
        with path.open("xb") as output:
            output.write(data)
    print(f"Verified {len(manifest['files'])} pinned OpenFootball files; "
          f"downloaded {len(pending)} missing files. No odds accessed.")


if __name__ == "__main__":
    main()
