"""
Fetch locomo10.json into data/locomo/.

The server may have no outbound internet. If the download fails this prints exact manual
instructions rather than dying obscurely -- copying the file in by hand is a perfectly
good outcome and the rest of the harness does not care how it got there.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

from .paths import LOCOMO_JSON, ensure_dirs

# Primary source: the official dataset repo.
SOURCES = [
    "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json",
    "https://media.githubusercontent.com/media/snap-research/locomo/main/data/locomo10.json",
]

MANUAL = f"""
Could not download the dataset automatically.

Fetch it manually (on any machine with internet) and place it at:
    {LOCOMO_JSON}

  git clone https://github.com/snap-research/locomo.git
  cp locomo/data/locomo10.json {LOCOMO_JSON}

or:
  curl -L -o {LOCOMO_JSON} \\
    {SOURCES[0]}

Then verify:
  python -m benchmarks.dataset --inspect
""".strip()


def download(force: bool = False) -> bool:
    ensure_dirs()

    if LOCOMO_JSON.exists() and not force:
        print(f"already present: {LOCOMO_JSON} ({LOCOMO_JSON.stat().st_size / 1e6:.1f} MB)")
        return True

    for url in SOURCES:
        print(f"trying {url}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "memora-benchmark"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = resp.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"  failed: {exc}")
            continue

        # Git-LFS pointer files are small and start with "version https://git-lfs".
        if payload[:40].lstrip().startswith(b"version https://git-lfs"):
            print("  got a git-lfs pointer, not the payload")
            continue

        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            print(f"  not valid JSON: {exc}")
            continue

        LOCOMO_JSON.write_bytes(payload)
        n = len(parsed) if isinstance(parsed, list) else "?"
        print(f"saved {LOCOMO_JSON} ({len(payload) / 1e6:.1f} MB, {n} samples)")
        return True

    print(MANUAL, file=sys.stderr)
    return False


if __name__ == "__main__":
    force = "--force" in sys.argv
    sys.exit(0 if download(force=force) else 1)
