"""
Downloads the 72 official 30-second iTunes preview clips analysed in the study.

Each recording is identified by its iTunes track ID in data/song_matches.csv. This script asks the
iTunes Lookup API (US store) for each track's preview URL and saves the clip to the path given in
the "file" column (data/previews/...m4a), where roughness_analysis.py expects it.

    python download_previews.py            # all 72 clips
    python download_previews.py --limit 3  # just the first three (for a quick test)

Uses only the Python standard library. The clips are copyrighted by their owners and are provided
by Apple for previewing; they are not part of this repository and should not be redistributed.
Apple may change or withdraw a preview over time; any track that cannot be found is reported.
"""
import csv
import json
import sys
import time
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
LOOKUP = "https://itunes.apple.com/lookup?country=US&id="


def fetch(url, tries=3):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


def main():
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    rows = list(csv.DictReader(open(DATA / "song_matches.csv", newline="", encoding="utf-8")))[:limit]
    ids = [r["trackId"] for r in rows]
    found = {}
    for i in range(0, len(ids), 50):  # the Lookup API accepts many IDs per request
        res = json.loads(fetch(LOOKUP + ",".join(ids[i:i + 50])))
        found.update({str(x["trackId"]): x.get("previewUrl") for x in res.get("results", [])})
    missing = []
    for r in rows:
        url = found.get(r["trackId"])
        out = DATA / r["file"]
        if not url:
            missing.append(f'{r["no"]} {r["title"]} ({r["trackId"]})')
            continue
        if out.exists() and out.stat().st_size > 0:
            print(f'{r["no"]} already present: {out.name}')
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(fetch(url))
        print(f'{r["no"]} downloaded: {out.name}')
    print(f"\n{len(rows) - len(missing)} of {len(rows)} previews available in {DATA / 'previews'}")
    if missing:
        print("Not available from the iTunes Lookup API:\n  " + "\n  ".join(missing))


if __name__ == "__main__":
    main()
