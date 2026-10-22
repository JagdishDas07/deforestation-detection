"""Pull Sentinel-2 L2A for the study dates, clean them, and export GeoTIFFs.

For each date window in src/data/regions.py:
  filter L2A by cloud % -> mask clouds/shadows via SCL -> median composite
  -> export blue/green/red/NIR/SWIR1/SWIR2 to Google Drive.

Exports run as Earth Engine batch tasks (server-side); this script only queues
them and prints their IDs. Monitor at https://code.earthengine.google.com/tasks
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ee  # noqa: E402
from data.earth_engine import (  # noqa: E402
    init, verify, get_cloud_free_composite, export_bands, DEFAULT_BANDS,
)
from data.regions import REGION, DATE1, DATE2, CLOUD_PCT  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folder", default="deforestation",
                   help="Google Drive folder to export into")
    p.add_argument("--scale", type=int, default=10, help="export resolution (m)")
    p.add_argument("--project", default=None, help="Earth Engine Cloud project id")
    p.add_argument("--wait", action="store_true", help="poll task status until done")
    args = p.parse_args()

    init(project=args.project)
    if not verify():
        raise SystemExit("Earth Engine not reachable — run `earthengine authenticate`")

    region = ee.Geometry.Rectangle(REGION["bounds"])
    print(f"Region : {REGION['name']}")
    print(f"Bounds : {REGION['bounds']}  bands: {list(DEFAULT_BANDS)}")

    tasks = []
    for window, tag in ((DATE1, "2020"), (DATE2, "2022")):
        composite = get_cloud_free_composite(region, *window, CLOUD_PCT)
        desc = f"s2_colniza_{tag}"
        task = export_bands(composite, region, desc, folder=args.folder,
                            scale=args.scale)
        tasks.append((desc, task))
        print(f"queued {desc}  ({window[0]} .. {window[1]})  id={task.id}")

    print(f"\n{len(tasks)} export task(s) queued to Drive folder '{args.folder}'.")
    if args.wait:
        import time
        while any(t.status().get("state") in ("READY", "RUNNING") for _, t in tasks):
            for desc, t in tasks:
                print(f"  {desc}: {t.status().get('state')}")
            time.sleep(30)


if __name__ == "__main__":
    main()
