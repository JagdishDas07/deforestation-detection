"""Build the NDVI threshold baseline scored against Dynamic World forest labels.

For the study region/window: mask the composite, compute NDVI from red/NIR,
threshold it, export the Dynamic World 'trees' mask, spot-check the two against
the imagery, and record the baseline IoU the U-Net must beat.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ee  # noqa: E402
from data.earth_engine import init, verify, get_cloud_free_composite, export_bands  # noqa: E402
from data.labels import get_trees_mask  # noqa: E402
from data.regions import REGION, DATE1, CLOUD_PCT  # noqa: E402
from evaluate import append_metrics  # noqa: E402


def ndvi(composite):
    return composite.normalizedDifference(["B8", "B4"]).rename("NDVI")


def forest_mask(composite, threshold=0.6):
    return ndvi(composite).gt(threshold).rename("forest")


def area_ha(mask, region, scale=10):
    return (mask.rename("m").multiply(ee.Image.pixelArea()).divide(1e4)
            .reduceRegion(ee.Reducer.sum(), region, scale, maxPixels=1e9).get("m").getInfo())


def iou(pred, truth, region, scale=10):
    """IoU between two binary Earth Engine masks, via area intersection/union."""
    inter = area_ha(pred.And(truth), region, scale) or 0.0
    union = area_ha(pred.Or(truth), region, scale) or 0.0
    return float(inter) / float(union) if union else 0.0


def spot_check(composite, trees, region, n=500, threshold=0.6, seed=42):
    """Sample points and compare the NDVI decision against the DW trees label."""
    stack = ndvi(composite).addBands(trees).addBands(composite.select(["B4", "B8"]))
    sample = stack.sample(region=region, scale=10, numPixels=n, seed=seed)
    features = sample.getInfo()["features"]
    agree = sum(int(p["NDVI"] > threshold) == int(p["trees"])
                for p in (f["properties"] for f in features))
    return agree, len(features)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--threshold", type=float, default=0.6, help="NDVI forest threshold")
    p.add_argument("--samples", type=int, default=500, help="spot-check sample points")
    p.add_argument("--project", default=None, help="Earth Engine Cloud project id")
    p.add_argument("--export-labels", action="store_true", help="export DW trees to Drive")
    p.add_argument("--folder", default="deforestation")
    p.add_argument("--metrics", default="results/metrics.csv")
    args = p.parse_args()

    init(project=args.project)
    if not verify():
        raise SystemExit("Earth Engine not reachable — run `earthengine authenticate`")

    region = ee.Geometry.Rectangle(REGION["bounds"])
    composite = get_cloud_free_composite(region, *DATE1, CLOUD_PCT)
    trees = get_trees_mask(region, *DATE1)
    pred = forest_mask(composite, args.threshold)

    score = iou(pred, trees, region)
    agree, total = spot_check(composite, trees, region, args.samples, args.threshold)
    print(f"region          : {REGION['name']}  {DATE1[0]}..{DATE1[1]}")
    print(f"NDVI threshold  : {args.threshold}")
    print(f"forest ha (NDVI): {area_ha(pred, region):.1f}")
    print(f"forest ha (DW)  : {area_ha(trees, region):.1f}")
    print(f"baseline IoU    : {score:.4f}   <- U-Net must beat this")
    print(f"spot-check      : {agree}/{total} = {agree / total:.3f} agreement" if total else "spot-check: no samples")

    append_metrics(args.metrics, {
        "run_id": time.strftime("%Y%m%d-%H%M%S"),
        "region": REGION["name"],
        "model": "ndvi-baseline",
        "iou": round(score, 4),
        "baseline_iou": round(score, 4),
    })

    if args.export_labels:
        export_bands(trees, region, "dw_trees_2020", folder=args.folder, bands=["trees"])
        print(f"queued dw_trees_2020 export to Drive folder '{args.folder}'")


if __name__ == "__main__":
    main()
