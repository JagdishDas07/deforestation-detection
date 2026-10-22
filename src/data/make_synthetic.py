"""Generate synthetic Sentinel-2-like GeoTIFFs for testing tile/split/train.

No Earth Engine needed. Produces a before/after 6-band composite and a matching
forest label, so the full pipeline can be exercised end-to-end (e.g. on Colab).
"""
import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds

BAND_NAMES = ["B2", "B3", "B4", "B8", "B11", "B12"]
FOREST = [400, 600, 300, 4500, 1500, 700]     # reflectance * 10000
CLEARED = [1900, 2000, 2000, 2100, 2400, 2200]


def _scene(size, clear_mask, shadow_mask, rng):
    bands = [np.full((size, size), float(v), np.float32)
             + rng.normal(0, 20, (size, size)) for v in FOREST]
    arr = np.stack(bands)
    for c, v in enumerate(CLEARED):
        arr[c][clear_mask] = v
    for c in range(6):
        arr[c][shadow_mask] *= 0.3
    return arr


def generate(out_dir, size=1024, seed=0, clear_px=60):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    s = size

    clear1 = np.zeros((s, s), bool)
    clear1[s // 10:s // 10 + 100, s // 10:s // 10 + 100] = True
    clear1[int(s * 0.68):int(s * 0.68) + 100, s // 2:s // 2 + 100] = True
    clear2 = clear1.copy()
    clear2[int(s * 0.39):int(s * 0.39) + clear_px, int(s * 0.68):int(s * 0.68) + clear_px] = True
    shadow = np.zeros((s, s), bool)
    shadow[int(s * 0.05):int(s * 0.15), int(s * 0.83):int(s * 0.93)] = True

    t1 = _scene(s, clear1, shadow, rng)
    t2 = _scene(s, clear2, shadow, rng)
    label = (~clear1).astype(np.uint8)

    prof = dict(driver="GTiff", height=s, width=s, crs="EPSG:4326",
                transform=from_bounds(-59.5, -9.8, -58.9, -9.2, s, s))
    for name, arr in (("t1.tif", t1), ("t2.tif", t2)):
        with rasterio.open(out / name, "w", **{**prof, "count": 6, "dtype": "float32"}) as dst:
            dst.write(arr)
    with rasterio.open(out / "label.tif", "w", **{**prof, "count": 1, "dtype": "uint8"}) as dst:
        dst.write(label, 1)

    print(f"synthetic GeoTIFFs -> {out}  (t1.tif, t2.tif, label.tif; {s}x{s}, 6 bands)")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/raw_synthetic")
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    generate(args.out, args.size, args.seed)


if __name__ == "__main__":
    main()
