"""Cut exported GeoTIFFs into 256x256 tiles, normalize, and split by area.

Reads a 6-band Sentinel-2 composite GeoTIFF and the matching Dynamic World
'trees' label GeoTIFF, normalizes each band using whole-scene statistics, tiles
to 256x256, and assigns whole geographic blocks to train/val/test (so evaluation
generalizes to unseen areas rather than to near-identical neighbouring tiles).

Outputs .npz tiles (image, mask) plus split.json and meta.json under --out.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.split import geographic_split  # noqa: E402
from data.tiling import tile_raster  # noqa: E402


def read_image(path):
    with rasterio.open(path) as src:
        return src.read().astype(np.float32), src.transform, src.crs


def read_label(path):
    with rasterio.open(path) as src:
        return (src.read(1) > 0).astype(np.uint8)


def band_stats(image):
    """Per-band mean/std over the whole scene (better than per-tile stats)."""
    flat = image.reshape(image.shape[0], -1)
    mean = np.nanmean(flat, axis=1)
    std = np.nanstd(flat, axis=1) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


def origins(h, w, tile_size):
    return [(x, y) for y in range(0, h - tile_size + 1, tile_size)
            for x in range(0, w - tile_size + 1, tile_size)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", required=True, help="6-band composite GeoTIFF")
    p.add_argument("--label", required=True, help="forest label GeoTIFF (nonzero = forest)")
    p.add_argument("--out", default="data/tiles")
    p.add_argument("--tile-size", type=int, default=256)
    p.add_argument("--block-size", type=int, default=1024,
                   help="geographic block edge in pixels (kept whole across splits)")
    p.add_argument("--ratios", type=float, nargs=3, default=(0.7, 0.15, 0.15))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    image, transform, crs = read_image(args.image)
    label = read_label(args.label)
    if label.shape != image.shape[1:]:
        raise SystemExit(f"label {label.shape} != image {image.shape[1:]}")

    mean, std = band_stats(image)
    mean = np.nan_to_num(mean, nan=0.0)
    std = np.nan_to_num(std, nan=1.0)
    mean, std = mean.astype(np.float32), std.astype(np.float32)
    image = (image - mean[:, None, None]) / std[:, None, None]

    image_tiles = tile_raster(image, args.tile_size)
    mask_tiles = tile_raster(label[None], args.tile_size)
    coords = origins(*image.shape[1:], args.tile_size)
    if len(coords) != len(image_tiles):
        raise SystemExit("tile count mismatch")

    split = geographic_split(coords, args.block_size, tuple(args.ratios), args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = {}
    for i, (img_t, msk_t) in enumerate(zip(image_tiles, mask_tiles)):
        # fill nans in tile
        if np.any(np.isnan(img_t)):
            img_t = np.nan_to_num(img_t, nan=0.0)
        name = f"tile_{i:05d}.npz"
        np.savez_compressed(out / name, image=img_t.astype(np.float32),
                            mask=msk_t[0].astype(np.uint8))
        names[i] = name

    named = {k: [names[i] for i in v] for k, v in split.items()}
    (out / "split.json").write_text(json.dumps(named, indent=2))
    (out / "meta.json").write_text(json.dumps({
        "tile_size": args.tile_size, "block_size": args.block_size,
        "band_mean": mean.tolist(), "band_std": std.tolist(),
        "origins": [list(c) for c in coords],
        "transform": list(transform)[:6], "crs": str(crs),
    }, indent=2))

    counts = {k: len(v) for k, v in named.items()}
    print(f"tiles: {len(coords)}  ->  {counts}")
    print(f"bands: {image.shape[0]}  mean: {np.round(mean, 2).tolist()}")
    print(f"wrote {out}/ (tiles + split.json + meta.json)")


if __name__ == "__main__":
    main()
