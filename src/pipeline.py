"""End-to-end: NDVI baseline, U-Net segmentation, and two-date change detection.

Runs the same three steps on a before/after pair of 6-band scenes:
  forest mask (NDVI threshold or U-Net) for each date -> cleared = before & ~after
  -> drop patches smaller than min_pixels -> report hectares.

Use --demo for a synthetic pair (runs with no data / no Earth Engine).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline.ndvi import compute_ndvi, ndvi_forest_mask  # noqa: E402
from change_detection import detect_clearing  # noqa: E402
from evaluate import append_metrics  # noqa: E402
from geo import pixel_area_ha_for  # noqa: E402
from models.unet import build_unet  # noqa: E402

# band order used throughout: B2, B3, B4(red), B8(NIR), B11, B12
RED_IDX, NIR_IDX = 2, 3


def read_geotiff(path):
    """Read a multi-band GeoTIFF -> (bands, H, W) float32."""
    import rasterio
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)


def load_meta(path):
    """Per-band mean/std saved by prepare_tiles.py."""
    data = json.loads(Path(path).read_text())
    return np.array(data["band_mean"], np.float32), np.array(data["band_std"], np.float32)


def scene_stats(image):
    """Per-band mean/std of one scene (fallback when no meta.json is given)."""
    flat = image.reshape(image.shape[0], -1)
    return flat.mean(axis=1), flat.std(axis=1) + 1e-6


def normalize(image, mean, std):
    """Apply the same whole-scene normalization used during training."""
    return (image - mean[:, None, None]) / std[:, None, None]


def baseline_forest(b4, b8, threshold=0.6):
    """NDVI threshold baseline -> binary forest mask."""
    return ndvi_forest_mask(b4, b8, threshold)


@torch.no_grad()
def unet_forest(model, image, device="cpu", threshold=0.5, chunk=1024):
    """U-Net inference on a (6, H, W) scene -> binary forest mask (H, W).

    Runs in `chunk`x`chunk` windows (a multiple of the encoder's 32 px stride)
    so a full scene fits in memory; a single forward pass would not.
    """
    model.eval()
    image = np.asarray(image, np.float32)
    h, w = image.shape[1:]
    prob = np.zeros((h, w), np.float32)
    for y in range(0, h, chunk):
        for x in range(0, w, chunk):
            block = torch.from_numpy(image[:, y:y + chunk, x:x + chunk])[None].to(device)
            prob[y:y + chunk, x:x + chunk] = torch.sigmoid(model(block))[0, 0].cpu().numpy()
    return prob > threshold


def synthetic_pair(size=512, seed=0):
    """A fully forested scene plus a copy with a 60x60 patch cleared (3600 px = 36 ha)."""
    rng = np.random.default_rng(seed)

    def bands():
        b4 = np.full((size, size), 0.05) + rng.normal(0, 0.005, (size, size))
        b8 = np.full((size, size), 0.45) + rng.normal(0, 0.005, (size, size))
        return b4, b8

    def stack(b4, b8):
        return np.stack([b4 * 0.5, b4, b4, b8, b8 * 0.6, b8 * 0.4])

    b4a, b8a = bands()
    b4b, b8b = bands()
    b4b[100:160, 100:160], b8b[100:160, 100:160] = 0.25, 0.20
    return stack(b4a, b8a), stack(b4b, b8b)


def load_pair(data_dir):
    """Load <data_dir>/t1.npy and <data_dir>/t2.npy (6, H, W) scenes."""
    t1 = np.load(Path(data_dir) / "t1.npy")
    t2 = np.load(Path(data_dir) / "t2.npy")
    return t1, t2


def report(name, before, after, min_pixels, pixel_area_ha):
    cleared, ha = detect_clearing(before, after, min_pixels, pixel_area_ha)
    print(f"{name:8s}: before {before.mean():.2%} forest -> after {after.mean():.2%} forest "
          f"| cleared {int(cleared.sum())} px = {ha:.2f} ha")
    return cleared, ha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--demo", action="store_true", help="use a synthetic before/after pair")
    p.add_argument("--data-dir", default=None, help="dir with t1.npy / t2.npy")
    p.add_argument("--t1", default=None, help="date-1 composite GeoTIFF (6 bands)")
    p.add_argument("--t2", default=None, help="date-2 composite GeoTIFF (6 bands)")
    p.add_argument("--meta", default=None, help="meta.json for U-Net normalization (from prepare_tiles)")
    p.add_argument("--unet", action="store_true", help="also run the U-Net")
    p.add_argument("--checkpoint", default=None, help="U-Net weights (else random/pretrained)")
    p.add_argument("--encoder-weights", default="imagenet")
    p.add_argument("--in-channels", type=int, default=6)
    p.add_argument("--ndvi-threshold", type=float, default=0.6)
    p.add_argument("--min-pixels", type=int, default=50, help="50 px = ~0.5 ha at 10 m")
    p.add_argument("--pixel-area-ha", type=float, default=None,
                   help="hectares per pixel (default: computed from the raster's resolution)")
    p.add_argument("--save-mask", default=None, help="write the cleared mask to this .npy")
    p.add_argument("--map", dest="map_html", default=None, help="render before/after + cleared map to this HTML")
    p.add_argument("--zoom", type=int, default=12)
    p.add_argument("--region", default="unknown")
    p.add_argument("--metrics", default=None, help="append this run to a metrics CSV")
    args = p.parse_args()

    if args.t1 and args.t2:
        t1, t2 = read_geotiff(args.t1), read_geotiff(args.t2)
    elif args.demo:
        t1, t2 = synthetic_pair()
    elif args.data_dir:
        t1, t2 = load_pair(args.data_dir)
    else:
        raise SystemExit("Provide --t1/--t2 GeoTIFFs, --data-dir, or use --demo")

    print(f"NDVI baseline (threshold {args.ndvi_threshold}, min patch {args.min_pixels} px)")

    pixel_area_ha = args.pixel_area_ha
    if pixel_area_ha is None:
        pixel_area_ha = pixel_area_ha_for(args.t1) if args.t1 else 0.01
    print(f"pixel area: {pixel_area_ha:.6f} ha/px"
          f"{'' if args.pixel_area_ha is None else ' (overridden)'}")

    cleared, hectares = report(
        "baseline",
        baseline_forest(t1[RED_IDX], t1[NIR_IDX], args.ndvi_threshold),
        baseline_forest(t2[RED_IDX], t2[NIR_IDX], args.ndvi_threshold),
        args.min_pixels, pixel_area_ha)

    if args.unet:
        weights = None if args.encoder_weights.lower() in ("none", "") else args.encoder_weights
        model = build_unet(in_channels=args.in_channels, encoder_weights=weights)
        if args.checkpoint:
            model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        if args.meta:
            mean, std = load_meta(args.meta)
            src = args.meta
        else:
            mean, std = scene_stats(t1)
            src = "scene stats (pass --meta for training-matched normalization)"
        print(f"\nU-Net (channels {args.in_channels}->1, enc={weights}, norm from {src})")
        cleared, hectares = report(
            "unet",
            unet_forest(model, normalize(t1, mean, std)),
            unet_forest(model, normalize(t2, mean, std)),
            args.min_pixels, pixel_area_ha)

    print(f"\nNewly cleared forest: {hectares:.2f} ha "
          f"({int(cleared.sum())} px, >= {args.min_pixels} px per patch)")

    if args.save_mask:
        Path(args.save_mask).parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_mask, cleared.astype(np.uint8))
        print(f"saved cleared mask -> {args.save_mask}")

    if args.map_html:
        if not (args.t1 and args.t2):
            raise SystemExit("--map requires --t1/--t2 GeoTIFFs for the before/after base layers")
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from app.map import build_map
        build_map(args.t1, args.t2, cleared, args.map_html, args.zoom, pixel_area_ha)
        print(f"saved cleared-area map -> {args.map_html}")

    if args.metrics:
        append_metrics(args.metrics, {
            "run_id": time.strftime("%Y%m%d-%H%M%S"),
            "region": args.region,
            "model": "unet+change" if args.unet else "ndvi+change",
            "split": "",
            "hectares_cleared": round(hectares, 2),
        })
        print(f"logged run -> {args.metrics}")


if __name__ == "__main__":
    main()
