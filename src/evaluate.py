"""Evaluate a trained U-Net on the held-out geographic area.

Reports pixel IoU/F1, the NDVI-baseline comparison, a worst-tile report for
debugging, and logs one row per run to results/metrics.csv.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline.ndvi import ndvi_forest_mask  # noqa: E402
from models.unet import build_unet  # noqa: E402

CSV_FIELDS = ["run_id", "seed", "region", "model", "iou", "f1",
              "baseline_iou", "hectares_cleared"]


def append_metrics(path, row):
    """Append one run to the metrics CSV, creating the header if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def pixel_metrics(pred, target):
    """IoU and F1 for a binary prediction vs target."""
    pred, target = np.asarray(pred).astype(bool), np.asarray(target).astype(bool)
    tp = np.logical_and(pred, target).sum()
    fp = np.logical_and(pred, ~target).sum()
    fn = np.logical_and(~pred, target).sum()
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return float(iou), float(f1)


@torch.no_grad()
def evaluate_model(model, dataset, device, batch_size=8, threshold=0.5, keep_arrays=True):
    """Run the model over a dataset. Returns aggregate + per-tile metrics.

    Each per-tile entry carries IoU/F1, error counts (tp/fp/fn) and, when
    keep_arrays is set, the image, label and prediction for inspection.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    files = getattr(dataset, "files", [f"tile_{i}" for i in range(len(dataset))])
    model.eval()
    per_tile, idx = [], 0
    for img, mask in loader:
        probs = torch.sigmoid(model(img.to(device))).cpu().numpy()
        for b in range(img.size(0)):
            true = mask[b, 0].numpy().astype(bool)
            pred = probs[b, 0] > threshold
            iou, f1 = pixel_metrics(pred, true)
            entry = {"file": files[idx], "iou": iou, "f1": f1,
                     "tp": int(np.logical_and(pred, true).sum()),
                     "fp": int(np.logical_and(pred, ~true).sum()),
                     "fn": int(np.logical_and(~pred, true).sum())}
            if keep_arrays:
                entry["image"] = img[b].numpy()
                entry["mask"] = true
                entry["pred"] = pred
            per_tile.append(entry)
            idx += 1
    return {
        "iou": float(np.mean([t["iou"] for t in per_tile])) if per_tile else 0.0,
        "f1": float(np.mean([t["f1"] for t in per_tile])) if per_tile else 0.0,
        "per_tile": per_tile,
    }


def worst_tiles(per_tile, k=10):
    """Per-tile results sorted by ascending IoU (the hardest tiles first)."""
    return sorted(per_tile, key=lambda t: t["iou"])[:k]


def error_rgb(image_band, true, pred):
    """Grayscale band with false positives red and false negatives blue."""
    band = image_band.astype(float)
    band = (band - band.min()) / (np.ptp(band) + 1e-6)
    rgb = np.repeat(band[..., None], 3, axis=2)
    rgb[np.logical_and(pred, ~true)] = [1, 0, 0]   # false positive
    rgb[np.logical_and(~pred, true)] = [0, 0.5, 1]  # false negative
    return rgb


def save_worst_tiles(per_tile, out_path, k=12, band=3, dpi=110):
    """Contact sheet of the worst tiles: input band, label, prediction, errors."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    worst = worst_tiles(per_tile, k)
    if not worst:
        return None
    n = len(worst)
    fig, axes = plt.subplots(n, 4, figsize=(9, 2.4 * n), squeeze=False)
    cols = [f"input (band {band})", "label", "prediction", "errors (FP red / FN blue)"]
    for row, t in enumerate(worst):
        ax = axes[row]
        ax[0].imshow(t["image"][band], cmap="gray")
        ax[1].imshow(t["mask"], cmap="gray", vmin=0, vmax=1)
        ax[2].imshow(t["pred"], cmap="gray", vmin=0, vmax=1)
        ax[3].imshow(error_rgb(t["image"][band], t["mask"], t["pred"]))
        ax[0].set_ylabel(f"IoU {t['iou']:.2f}", fontsize=8)
        for c, a in enumerate(ax):
            a.set_xticks([]); a.set_yticks([])
            if row == 0:
                a.set_title(cols[c], fontsize=9)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def write_per_tile_csv(per_tile, out_path):
    """Write per-tile IoU/F1 and error counts to CSV."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "iou", "f1", "tp", "fp", "fn"])
        for t in per_tile:
            w.writerow([t["file"], round(t["iou"], 4), round(t["f1"], 4),
                        t["tp"], t["fp"], t["fn"]])
    return path


def ndvi_baseline(dataset, red_idx=2, nir_idx=3, threshold=0.6, batch_size=8):
    """Mean per-tile IoU of the NDVI threshold baseline on the same tiles.

    Note: pass raw (unnormalized) tiles here; running NDVI on per-tile
    normalized bands is not physically meaningful.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    ious = []
    for img, mask in loader:
        for b in range(img.size(0)):
            red = img[b, red_idx].numpy()
            nir = img[b, nir_idx].numpy()
            true = mask[b, 0].numpy().astype(bool)
            iou, _ = pixel_metrics(ndvi_forest_mask(red, nir, threshold), true)
            ious.append(iou)
    return float(np.mean(ious)) if ious else 0.0


def main():
    from train import TileDataset  # deferred to avoid an import cycle

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data/tiles")
    p.add_argument("--split", default="test")
    p.add_argument("--checkpoint", default="results/unet.pth")
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--encoder-weights", default="imagenet")
    p.add_argument("--in-channels", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--baseline-iou", type=float, default=None,
                   help="use a precomputed baseline IoU instead of computing NDVI")
    p.add_argument("--red-idx", type=int, default=2)
    p.add_argument("--nir-idx", type=int, default=3)
    p.add_argument("--baseline-threshold", type=float, default=0.6)
    p.add_argument("--worst-k", type=int, default=10)
    p.add_argument("--snapshot", default="results/worst_tiles.png",
                   help="save a contact sheet of the worst tiles ('' to skip)")
    p.add_argument("--per-tile-csv", default="results/per_tile.csv",
                   help="write per-tile IoU/F1 and error counts ('' to skip)")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--region", default="unknown")
    p.add_argument("--hectares-cleared", type=float, default=None,
                   help="optional hectares from the change-detection step, logged with this run")
    p.add_argument("--metrics", default="results/metrics.csv")
    args = p.parse_args()

    from train import collect_tiles

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )

    files = collect_tiles(args.data_dir, args.split)
    if not files:
        raise SystemExit(f"No {args.split} tiles found under {args.data_dir!r}")
    dataset = TileDataset(files, augment=False)

    weights = None if args.encoder_weights.lower() in ("none", "") else args.encoder_weights
    model = build_unet(in_channels=args.in_channels, encoder=args.encoder,
                       encoder_weights=weights).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    result = evaluate_model(model, dataset, device, args.batch_size, args.threshold)

    if args.baseline_iou is not None:
        baseline = args.baseline_iou
    else:
        baseline = ndvi_baseline(dataset, args.red_idx, args.nir_idx,
                                 args.baseline_threshold, args.batch_size)

    print(f"{args.split} tiles: {len(dataset)}")
    print(f"U-Net   IoU {result['iou']:.4f}  F1 {result['f1']:.4f}")
    print(f"NDVI    IoU {baseline:.4f}")
    print(f"\nWorst {args.worst_k} tiles (IoU / F1 / false-neg% / false-pos%):")
    for t in worst_tiles(result["per_tile"], args.worst_k):
        pos = t["fn"] + t["tp"] or 1
        print(f"  IoU {t['iou']:.4f}  F1 {t['f1']:.4f}  "
              f"FN {100 * t['fn'] / pos:5.1f}%  FP {100 * t['fp'] / pos:5.1f}%  {t['file']}")

    if args.snapshot:
        save_worst_tiles(result["per_tile"], args.snapshot, args.worst_k)
        print(f"\nsaved worst-tile contact sheet -> {args.snapshot}")
    if args.per_tile_csv:
        write_per_tile_csv(result["per_tile"], args.per_tile_csv)
        print(f"saved per-tile metrics      -> {args.per_tile_csv}")

    append_metrics(args.metrics, {
        "run_id": time.strftime("%Y%m%d-%H%M%S"),
        "seed": args.seed,
        "region": args.region,
        "model": f"unet-{args.encoder}",
        "iou": round(result["iou"], 4),
        "f1": round(result["f1"], 4),
        "baseline_iou": round(baseline, 4),
        "hectares_cleared": (round(args.hectares_cleared, 2)
                             if args.hectares_cleared is not None else ""),
    })


if __name__ == "__main__":
    main()
