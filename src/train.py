"""Training loop: Dice+BCE loss, flip/rotation augmentation, fixed seeds."""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.unet import build_unet  # noqa: E402
from evaluate import append_metrics  # noqa: E402


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dice_bce_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    bce = torch.nn.functional.binary_cross_entropy(pred, target)
    intersection = (pred * target).sum()
    dice = 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return bce + dice


def random_flip_rotate(img, mask):
    """Random horizontal/vertical flips and 90-degree rotations, applied to both."""
    if torch.rand(1).item() < 0.5:
        img, mask = torch.flip(img, [2]), torch.flip(mask, [2])
    if torch.rand(1).item() < 0.5:
        img, mask = torch.flip(img, [1]), torch.flip(mask, [1])
    k = int(torch.randint(0, 4, (1,)).item())
    if k:
        img, mask = torch.rot90(img, k, [1, 2]), torch.rot90(mask, k, [1, 2])
    return img, mask


class TileDataset(Dataset):
    """Loads (image, mask) tiles saved as .npz files with keys 'image'/'mask'.

    image: (bands, H, W) float32, mask: (H, W) binary.
    """

    def __init__(self, files, augment=False):
        self.files = [str(f) for f in files]
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        data = np.load(self.files[i])
        img = torch.from_numpy(data["image"].astype(np.float32))
        mask = torch.from_numpy(data["mask"].astype(np.float32))[None]
        if self.augment:
            img, mask = random_flip_rotate(img, mask)
        return img, mask


def batch_metrics(logits, target, threshold=0.5):
    """IoU and F1 over all pixels in a batch of logits."""
    pred = (torch.sigmoid(logits) > threshold).float()
    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    iou = tp / (tp + fp + fn + 1e-6)
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-6)
    return float(iou), float(f1)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    for img, mask in loader:
        img, mask = img.to(device), mask.to(device)
        optimizer.zero_grad()
        loss = dice_bce_loss(model(img), mask)
        loss.backward()
        optimizer.step()
        total += loss.item() * img.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    ious, f1s = [], []
    for img, mask in loader:
        img, mask = img.to(device), mask.to(device)
        iou, f1 = batch_metrics(model(img), mask)
        ious.append(iou)
        f1s.append(f1)
    return float(np.mean(ious)), float(np.mean(f1s))


def collect_tiles(root, split):
    """Locate tiles for a split: {root}/split.json listing files, or a subdir."""
    root = Path(root)
    manifest = root / "split.json"
    if manifest.exists():
        mapping = json.loads(manifest.read_text())
        return [str(root / name) for name in mapping.get(split, [])]
    sub = root / split
    if sub.is_dir():
        return sorted(str(p) for p in sub.rglob("*.npz"))
    return sorted(str(p) for p in root.glob("*.npz"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data/tiles")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--in-channels", type=int, default=6)
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--encoder-weights", default="imagenet",
                   help="'imagenet', or 'none' for random init (offline/no download)")
    p.add_argument("--device", default="auto")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--out", default="results/unet.pth")
    p.add_argument("--metrics", default="results/metrics.csv")
    p.add_argument("--region", default="unknown")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )

    train_files = collect_tiles(args.data_dir, "train")
    val_files = collect_tiles(args.data_dir, "val")
    if not train_files:
        raise SystemExit(f"No training tiles found under {args.data_dir!r}")

    train_loader = DataLoader(
        TileDataset(train_files, augment=True),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
    )
    val_loader = None
    if val_files:
        val_loader = DataLoader(
            TileDataset(val_files, augment=False),
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        )

    weights = None if args.encoder_weights.lower() in ("none", "") else args.encoder_weights
    model = build_unet(in_channels=args.in_channels, encoder=args.encoder,
                       encoder_weights=weights).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    best_iou, best_f1 = -1.0, 0.0
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        if val_loader is not None:
            iou, f1 = validate(model, val_loader, device)
            if iou > best_iou:
                best_iou, best_f1 = iou, f1
                torch.save(model.state_dict(), args.out)
            print(f"epoch {epoch:3d}  loss {loss:.4f}  val IoU {iou:.4f}  val F1 {f1:.4f}")
        else:
            torch.save(model.state_dict(), args.out)
            print(f"epoch {epoch:3d}  loss {loss:.4f}  (no val split)")

    if val_loader is not None:
        append_metrics(args.metrics, {
            "run_id": time.strftime("%Y%m%d-%H%M%S"),
            "seed": args.seed,
            "region": args.region,
            "model": f"unet-{args.encoder}",
            "iou": round(best_iou, 4),
            "f1": round(best_f1, 4),
        })
        print(f"best val IoU {best_iou:.4f}  F1 {best_f1:.4f}  -> {args.out}")


if __name__ == "__main__":
    main()
