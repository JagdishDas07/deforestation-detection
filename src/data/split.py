"""Geographic block split: hold out whole spatial blocks, never random tiles.

Neighbouring tiles share nearly identical pixels, so a random split leaks
information between train and test. Group tiles into square geographic blocks
and assign whole blocks to train/val/test so evaluation measures generalization
to genuinely unseen areas.
"""
import json
import random
from collections import defaultdict


def block_coords(x, y, block_size):
    """Map a tile origin (x, y) in pixels to its (col, row) block index."""
    return (x // block_size, y // block_size)


def group_blocks(tile_coords, block_size):
    """Group tile indices by geographic block.

    tile_coords: list of (x, y) tile origins (pixels).
    block_size:  block edge in pixels (e.g. block_size=1024 with 256px tiles
                 gives 4x4 tiles per block).
    Returns {(col, row): [tile_idx, ...]}.
    """
    blocks = defaultdict(list)
    for idx, (x, y) in enumerate(tile_coords):
        blocks[block_coords(x, y, block_size)].append(idx)
    return dict(blocks)


def geographic_split(tile_coords, block_size, ratios=(0.7, 0.15, 0.15), seed=42):
    """Assign whole geographic blocks to train/val/test.

    Returns {"train": [tile_idx, ...], "val": [...], "test": [...]}. When there
    are at least three blocks, val and test are guaranteed at least one block.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("ratios must sum to 1.0")

    blocks = group_blocks(tile_coords, block_size)
    keys = sorted(blocks)
    random.Random(seed).shuffle(keys)
    n = len(keys)

    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    if n >= 3:
        n_train = max(1, min(n - 2, n_train))
        n_val = max(1, min(n - n_train - 1, n_val))

    parts = {
        "train": keys[:n_train],
        "val": keys[n_train:n_train + n_val],
        "test": keys[n_train + n_val:],
    }
    return {name: [i for k in ks for i in blocks[k]] for name, ks in parts.items()}


def save_split(split, path):
    """Persist a split mapping to JSON (indices are stored as lists)."""
    with open(path, "w") as f:
        json.dump({k: [int(i) for i in v] for k, v in split.items()}, f, indent=2)


def save_split_by_name(split, tile_names, path):
    """Persist a split as {split: [filename, ...]} so tile files can be located."""
    named = {k: [tile_names[i] for i in v] for k, v in split.items()}
    with open(path, "w") as f:
        json.dump(named, f, indent=2)
