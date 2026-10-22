"""Tile rasters into fixed-size patches and split by geographic block (not randomly)."""
import numpy as np

def tile_raster(array, tile_size=256):
    """array: (bands, H, W) -> list of (bands, tile_size, tile_size) tiles."""
    _, h, w = array.shape
    tiles = []
    for y in range(0, h - tile_size + 1, tile_size):
        for x in range(0, w - tile_size + 1, tile_size):
            tiles.append(array[:, y:y+tile_size, x:x+tile_size])
    return tiles

def normalize(tile, mean=None, std=None):
    mean = mean if mean is not None else tile.mean(axis=(1, 2), keepdims=True)
    std = std if std is not None else tile.std(axis=(1, 2), keepdims=True) + 1e-6
    return (tile - mean) / std

def split_by_region(tile_coords, test_region_bounds):
    """tile_coords: list of (x, y) origins. Returns (train_idx, test_idx) by a geographic cutoff."""
    train_idx, test_idx = [], []
    for i, (x, y) in enumerate(tile_coords):
        if test_region_bounds[0] <= x <= test_region_bounds[1]:
            test_idx.append(i)
        else:
            train_idx.append(i)
    return train_idx, test_idx
