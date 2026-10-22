"""NDVI threshold baseline for forest cover."""
import numpy as np

def compute_ndvi(red, nir):
    return (nir - red) / (nir + red + 1e-6)

def ndvi_forest_mask(red, nir, threshold=0.6):
    return compute_ndvi(red, nir) > threshold

def iou(pred_mask, true_mask):
    pred_mask, true_mask = pred_mask.astype(bool), true_mask.astype(bool)
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    return intersection / union if union > 0 else 0.0
