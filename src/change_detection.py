"""Compare forest masks across two dates, filter small noise patches, report hectares."""
import numpy as np
from scipy import ndimage

def detect_clearing(forest_t1, forest_t2, min_pixels=50, pixel_area_ha=0.01):
    """min_pixels=50 at 10m resolution ~= 0.5 hectares."""
    cleared = forest_t1 & ~forest_t2
    labels, n = ndimage.label(cleared)
    sizes = ndimage.sum(cleared, labels, range(1, n + 1))
    keep = np.where(sizes >= min_pixels)[0] + 1
    cleared_clean = np.isin(labels, keep)
    hectares = cleared_clean.sum() * pixel_area_ha
    return cleared_clean, hectares
