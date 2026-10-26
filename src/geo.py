"""Geometric helpers shared by the change-detection and mapping steps."""
import math


def pixel_area_ha(src):
    """Ground area of one pixel in hectares, from an open raster.

    A 10 m Sentinel-2 product is not exactly 10 x 10 m on the ground: the
    latitude spacing is fixed, but the longitude spacing shrinks with
    cos(latitude). At these latitudes a flat 0.01 ha per pixel overstates the
    real area by ~1.5%.

    Handles both geographic (degree) and projected (metre) rasters, and uses
    the full transform determinant so rotated/sheared grids are measured
    correctly instead of assuming an axis-aligned square pixel.
    """
    t = src.transform
    det = abs(t.a * t.e - t.b * t.d)  # pixel area in CRS units^2
    if src.crs is not None and src.crs.is_geographic:
        lat = (src.bounds.top + src.bounds.bottom) / 2
        m_lat = 111132.92
        m_lon = 111319.49 * math.cos(math.radians(lat))
        area_m2 = det * m_lon * m_lat
    else:
        area_m2 = det  # projected CRS: units are already metres
    return area_m2 / 10000.0


def pixel_area_ha_for(path):
    """Ground area of one pixel in hectares, for a raster on disk."""
    import rasterio
    with rasterio.open(path) as src:
        return pixel_area_ha(src)
