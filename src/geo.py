"""Geometric helpers shared by the change-detection and mapping steps."""
import math


def pixel_area_ha(src):
    """Ground area of one pixel in hectares, from an open raster.

    A 10 m Sentinel-2 product is not exactly 10 x 10 m on the ground: the
    latitude spacing is fixed, but the longitude spacing shrinks with
    cos(latitude). At these latitudes a flat 0.01 ha per pixel overstates the
    real area by ~1.5%.
    """
    res_x = abs(src.transform.a)   # degrees of longitude per pixel
    res_y = abs(src.transform.e)   # degrees of latitude per pixel
    lat = (src.bounds.top + src.bounds.bottom) / 2
    m_lat = 111132.92
    m_lon = 111319.49 * math.cos(math.radians(lat))
    return (res_x * m_lon) * (res_y * m_lat) / 10000.0


def pixel_area_ha_for(path):
    """Ground area of one pixel in hectares, for a raster on disk."""
    import rasterio
    with rasterio.open(path) as src:
        return pixel_area_ha(src)
