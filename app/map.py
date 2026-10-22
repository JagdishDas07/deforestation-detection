"""Render before/after imagery with cleared areas in red on an interactive map."""
import argparse
from pathlib import Path

import folium
import numpy as np
import rasterio
from rasterio.features import shapes

# band order in the exported composite: B2, B3, B4, B8, B11, B12
RGB_IDX = (2, 1, 0)  # red, green, blue


def build_clearing_map(center_lat, center_lon, cleared_geojson, zoom=10):
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom)
    folium.GeoJson(
        cleared_geojson,
        style_function=lambda x: {"fillColor": "red", "color": "red", "weight": 1, "fillOpacity": 0.5},
        name="Newly cleared forest",
    ).add_to(m)
    folium.LayerControl().add_to(m)
    return m


def mask_to_geojson(mask, transform):
    """Vectorize a binary mask into a GeoJSON FeatureCollection."""
    feats = [{"type": "Feature", "geometry": geom, "properties": {}}
             for geom, val in shapes(mask.astype("uint8"), mask=mask.astype(bool),
                                     transform=transform)
             if val == 1]
    return {"type": "FeatureCollection", "features": feats}


def rgb_from_bands(image, scale=(0, 3000), max_dim=2048):
    """(bands, H, W) -> uint8 (H, W, 3) RGB for display.

    Downsampled to `max_dim` on the longest side so the overlay inlined in the
    HTML stays a few MB instead of ~100 MB; NaN reflectance becomes 0.
    """
    step = max(1, -(-max(image.shape[1:]) // max_dim))
    rgb = np.transpose(image[list(RGB_IDX), ::step, ::step], (1, 2, 0)).astype(float)
    rgb = np.nan_to_num(rgb, nan=scale[0])
    rgb = np.clip((rgb - scale[0]) / (scale[1] - scale[0]), 0, 1)
    return (rgb * 255).astype("uint8")


def legend_html(hectares, pixels, color="red"):
    """Floating box showing the total cleared-hectares figure."""
    return (
        '<div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;'
        ' background: rgba(255,255,255,0.92); padding: 10px 14px; border: 2px solid #444;'
        ' border-radius: 6px; font: 13px/1.5 sans-serif;">'
        '<b>Newly cleared forest</b><br>'
        f'<span style="display:inline-block;width:12px;height:12px;background:{color};'
        'margin-right:6px;"></span>forest &rarr; non-forest<br>'
        f'<b>{hectares:,.1f} ha</b> ({pixels:,} px @ 10&nbsp;m)</div>'
    )


def build_map(t1_path, t2_path, cleared_mask, out_html, zoom=12, pixel_area_ha=0.01):
    with rasterio.open(t1_path) as src:
        t1, transform, bounds = src.read().astype(np.float32), src.transform, src.bounds
    with rasterio.open(t2_path) as src:
        t2 = src.read().astype(np.float32)

    center = [(bounds.bottom + bounds.top) / 2, (bounds.left + bounds.right) / 2]
    extent = [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]

    m = folium.Map(location=center, zoom_start=zoom)
    folium.raster_layers.ImageOverlay(rgb_from_bands(t1), bounds=extent, name="date 1").add_to(m)
    folium.raster_layers.ImageOverlay(rgb_from_bands(t2), bounds=extent, name="date 2").add_to(m)
    folium.GeoJson(
        mask_to_geojson(cleared_mask, transform),
        style_function=lambda x: {"fillColor": "red", "color": "red", "weight": 1, "fillOpacity": 0.5},
        name="newly cleared",
    ).add_to(m)
    folium.LayerControl().add_to(m)

    pixels = int(np.asarray(cleared_mask).sum())
    m.get_root().html.add_child(folium.Element(legend_html(pixels * pixel_area_ha, pixels)))

    Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    m.save(out_html)
    return out_html


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--t1", required=True, help="date-1 composite GeoTIFF")
    p.add_argument("--t2", required=True, help="date-2 composite GeoTIFF")
    p.add_argument("--cleared", required=True, help="cleared mask .npy (bool/uint8)")
    p.add_argument("--out", default="results/clearing_map.html")
    p.add_argument("--zoom", type=int, default=12)
    p.add_argument("--pixel-area-ha", type=float, default=0.01)
    args = p.parse_args()

    cleared = np.load(args.cleared).astype(bool)
    path = build_map(args.t1, args.t2, cleared, args.out, args.zoom, args.pixel_area_ha)
    print(f"wrote {path}  ({int(cleared.sum())} cleared px = {cleared.sum() * args.pixel_area_ha:.2f} ha)")


if __name__ == "__main__":
    main()
