"""Earth Engine auth, Sentinel-2 L2A filtering, cloud masking, and export."""
import os

import ee

# Sentinel-2 L2A Scene Classification Layer (SCL) classes to mask out:
#   0 no-data, 1 saturated/defective, 3 cloud shadow,
#   8 cloud (medium prob), 9 cloud (high prob), 10 thin cirrus, 11 snow/ice
SCL_MASK_CLASSES = (0, 1, 3, 8, 9, 10, 11)

# blue, green, red, NIR, SWIR-1, SWIR-2
DEFAULT_BANDS = ("B2", "B3", "B4", "B8", "B11", "B12")

def init(project=None):
    """Initialize Earth Engine.

    Pass your Cloud project id, or set the EE_PROJECT environment variable.
    Modern Earth Engine accounts require a project for ee.Initialize().
    """
    ee.Initialize(project=project or os.environ.get("EE_PROJECT"))

def verify():
    """Return True if Earth Engine responds to a trivial request."""
    return bool(ee.Number(1).add(1).getInfo() == 2)

def get_s2_collection(region, start_date, end_date, cloud_pct=20):
    """Sentinel-2 L2A scenes over a region/date window below a cloud threshold."""
    return (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(region)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct)))

def mask_clouds_shadows(img):
    """Mask clouds, cloud shadows, cirrus, snow and bad pixels using the SCL band."""
    scl = img.select("SCL")
    mask = ee.Image.constant(1)
    for cls in SCL_MASK_CLASSES:
        mask = mask.And(scl.neq(cls))
    return img.updateMask(mask)

def get_cloud_free_composite(region, start_date, end_date, cloud_pct=20):
    """Filter L2A by cloud %, mask via SCL, and return a median composite."""
    s2 = get_s2_collection(region, start_date, end_date, cloud_pct)
    return s2.map(mask_clouds_shadows).median().clip(region)

def export_bands(image, region, description, folder="deforestation",
                 bands=DEFAULT_BANDS, scale=10, file_format="GeoTIFF"):
    """Export the selected bands of a composite to Google Drive as a GeoTIFF."""
    task = ee.batch.Export.image.toDrive(
        image=image.select(list(bands)),
        description=description,
        folder=folder,
        fileNamePrefix=description,
        region=region,
        scale=scale,
        maxPixels=1e9,
        fileFormat=file_format,
    )
    task.start()
    return task
