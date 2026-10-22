"""Dynamic World forest labels — the 'trees' class used as ground-truth mask.

Dynamic World labels: 0 water, 1 trees, 2 grass, 3 flooded vegetation,
4 crops, 5 shrub & scrub, 6 built, 7 bare, 8 snow & ice.
"""
import ee

DW_COLLECTION = "GOOGLE/DYNAMICWORLD/V1"
TREES_CLASS = 1


def get_dynamic_world(region, start_date, end_date):
    """Dynamic World image collection over a region and date window."""
    return (ee.ImageCollection(DW_COLLECTION)
            .filterBounds(region)
            .filterDate(start_date, end_date))


def get_trees_mask(region, start_date, end_date, method="mode"):
    """Binary 'trees' mask over the window.

    method='mode' -> most frequent label equals trees
    method='prob' -> mean trees probability >= 0.5
    """
    dw = get_dynamic_world(region, start_date, end_date)
    if method == "prob":
        mask = dw.select("trees").mean().gte(0.5)
    else:
        mask = dw.select("label").mode().eq(TREES_CLASS)
    return mask.rename("trees").clip(region).toByte()
