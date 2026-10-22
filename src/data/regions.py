"""Chosen study area and matched-season date windows.

Region: Colniza, Mato Grosso — part of the Brazilian "arc of deforestation",
a well-documented clearing hotspot (INPE DETER/PRODES monitoring).

Dates: two dry-season windows (July–August) in different years. Sampling the
same season avoids confusing seasonal vegetation shifts with real forest loss.
"""

REGION = {
    "name": "Amazon arc of deforestation - Colniza, Mato Grosso, BR",
    "bounds": [-59.5, -9.8, -58.9, -9.2],  # [west, south, east, north], EPSG:4326
}

DATE1 = ("2020-07-01", "2020-08-31")
DATE2 = ("2022-07-01", "2022-08-31")

CLOUD_PCT = 20
