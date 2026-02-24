"""Shared constants for the VA Woods visualization app."""

POSITRON_TILES = "https://{s}.basemaps.cartocdn.com/rastertiles/light_all/{z}/{x}/{y}.png"
POSITRON_ATTR = (
    '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>, '
    '&copy; <a href="https://carto.com/attributions">CARTO</a>'
)

SATELLITE_TILES = (
    "https://server.arcgisonline.com/ArcGIS/rest/services"
    "/World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
SATELLITE_ATTR = (
    "Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, "
    "GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community"
)

PARCEL_BORDER = "#394a10"
HARVESTED_FILL = "#c5ff49"
HARVESTED_FILL_OPACITY = 0.35
HARVEST_POINT = "#3c3c32"

VA_CENTER = [37.5, -79.0]
VA_ZOOM = 7

API_URL = "http://localhost:8000"
