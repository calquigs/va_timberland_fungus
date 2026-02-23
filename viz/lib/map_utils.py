"""Shared Folium map helpers."""

import folium

from lib.constants import POSITRON_ATTR, POSITRON_TILES, VA_CENTER, VA_ZOOM


def base_map(center=None, zoom=None) -> folium.Map:
    """Return a Folium Map with Positron basemap tiles."""
    m = folium.Map(
        location=center or VA_CENTER,
        zoom_start=zoom or VA_ZOOM,
        tiles=None,
        control_scale=True,
    )
    folium.TileLayer(
        tiles=POSITRON_TILES,
        attr=POSITRON_ATTR,
        name="Positron",
        control=True,
    ).add_to(m)
    return m
