import osmnx as ox
import pandas as pd
from typing import Optional


def get_ice_cream_places(
    city: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    radius: int = 1000,
    cache_dir: str = "cache",
) -> pd.DataFrame:
    if city:
        lat, lon = ox.geocode(city)
    else:
        if latitude is None or longitude is None:
            raise ValueError("Debes indicar una ciudad o coordenadas (latitud y longitud).")
        lat, lon = latitude, longitude

    ox.settings.use_cache = True
    ox.settings.cache_folder = cache_dir

    gdf = ox.features_from_point((lat, lon), tags={"amenity": "ice_cream"}, dist=radius)
    if gdf.empty:
        return pd.DataFrame(columns=["osmid", "name", "lat", "lon", "addr:street", "addr:housenumber", "amenity"])

    df = gdf.reset_index()
    df["lat"] = df.geometry.y
    df["lon"] = df.geometry.x

    cols = ["osmid", "name", "lat", "lon", "addr:street", "addr:housenumber", "amenity"]
    for col in cols:
        if col not in df.columns:
            df[col] = None
    df = df[cols]
    df = df.where(pd.notnull(df), None)
    return df
