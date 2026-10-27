"""Feature engineering functions for the King County house sales data.

Every function takes a DataFrame and returns a NEW DataFrame.
Call them on cleaned data (see cleaning.py).
"""
import numpy as np
import pandas as pd

# The "center of wealth" used in the notebook (latitude, longitude).
CENTER_LAT = 47.62774
CENTER_LONG = -122.24194
# The notebook corrects the longitude distance with the cosine of THIS latitude.
CENTER_COS_LAT = 47.6219
# Kilometres for one degree of latitude: 2 * pi * Earth radius / 360.
EARTH_RADIUS_KM = 6378
KM_PER_DEGREE = 2 * np.pi * EARTH_RADIUS_KM / 360


def distance_km(long, lat, ref_long, ref_lat, cos_lat=None):
    """Distance in km between points and a reference location (flat-earth formula).

    Works with single numbers and with numpy arrays (also with broadcasting).
    `cos_lat` is the latitude used to shrink the longitude distance.
    By default it is the latitude of the reference location.
    """
    if cos_lat is None:
        cos_lat = ref_lat
    delta_long = (long - ref_long) * np.cos(np.radians(cos_lat))
    delta_lat = lat - ref_lat
    return np.sqrt(delta_long**2 + delta_lat**2) * KM_PER_DEGREE


def add_sqft_price(df: pd.DataFrame) -> pd.DataFrame:
    """Add `sqft_price`: price per square foot of living plus lot area.

    It uses the price, so it can NOT be a feature for a model (data leakage).
    """
    df = df.copy()
    df["sqft_price"] = (df["price"] / (df["sqft_living"] + df["sqft_lot"])).round(2)
    return df


def add_center_distance(df: pd.DataFrame) -> pd.DataFrame:
    """Add `center_distance`: km from the house to the center of wealth."""
    df = df.copy()
    df["center_distance"] = distance_km(
        df["long"], df["lat"], CENTER_LONG, CENTER_LAT, cos_lat=CENTER_COS_LAT
    )
    return df


def get_waterfront_locations(df: pd.DataFrame) -> pd.DataFrame:
    """Return the `long` and `lat` of every waterfront house.

    These are the reference points for `water_distance`. Compute them once from
    the training data and reuse them for new houses.
    """
    return df.loc[df["waterfront"] == 1, ["long", "lat"]].reset_index(drop=True)


def add_water_distance(df: pd.DataFrame, waterfront_locations: pd.DataFrame) -> pd.DataFrame:
    """Add `water_distance`: km to the nearest waterfront house.

    One numpy calculation replaces the two loops of the notebook.
    """
    df = df.copy()
    ref_long = waterfront_locations["long"].to_numpy()[None, :]
    ref_lat = waterfront_locations["lat"].to_numpy()[None, :]
    long = df["long"].to_numpy()[:, None]
    lat = df["lat"].to_numpy()[:, None]
    all_distances = distance_km(long, lat, ref_long, ref_lat)
    df["water_distance"] = all_distances.min(axis=1)
    return df


def add_features(df: pd.DataFrame, waterfront_locations: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add all new features.

    - `sqft_price` is added only if the data has a `price` column.
      A NEW house has no price yet, so it gets the other two features only.
    - For new houses, pass the `waterfront_locations` of the training data.
      If you pass nothing, they are taken from `df` itself.
    """
    if waterfront_locations is None:
        waterfront_locations = get_waterfront_locations(df)
    if "price" in df.columns:
        df = add_sqft_price(df)
    return df.pipe(add_center_distance).pipe(add_water_distance, waterfront_locations)
