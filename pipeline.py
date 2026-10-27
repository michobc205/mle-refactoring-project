"""The reusable pipeline: raw CSV file -> clean data with all features.

Run it from the command line:   python pipeline.py
Or use it in a notebook:        from pipeline import run_pipeline
"""
import pandas as pd

from cleaning import DATA_PATH, clean_data, load_data
from features import add_features

# Columns that a model must NOT use as input:
# `price` is the target, `sqft_price` is built from the price (data leakage),
# and `date` is text.
NON_FEATURE_COLUMNS = ["price", "sqft_price", "date"]


def prepare_data(
    df: pd.DataFrame, waterfront_locations: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Clean the data and add the new features (steps 1 and 2 of the project)."""
    return add_features(clean_data(df), waterfront_locations)


def run_pipeline(path: str = DATA_PATH) -> pd.DataFrame:
    """Read the CSV file and return the prepared data."""
    return prepare_data(load_data(path))


def split_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return the model input `X` and the target `y` (the price)."""
    X = df.drop(columns=NON_FEATURE_COLUMNS)
    y = df["price"]
    return X, y


if __name__ == "__main__":
    prepared = run_pipeline()
    X, y = split_features_and_target(prepared)
    print("Prepared data:", prepared.shape)
    print("Model input X:", X.shape, "| target y:", y.shape)
    print("Columns of X:", list(X.columns))
