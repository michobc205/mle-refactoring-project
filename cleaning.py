"""Data cleaning functions for the King County house sales data.

Each function takes a DataFrame and returns a NEW DataFrame.
The input is never changed, so every step is easy to test.
"""
import numpy as np
import pandas as pd

DATA_PATH = "data/King_County_House_prices_dataset.csv"

# A house with more bedrooms than this is a data-entry error.
# In this data set only one row is above it: 33 bedrooms, 2 bathrooms, 1,620 sqft.
MAX_PLAUSIBLE_BEDROOMS = 30


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Read the house sales from the CSV file."""
    return pd.read_csv(path)


def remove_implausible_houses(
    df: pd.DataFrame, max_bedrooms: int = MAX_PLAUSIBLE_BEDROOMS
) -> pd.DataFrame:
    """Drop houses with an impossible number of bedrooms.

    The notebook dropped the row with the fixed label 15856.
    Here we use a condition, so it still works if the row order changes.
    """
    return df.loc[df["bedrooms"] <= max_bedrooms].copy()


def fix_basement_size(df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild `sqft_basement` as `sqft_living - sqft_above`.

    The original column is text, because some rows contain '?'.
    The new column is a number.
    """
    df = df.copy()
    df["sqft_basement"] = df["sqft_living"] - df["sqft_above"]
    return df


def fill_missing_view_and_waterfront(df: pd.DataFrame) -> pd.DataFrame:
    """Replace missing `view` and `waterfront` values with 0 (the most common value)."""
    df = df.copy()
    df[["view", "waterfront"]] = df[["view", "waterfront"]].fillna(0)
    return df


def add_last_known_change(df: pd.DataFrame) -> pd.DataFrame:
    """Combine `yr_built` and `yr_renovated` into `last_known_change`.

    If the house was renovated, use the renovation year.
    If `yr_renovated` is 0 or missing, use the year it was built.
    The two original columns are removed.
    """
    df = df.copy()
    renovated = df["yr_renovated"].notna() & (df["yr_renovated"] != 0)
    df["last_known_change"] = np.where(
        renovated, df["yr_renovated"], df["yr_built"]
    ).astype(int)
    return df.drop(columns=["yr_renovated", "yr_built"])


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run all cleaning steps, in the same order as the notebook."""
    return (
        df.pipe(remove_implausible_houses)
        .pipe(fix_basement_size)
        .pipe(fill_missing_view_and_waterfront)
        .pipe(add_last_known_change)
    )
