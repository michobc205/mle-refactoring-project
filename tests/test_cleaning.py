import numpy as np
import pandas as pd
import pytest

from cleaning import (
    add_last_known_change,
    clean_data,
    fill_missing_view_and_waterfront,
    fix_basement_size,
    load_data,
    remove_implausible_houses,
)


def test_remove_implausible_houses_drops_the_33_bedroom_house():
    df = pd.DataFrame({"bedrooms": [3, 33, 4]}, index=[10, 15856, 20])
    result = remove_implausible_houses(df)
    assert list(result["bedrooms"]) == [3, 4]
    assert 15856 not in result.index


def test_fix_basement_size_replaces_the_question_mark():
    df = pd.DataFrame(
        {
            "sqft_living": [2000, 1500],
            "sqft_above": [1500, 1500],
            "sqft_basement": ["500", "?"],
        }
    )
    result = fix_basement_size(df)
    assert list(result["sqft_basement"]) == [500, 0]
    assert pd.api.types.is_numeric_dtype(result["sqft_basement"])


def test_fill_missing_view_and_waterfront_uses_zero():
    df = pd.DataFrame({"view": [np.nan, 3.0], "waterfront": [1.0, np.nan]})
    result = fill_missing_view_and_waterfront(df)
    assert list(result["view"]) == [0.0, 3.0]
    assert list(result["waterfront"]) == [1.0, 0.0]


def test_add_last_known_change_uses_renovation_year_or_build_year():
    df = pd.DataFrame(
        {"yr_built": [1950, 1960, 1970], "yr_renovated": [0.0, 1990.0, np.nan]}
    )
    result = add_last_known_change(df)
    assert list(result["last_known_change"]) == [1950, 1990, 1970]
    assert "yr_built" not in result.columns
    assert "yr_renovated" not in result.columns


def test_cleaning_functions_do_not_change_their_input():
    df = pd.DataFrame({"bedrooms": [3, 33], "view": [np.nan, 1.0], "waterfront": [0.0, np.nan]})
    before = df.copy()
    remove_implausible_houses(df)
    fill_missing_view_and_waterfront(df)
    pd.testing.assert_frame_equal(df, before)


# ---- Tests on the real data: the numbers come from the original notebook ----
@pytest.fixture(scope="module")
def cleaned():
    return clean_data(load_data())


def test_cleaned_data_has_21596_rows(cleaned):
    assert len(cleaned) == 21596


def test_cleaned_data_has_no_missing_values(cleaned):
    assert cleaned.isnull().sum().sum() == 0


def test_cleaned_data_has_numeric_basement_and_new_year_column(cleaned):
    assert pd.api.types.is_numeric_dtype(cleaned["sqft_basement"])
    assert "yr_built" not in cleaned.columns
    assert round(cleaned["last_known_change"].mean(), 3) == 1972.946
