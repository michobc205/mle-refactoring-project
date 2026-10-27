import pandas as pd
import pytest

from cleaning import clean_data, load_data
from features import (
    KM_PER_DEGREE,
    add_center_distance,
    add_features,
    add_sqft_price,
    add_water_distance,
    distance_km,
    get_waterfront_locations,
)


def test_distance_km_is_zero_at_the_reference_point():
    assert distance_km(-122.0, 47.5, -122.0, 47.5) == 0


def test_distance_km_one_degree_of_latitude_is_about_111_km():
    assert distance_km(-122.0, 48.5, -122.0, 47.5) == pytest.approx(KM_PER_DEGREE)
    assert KM_PER_DEGREE == pytest.approx(111.3, abs=0.1)


def test_add_sqft_price_divides_price_by_living_plus_lot():
    df = pd.DataFrame({"price": [300000], "sqft_living": [2000], "sqft_lot": [8000]})
    assert add_sqft_price(df)["sqft_price"].iloc[0] == 30.0


def test_add_center_distance_is_small_near_the_center():
    df = pd.DataFrame({"lat": [47.62774, 47.7], "long": [-122.24194, -122.3]})
    result = add_center_distance(df)
    assert result["center_distance"].iloc[0] < 0.5
    assert result["center_distance"].iloc[1] > result["center_distance"].iloc[0]


def test_water_distance_is_zero_for_a_waterfront_house_and_positive_elsewhere():
    df = pd.DataFrame(
        {"lat": [47.5, 47.6], "long": [-122.2, -122.2], "waterfront": [1.0, 0.0]}
    )
    result = add_water_distance(df, get_waterfront_locations(df))
    assert result["water_distance"].iloc[0] == 0
    assert result["water_distance"].iloc[1] == pytest.approx(0.1 * KM_PER_DEGREE)


def test_feature_functions_do_not_change_their_input():
    df = pd.DataFrame({"price": [1.0], "sqft_living": [1], "sqft_lot": [1]})
    before = df.copy()
    add_sqft_price(df)
    pd.testing.assert_frame_equal(df, before)


# ---- Tests on the real data: the numbers come from the original notebook ----
@pytest.fixture(scope="module")
def featured():
    return add_features(clean_data(load_data()))


def test_featured_data_keeps_all_rows_and_adds_three_columns(featured):
    assert len(featured) == 21596
    for column in ("sqft_price", "center_distance", "water_distance"):
        assert column in featured.columns


def test_sqft_price_matches_the_notebook(featured):
    assert round(featured["sqft_price"].mean(), 4) == 59.7286
    assert featured["sqft_price"].max() == 399.48


def test_center_distance_matches_the_notebook(featured):
    assert round(featured["center_distance"].mean(), 4) == 17.5333
    assert round(featured["center_distance"].max(), 4) == 70.1929


def test_water_distance_matches_the_notebook(featured):
    assert round(featured["water_distance"].mean(), 3) == 5.544
    assert round(featured["water_distance"].max(), 4) == 57.4277
    assert (featured.loc[featured["waterfront"] == 1, "water_distance"] == 0).all()
