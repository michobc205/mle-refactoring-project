import pandas as pd
import pytest

from cleaning import load_data
from features import get_waterfront_locations
from pipeline import (
    NON_FEATURE_COLUMNS,
    prepare_data,
    run_pipeline,
    split_features_and_target,
)


@pytest.fixture(scope="module")
def prepared():
    return run_pipeline()


def test_pipeline_gives_clean_data_with_all_features(prepared):
    assert prepared.shape == (21596, 23)
    assert prepared.isnull().sum().sum() == 0
    for column in ("last_known_change", "sqft_price", "center_distance", "water_distance"):
        assert column in prepared.columns


def test_pipeline_does_not_change_the_raw_data():
    raw = load_data()
    before = raw.copy()
    prepare_data(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_model_input_has_no_price_columns(prepared):
    X, y = split_features_and_target(prepared)
    for column in NON_FEATURE_COLUMNS:
        assert column not in X.columns
    assert X.shape == (21596, 20)
    assert len(y) == 21596


def test_target_is_the_price(prepared):
    _, y = split_features_and_target(prepared)
    assert y.name == "price"
    assert round(y.mean(), 2) == 540291.96


def test_same_reference_points_give_the_same_features_for_known_houses(prepared):
    # Reuse: the reference points of the training data, applied to the first 100 raw houses.
    locations = get_waterfront_locations(prepared)
    again = prepare_data(load_data().head(100), locations)
    for column in ("center_distance", "water_distance", "last_known_change"):
        pd.testing.assert_series_equal(
            again[column], prepared.loc[again.index, column], check_names=True
        )
