import os

import pandas as pd
import pytest
import xarray as xr

from oceansoda_ethzv2.data.utils.zarr_utils import (
    is_time_adjacent_timestep,
    is_time_first_in_year,
    is_time_in_dataset,
)


def make_dummy_dataset(time):
    """Create a dummy xarray Dataset with a time coordinate."""
    return xr.Dataset(
        {
            "temperature": (("time", "lat", "lon"), [[[1, 2], [3, 4]]]),
        },
        coords={
            "time": [pd.Timestamp(time)],
            "lat": [10, 20],
            "lon": [30, 40],
        },
    )


def test_is_time_in_dataset_true():
    """Test is_time_in_dataset returns True when the time is present in the dataset."""
    time = "2023-01-01"
    ds = make_dummy_dataset(time)
    assert is_time_in_dataset(pd.Timestamp(time), ds) is True


def test_is_time_in_dataset_false():
    """Test is_time_in_dataset returns False when the time is not present in the dataset."""
    time = "2023-01-01"
    ds = make_dummy_dataset(time)
    assert is_time_in_dataset(pd.Timestamp("2023-01-02"), ds) is False


def test_is_time_in_dataset_empty_dataset():
    """Test is_time_in_dataset returns False for an empty dataset."""
    ds = xr.Dataset(coords={"time": [], "lat": [10, 20], "lon": [30, 40]})
    assert is_time_in_dataset(pd.Timestamp("2023-01-01"), ds) is False


def test_is_time_in_dataset_custom_dim_name():
    """Test is_time_in_dataset works with a custom dimension name."""
    time = "2023-01-01"
    ds = xr.Dataset(
        {
            "temperature": (("custom_time", "lat", "lon"), [[[1, 2], [3, 4]]]),
        },
        coords={
            "custom_time": [pd.Timestamp(time)],
            "lat": [10, 20],
            "lon": [30, 40],
        },
    )
    assert is_time_in_dataset(pd.Timestamp(time), ds, dim_name="custom_time") is True
