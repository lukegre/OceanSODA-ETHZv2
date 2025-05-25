import os

import pandas as pd
import pytest
import xarray as xr

from oceansoda_ethzv2.data.utils.zarr_utils import (
    is_data_adjacent_timestep,
    is_data_in_zarr,
    is_data_start_of_year,
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


def test_is_data_in_zarr(tmp_path):
    # Create a dummy dataset
    ds = make_dummy_dataset("2023-01-01")

    # Define a temporary Zarr file path
    zarr_path = os.path.join(tmp_path, "test.zarr")

    # Write the dataset to the Zarr store
    ds.to_zarr(zarr_path)
    assert is_data_in_zarr(ds, zarr_path, append_dim="time") is True

    # Create a new dataset with overlapping time
    ds_overlapping = make_dummy_dataset("2023-01-01")
    assert is_data_in_zarr(ds_overlapping, zarr_path, append_dim="time") is True

    # Create a new dataset with non-overlapping time
    ds_non_overlapping = make_dummy_dataset("2023-01-02")
    assert is_data_in_zarr(ds_non_overlapping, zarr_path, append_dim="time") is False


def test_is_data_adjacent(tmp_path):
    # Create a dummy dataset
    ds = make_dummy_dataset("2023-01-01")

    # Define a temporary Zarr file path
    zarr_path = os.path.join(tmp_path, "test.zarr")

    # Write the dataset to the Zarr store
    ds.to_zarr(zarr_path)

    # Create a new dataset that is adjacent within the tolerance
    ds_adjacent = make_dummy_dataset("2023-01-09")
    assert is_data_adjacent_timestep(ds_adjacent, zarr_path, tolerance="8D") is True

    # Create a new dataset that is not adjacent (outside the tolerance)
    ds_not_adjacent = make_dummy_dataset("2023-01-15")
    assert (
        is_data_adjacent_timestep(ds_not_adjacent, zarr_path, tolerance="8D") is False
    )


def test_is_data_first_write_safe(tmp_path):
    # Create a dummy dataset within the tolerance of the start of the year
    ds_within_threshold = make_dummy_dataset("2023-01-05")

    # Test when the Zarr store does not exist and the dataset is within the tolerance
    assert (
        is_data_start_of_year(ds_within_threshold, append_dim="time", tolerance="8D")
        is True
    )

    # Create a dummy dataset outside the tolerance of the start of the year
    ds_outside_threshold = make_dummy_dataset("2023-01-15")
    assert (
        is_data_start_of_year(ds_outside_threshold, append_dim="time", tolerance="8D")
        is False
    )
