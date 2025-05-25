import functools
import pathlib
from abc import ABC
from typing import Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger


class ZarrDataset(ABC):
    __slots__ = ("entries", "spatial_res", "temporal_res")

    def __init__(self, entries: list[dict], spatial_res: float, temporal_res: str):
        """
        Initialize the ZarrDataset with a list of entries.

        Parameters
        ----------
        entries : list[dict]
            A list of dictionaries containing dataset entries.
        """
        raise NotImplementedError(
            "This class is an abstract base class and should not be instantiated directly."
        )

    def __getitem__(self, key: tuple[int, int]):
        """
        Allow access to dataset properties as attributes.
        """
        if isinstance(key, tuple) and len(key) == 2:
            # If key is a tuple, return the dataset for the specified year and index
            year, index = key
            return self.get_time_stride(year=year, index=index)
        else:
            raise NotImplementedError(
                "Only tuple keys of the form (year, index) are supported for CMEMSDataset."
            )

    def _open_full_dataset(self) -> xr.Dataset:
        """
        Open the full dataset from the Zarr store.

        Returns
        -------
        xr.Dataset
            The full dataset as an xarray Dataset.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    def _regrid_data(self, ds) -> xr.Dataset:
        raise NotImplementedError("This method should be implemented in subclasses.")

    @functools.cached_property
    def data(self) -> xr.Dataset:
        """
        Get the CMEMS dataset as an xarray Dataset.
        This method is a wrapper around the open method.
        """
        return self._open_full_dataset()

    def get_time_stride(
        self,
        year: Optional[int] = None,
        index: Optional[int] = None,
        time: Optional[pd.Timestamp | str] = None,
        freq="8D",
    ):
        from .date_utils import DateWindows

        dw = DateWindows(window_span=freq)

        t0, t1 = dw.get_window_edges(time=time, year=year, index=index)

        assert (t1 - t0) <= pd.Timedelta(freq), (
            "Time range must be at most one frequency apart."
        )

        ds = self.data.sel(time=slice(t0, t1))
        return ds

    def get_time_stride_regridded(
        self,
        year: Optional[int] = None,
        index: Optional[int] = None,
        time: Optional[pd.Timestamp | str] = None,
        freq="8D",
    ) -> xr.Dataset:
        """
        Get a time stride from the dataset and regrid it to the target grid.

        Parameters
        ----------
        year : int, optional
            The year to get the data for.
        index : int, optional
            The index of the data to get.
        time : pd.Timestamp or str, optional
            The time to get the data for.
        freq : str, optional
            The frequency of the data to get.
        target_grid : dict, optional
            The target grid to regrid the data to.

        Returns
        -------
        xr.Dataset
            The regridded dataset.
        """

        ds = self.get_time_stride(year=year, index=index, time=time, freq=freq)
        ds = self._regrid_data(ds)

        return ds


def is_data_timeseries_safe_for_zarr(
    ds_to_write, zarr_filename, append_dim="time", tolerance="8D"
):
    data_is_start_of_year = is_data_start_of_year(
        ds_to_write, append_dim=append_dim, tolerance=tolerance
    )
    zarr_file_exists = pathlib.Path(zarr_filename).exists()

    if not zarr_file_exists and data_is_start_of_year:
        return True
    elif not zarr_file_exists and not data_is_start_of_year:
        raise ValueError(
            f"Dataset not safe for zarr: data is not at the start of the year and writing new file [{zarr_filename}]. "
        )

    data_is_in_zarr = is_data_in_zarr(ds_to_write, zarr_filename, append_dim=append_dim)
    data_is_adjacent = is_data_adjacent_timestep(
        ds_to_write, zarr_filename, tolerance=tolerance
    )

    if data_is_in_zarr:
        raise ValueError(
            f"Dataset not safe for zarr! Data is already in zarr store [{zarr_filename}]."
        )
    elif not data_is_adjacent:
        raise ValueError(
            f"Dataset not safe for zarr: data is not adjacent to existing data in zarr store [{zarr_filename}]."
        )
    else:
        return True


def is_data_in_zarr(ds_to_write, zarr_filename, append_dim="time"):
    """
    Check if the data is already written in the Zarr store.

    This needs a more nuanced approach since append does not
    check if the data is already written, and will simply add
    a duplicate.

    Parameters
    ----------
    ds_to_write : xarray.Dataset
        The dataset to be written to the Zarr store.
    zarr_filename : str
        The path to the Zarr store.

    Returns
    -------
    bool
        True if the data is already written, False otherwise.
    """

    # Open the Zarr store
    ds_zarr = xr.open_zarr(zarr_filename)

    # Check if the coordinates match
    coord_zarr = ds_zarr.coords[append_dim]
    coord_ds = ds_to_write.coords[append_dim]
    if coord_ds.isin(coord_zarr).any():
        return True
    else:
        # print(f"Data is already in Zarr store: {zarr_filename}.")
        return False


def is_data_adjacent_timestep(
    ds_to_write: xr.Dataset, zarr_filename: str, tolerance: str = "8D"
):
    """
    Check if the coordinate along the `append_dim` is adjacent to the
    existing data in the Zarr store.

    This is useful to determine if the data can be appended without
    creating a gap in the time series.

    Parameters
    ----------
    ds_to_write : xarray.Dataset
        The dataset to be checked for adjacency.
    zarr_filename : str
        The path to the Zarr store.
    append_dim : str, optional
        The dimension along which to check adjacency. Default is 'time'.
    tolerance : str, optional
        The tolerance for adjacency check, e.g., '8D' for 8 days. Default is '8D'.

    Returns
    -------
    bool
        True if the data is adjacent, False otherwise.
    """
    # Open the Zarr store
    ds_zarr = xr.open_zarr(zarr_filename)

    assert "time" in ds_zarr.coords, "Zarr store must have a 'time' coordinate."
    assert "time" in ds_to_write.coords, (
        "Dataset to write must have a 'time' coordinate."
    )

    # Get the coordinates along the append dimension
    last_in_zarr = ds_zarr.coords["time"][-1].values
    first_in_ds = ds_to_write.coords["time"][0].values

    dt = pd.Timedelta(tolerance)
    # Check if the difference is within the tolerance
    diff_abs = abs(first_in_ds - last_in_zarr)

    if diff_abs <= dt:
        return True
    else:
        # print(f"Data is not adjacent to existing data in Zarr store: {zarr_filename}.")
        return False


def is_data_start_of_year(ds_to_write: xr.Dataset, append_dim="time", tolerance="8D"):
    """
    Concernting time series, if it is the first step to be written to the file,
    then it must be within the tolerance of the start of any given year

    Parameters
    ----------
    ds_to_write : xarray.Dataset
        The dataset to be checked.
    zarr_filename : str
        The path to the Zarr store.
    append_dim : str, optional
        The dimension along which to check adjacency. Default is 'time'.
    tolerance : str, optional
        The tolerance for adjacency check, e.g., '8D' for 8 days. Default is '8D'.

    Returns
    -------
    bool
        True if the data is the first write, False otherwise.
    """

    first_in_ds = ds_to_write[append_dim][0].values
    first_in_ds = pd.Timestamp(first_in_ds)

    year = first_in_ds.year
    first_of_year = pd.Timestamp(f"{year}-01-01")
    diff = first_in_ds - first_of_year
    dt = pd.Timedelta(tolerance)
    diff_abs = abs(diff)
    if diff_abs < dt:
        return True
    else:
        # print(f"Data is not within the tolerance of the start of the year: {first_in_ds}.")
        return False


def save_to_zarr(ds, fname, append_dim="time", tolerance="8D", progress=False):
    """
    Save the xarray Dataset to a Zarr file.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to be saved.
    fname : str
        The path to the Zarr file.
    """
    if progress:
        from tqdm.dask import TqdmCallback as ProgressBar
    else:
        from .download import DummyProgress as ProgressBar

    from .date_utils import DateWindows

    dw = DateWindows(window_span=tolerance)

    year = np.unique(ds.time.dt.year.values)
    if len(year) != 1:
        raise ValueError(f"Dataset must contain data for a single year, found: {year}")
    group = str(year[0])
    fname = pathlib.Path(fname)
    if not is_data_timeseries_safe_for_zarr(
        ds, fname / group, append_dim="time", tolerance="8D"
    ):
        raise ValueError(f"Data is not safe to write to Zarr store: {fname}")

    # Ensure the directory exists
    fname.parent.mkdir(parents=True, exist_ok=True)

    # Save the dataset to Zarr format
    props = {
        "consolidated": True,
        "mode": "a" if (fname / group).exists() else "w",
        "append_dim": append_dim if (fname / group).exists() else None,
        "compute": True,
        "group": None if group == "" else group,
    }

    time_str = ds.time.dt.strftime("%Y-%m-%d").values[0]
    _, index = dw.get_index(time=time_str)
    name = f"{fname}/{group}:{index}"
    with ProgressBar(desc=f"Saving {name}"):
        ds.to_zarr(fname, **props)
    logger.success(f"Saved to: {name}")


def save_vars_to_zarrs(
    ds: xr.Dataset,
    fname_fmt: str,
    group_by_year: bool = True,
    progress=False,
    error_handling="raise",
):
    assert "{var}" in fname_fmt, "`fname_fmt` must contain a {var} placeholder"

    ds = ds.chunk({"time": 1, "lat": 360, "lon": 360})

    if group_by_year:
        year = np.unique(ds.time.dt.year)
        assert year.size == 1, (
            "When group_by_year=True, then ds may only contain a single year of data"
        )

    for var in ds.data_vars:
        fname = pathlib.Path(fname_fmt.format(var=var))
        fname.parent.mkdir(exist_ok=True, parents=True)

        if error_handling == "raise":
            save_to_zarr(ds[[var]], fname, tolerance="8D", progress=progress)
        else:
            try:
                save_to_zarr(ds[[var]], fname, tolerance="8D", progress=progress)
            except Exception as e:
                if error_handling == "ignore":
                    continue
                elif error_handling == "warn":
                    e = str(e).replace("\n", "")
                    logger.warning(f"Failed to save   {var: <30} {e}")
                else:
                    raise ValueError(
                        f"Invalid error_handling option: {error_handling}. Choose 'raise', 'warn', or 'ignore'."
                    )
