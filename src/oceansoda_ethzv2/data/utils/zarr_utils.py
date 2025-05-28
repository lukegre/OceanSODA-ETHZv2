import pathlib
from abc import ABC, abstractmethod
from functools import cached_property, lru_cache
from typing import Callable, Literal, Optional, Union, final

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger
from tqdm.dask import TqdmCallback

from .core import CoreDataset
from .download import DummyProgress


class ZarrDataset(CoreDataset, ABC):
    @final
    def __init__(
        self,
        entries: list[dict],
        spatial_res: float = 0.25,
        temporal_res: str = "8D",
        save_path: str = "../data/{temporal_res}_{spatial_res}/{{var}}-{temporal_res}_{spatial_res}.zarr",
    ):
        if not isinstance(entries, list):
            raise TypeError(
                f"The input key with the following values is invalid {str(entries)}"
            )

        self.entries = self._process_entries(entries)

        self.spatial_res = spatial_res  # Default spatial resolution in degrees
        self.temporal_res = temporal_res  # Default window size for temporal resolution
        self.save_path = save_path.format(
            temporal_res=temporal_res,
            spatial_res=f"{str(spatial_res).replace('.', '')}",
        )

    def _process_entries(self, entries: list[dict]) -> list[dict]:
        """
        Process the entries to ensure they are in the correct format.
        This method can be overridden if additional processing is needed.
        """
        for entry in entries:
            self._validate_entry(entry)
        return entries

    @abstractmethod
    def _validate_entry(self, entry: dict): ...

    @abstractmethod
    @lru_cache(maxsize=1)
    def _open_full_dataset(self) -> xr.Dataset: ...

    @abstractmethod
    def _regrid_data(self, ds: xr.Dataset) -> xr.Dataset: ...

    @final
    @property
    def full_dataset(self) -> xr.Dataset:
        """
        Get the CMEMS dataset as an xarray Dataset.
        This method is a wrapper around the open method.
        """
        return self._open_full_dataset()

    @final
    def _get_unprocessed_timestep_remote(
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

        ds = self.full_dataset.sel(time=slice(t0, t1))
        return ds


def is_time_safe_for_zarr(
    time: pd.Timestamp, zarr_filename: str, dim_name="time", window_span: str = "8D"
) -> tuple[bool, str]:
    """
    Check if a given time is safe to write to the Zarr store.

    Parameters
    ----------
    time : pd.Timestamp
        The time to check.
    zarr_filename : str
        The path to the Zarr store.
    dim_name : str, optional
        The name of the dimension to check. Default is 'time'.
    window_span : str, optional
        The window_span for adjacency check, e.g., '8D' for 8 days. Default is '8D'.

    Returns
    -------
    bool
        True if the time is safe to write, False otherwise.
    str
        A message indicating the status of the time check.
    """
    from .date_utils import DateWindows

    year, index = DateWindows(window_span=window_span).get_index(time=time)

    messages = {
        "exists": f"Given timestep [{year=}, {index=}] already exists in {zarr_filename}",
        "not_adjacent": f"Given timestep [{year=}, {index=}] but expected index [{{expected_index}}] in {zarr_filename}",
        "not_first": f"File {zarr_filename} does not exist, but the timestep [{year=}, {index=}] is not the first step in the year",
    }

    if not pathlib.Path(zarr_filename).exists():
        if is_time_first_in_year(time, window_span):
            return True, ""
        else:
            return False, messages["not_first"]

    ds = open_cached_zarr(zarr_filename)

    if is_time_in_dataset(time, ds, dim_name):
        return False, messages["exists"]
    if not is_time_adjacent_timestep(time, ds, dim_name, window_span):
        zarr_n_timesteps = ds[dim_name].size
        return False, messages["not_adjacent"].format(expected_index=zarr_n_timesteps)

    # finally if all checks passed, return True
    return True, ""


def is_time_in_dataset(
    time: pd.Timestamp, ds: xr.Dataset, dim_name: str = "time"
) -> bool:
    """
    Check if a given time is present in the dataset along the specified dimension.

    Parameters
    ----------
    time : pd.Timestamp
        The time to check.
    ds : xr.Dataset
        The dataset to check against.
    dim_name : str, optional
        The name of the dimension to check. Default is 'time'.

    Returns
    -------
    bool
        True if the time is present in the dataset, False otherwise.
    """
    ds_time = ds.coords[dim_name].to_index()
    if any(ds_time == time):
        return True
    else:
        return False


def is_time_adjacent_timestep(
    time: pd.Timestamp, ds: xr.Dataset, dim_name="time", window_span="8D"
) -> bool:
    """
    Check if a given time is adjacent to the existing data in the dataset along the specified dimension.

    Parameters
    ----------
    time : pd.Timestamp
        The time to check.
    ds : xr.Dataset
        The dataset to check against.
    dim_name : str, optional
        The name of the dimension to check. Default is 'time'.
    window_span : str, optional
        The window_span for adjacency check, e.g., '8D' for 8 days. Default is '8D'.

    Returns
    -------
    bool
        True if the time is adjacent, False otherwise.
    """

    ds_time = ds.coords[dim_name].to_index()

    if len(ds_time) == 0:
        return True  # If no data exists, any time is considered adjacent

    last_in_ds = ds_time[-1]

    if time <= last_in_ds:
        return False  # Time must be after the last time in the dataset

    diff_abs = abs(time - last_in_ds)

    dt = pd.Timedelta(window_span)
    return (
        diff_abs <= dt
    )  # time lies within the window_span of the last time in the dataset


def is_time_first_in_year(time: pd.Timestamp, window_span: str = "8D") -> bool:
    """
    Check if the given time is the first time step of the year within a specified window_span.

    Parameters
    ----------
    time : pd.Timestamp
        The time to check.
    window_span : str, optional
        The window_span for adjacency check, e.g., '8D' for 8 days. Default is '8D'.

    Returns
    -------
    bool
        True if the time is the first time step of the year within the window_span, False otherwise.
    """

    year = time.year
    first_of_year = pd.Timestamp(f"{year}-01-01")
    first_window = first_of_year + pd.Timedelta(window_span)

    if time < first_of_year:
        logger.critical(
            "Something very wrong has happened. Date is less than Jan 1st of the year."
        )
        return False
    if time >= first_window:
        return False
    else:
        return True


@lru_cache(maxsize=1)
def open_cached_zarr(fname: Union[str, pathlib.Path]) -> xr.Dataset:
    """
    Open a Zarr file, caching the result for future calls.

    Parameters
    ----------
    fname : str or pathlib.Path
        The path to the Zarr file.

    Returns
    -------
    xr.Dataset
        The opened xarray Dataset.
    """
    return xr.open_zarr(fname, consolidated=False)


def _check_single_year(ds: xr.Dataset):
    year = np.unique(ds.time.dt.year.values)
    if len(year) != 1:
        raise ValueError(f"Dataset must contain data for a single year, found: {year}")


def _check_single_timestep(ds: xr.Dataset) -> None:
    time_index = ds.time.to_index()
    if len(time_index) != 1:
        raise ValueError(
            f"Dataset must contain a single time step, found: {len(time_index)}"
        )


def save_to_zarr(
    ds: xr.Dataset,
    zarr_filename: str,
    append_dim: str = "time",
    window_span: str = "8D",
    progress: bool = False,
    error_handling: Literal["raise", "warn", "ignore"] = "raise",
) -> None:
    """
    Save the xarray Dataset to a Zarr file.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to be saved.
    fname : str
        The path to the Zarr file.
    """
    ProgressBar = TqdmCallback if progress else DummyProgress

    _check_single_year(ds)
    _check_single_timestep(ds)

    time = ds.time.to_index()[0]

    group = str(time.year)
    fname = pathlib.Path(zarr_filename)
    fname_group = fname / group

    safe_for_zarr, message = is_time_safe_for_zarr(
        time, str(fname_group), window_span=window_span
    )

    if not safe_for_zarr:
        match error_handling:
            case "raise":
                raise ValueError(message)
            case "warn":
                logger.warning(f"Skipping saving: {message}")
                return
            case "ignore":
                logger.trace(f"Skipping saving {fname} due to: {message}")
                return

    fname.parent.mkdir(parents=True, exist_ok=True)
    # Save the dataset to Zarr format
    name = f"{fname}/{group}:{time:%Y-%m-%d %H:%M}"
    with ProgressBar(desc=f"Saving {name}"):
        logger.info(f"Starting to write {fname} for time {time}")
        ds.to_zarr(
            store=fname,
            mode="a" if (fname_group).exists() else "w",
            append_dim=append_dim if (fname_group).exists() else None,
            group=None if group == "" else group,
        )

    logger.success(f"Saved to: {name}")


def save_vars_to_zarrs(
    ds: xr.Dataset,
    fname_fmt: str,
    group_by_year: bool = True,
    progress=False,
    window_span="8D",
    error_handling: Literal["raise", "warn", "ignore"] = "raise",
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
        ds_var = ds[[var]]
        save_to_zarr(
            ds=ds_var,
            zarr_filename=str(fname),
            window_span=window_span,
            progress=progress,
            error_handling=error_handling,
        )


def open_zarr_groups(
    zarr_root, concat_dim="time", group_validator: Callable | None = None
):
    """
    Finds all groups in a Zarr root, opens them, and concatenates them along the specified dimension.

    Parameters:
        zarr_root (str): Path to the Zarr root directory.
        concat_dim (str): Dimension along which to concatenate the groups.

    Returns:
        xarray.Dataset: Concatenated dataset.
    """
    import zarr

    # Find all groups in the Zarr root
    store = zarr.open(zarr_root, mode="r")
    groups = list(store.group_keys())

    # Open all groups as datasets
    datasets = []
    for group in groups:
        ds = xr.open_zarr(f"{zarr_root}/{group}", consolidated=False)
        if group_validator is not None:
            group_validator(ds)
        datasets.append(ds)

    # Concatenate datasets along the specified dimension
    concatenated_ds = xr.concat(datasets, dim=concat_dim)

    return concatenated_ds
