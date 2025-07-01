import pathlib
from abc import ABC, abstractmethod
from functools import cached_property, lru_cache
from typing import Literal, Optional, final

import pandas as pd
import xarray as xr
from loguru import logger
from pydantic import BaseModel, Field

from .date_utils import DateWindows

RECENT_TIMESTEP = (pd.Timestamp.now() - pd.DateOffset(days=8)).strftime("%Y-%m-%d")


class DataEntry(BaseModel):
    source: str = Field(
        ...,
        description="Source of the dataset, that can be a URL with placeholders for {time, version, window_span, spatial_res}, or a CMEMS dataset ID. ",
    )
    product: str = Field(..., description="Product name of the dataset, e.g. 'ostia'.")
    vars: dict[str, str] = Field(
        {},
        description="Variables to rename in the dataset, e.g. {'CHL_mean': 'chl_occci'}",
    )
    time_start: str = Field(
        "1982-01-01",
        description="Start time of the dataset. Defaults to 1982-01-01. If provided, will not try to download before this time, instead raise FileNotFoundError if no data is available.",
    )
    time_end: str = Field(
        RECENT_TIMESTEP,
        description="End time of the dataset. Defaults to 8 days ago. If provided, will not try to download after this time, instead raise FileNotFoundError if no data is available.",
    )
    flag: int | None = Field(
        None,
        description="Flag to indicate the dataset label, useful if multiple entries in a dataset. ",
    )
    kwargs: dict[str, str] = Field(
        {},
        description="Additional keyword arguments. Passed to `source` for string formatting and offers extensibility for more advanced use cases (e.g., fsspec_kwargs for remote datasets).",
    )
    checks: tuple[str, ...] = Field(
        ("fix_timestep", "add_time_bnds", "check_lon_lat"),
        description="Checks to perform on the dataset. Defaults to ('fix_timestep', 'add_time_bnds', 'check_lon_lat').",
    )


class DataSet(BaseModel):
    datasets: list[DataEntry] = Field(
        ...,
        description="List of datasets to be used in the DataSet. Each dataset is defined by a DataEntry object.",
    )
    save_path: str = Field(
        "../data/{window_span}_{spatial_res}/{{var}}-{window_span}_{spatial_res}.zarr",
        description="Path to save the dataset, can contain placeholders for {window_span, spatial_res, {var}}.",
    )
    spatial_res: float = Field(
        0.25,
        description="Spatial resolution in degrees, e.g. 0.25 for 0.25 degrees.",
        ge=0.005,
        le=5.0,
    )
    window_span: str = Field(
        "8D", description="Temporal resolution of the dataset, e.g. '8D' for 8 days."
    )


class CoreDataset(ABC):
    spatial_res: float = 0.25
    window_span: str = "8D"
    save_path: str = (
        "../data/{window_span}_{spatial_res}/{{var}}-{window_span}_{spatial_res}.zarr"
    )
    checks: tuple[str, ...] = ()
    source_path: str = ""
    vars: dict[str, str] = {}
    _dim_names: set = {"time", "lat", "lon", "depth"}
    time_start: Optional[str] = None
    time_end: Optional[str] = None

    def _other_repr(self) -> tuple[str, ...]:
        """
        Additional representation for the dataset.
        This method should be implemented by subclasses to provide additional information.
        """
        return ()

    @final
    def __repr__(self):
        repr = (
            f"{self.__class__.__name__}(",
            f"spatial_res={self.spatial_res}, ",
            f"window_span='{self.window_span}', ",
            f"save_path='{self.save_path}', ",
            f"source_path='{self.source_path}', ",
            f"time_start={self.time_start}, ",
            f"time_end={self.time_end}, ",
            f"vars={self.vars}, ",
            f"checks={self.checks}",
        )
        repr += self._other_repr()
        repr += (")",)
        return "\n    ".join(repr)

    @final
    def __getitem__(self, key: tuple[int, int]) -> xr.Dataset:
        """
        Allow access to dataset properties as attributes.
        """
        if isinstance(key, tuple) and len(key) == 2:
            # If key is a tuple, return the dataset for the specified year and index
            year, index = key
            return self.get_timestep(year=year, index=index)
        else:
            raise NotImplementedError(
                "Only tuple keys of the form (year, index) are supported for CMEMSDataset."
            )

    @final
    @property
    def date_windows(self) -> DateWindows:
        """
        Get the date windows for the dataset.

        Returns
        -------
        DateWindows
            The date windows for the dataset.
        """
        return DateWindows(window_span=self.window_span)

    @abstractmethod
    def _get_unprocessed_timestep_local(
        self,
        year: Optional[int] = None,
        index: Optional[int] = None,
        time: Optional[pd.Timestamp | str] = None,
    ) -> xr.Dataset: ...

    @abstractmethod
    def _get_unprocessed_timestep_remote(
        self,
        year: Optional[int] = None,
        index: Optional[int] = None,
        time: Optional[pd.Timestamp | str] = None,
    ) -> xr.Dataset: ...

    @final
    def get_timestep_unprocessed(
        self,
        year: Optional[int] = None,
        index: Optional[int] = None,
        time: Optional[pd.Timestamp | str] = None,
        error_handling: Literal["raise", "warn", "ignore"] = "warn",
    ) -> xr.Dataset:
        """
        Get a time stride from the dataset without regridding it.
        This means that the data is returned in its original form,
        unless the function self._opoener does some preprocessing.

        Parameters
        ----------
        year : int, optional
            The year to get the data for.
        index : int, optional
            The index of the data to get.
        time : pd.Timestamp or str, optional
            The time to get the data for.
        error_handling : Literal['raise', 'warn', 'ignore'], optional
            How to handle errors when the requested timestep is out of bounds.
            'raise' will raise an error, 'warn' will log a warning, and 'ignore'
            will log a debug message.  Defaults to 'warn' so that data can always
            be retrieved, but the user is informed of the issue.

        Returns
        -------
        xr.Dataset
            The unprocessed dataset for the specified timestep.

        Raises
        ------
        ValueError
            1. if the (year+index) or (time) is not provided
            2. if the requested timestep is out of bounds
        FileNotFoundError
            1. when no files were found for the requested timestep and
               `error_handling` is set to "raise".
        """

        in_bounds, message = self._is_in_bounds(year, index, time)
        if not in_bounds:
            if error_handling == "ignore":
                logger.debug(message)
            elif error_handling == "warn":
                logger.warning(message)
            else:
                raise ValueError(message)
            return xr.Dataset()  # return empty dataset if out of bounds

        protocol = get_path_protocol(self.source_path)
        local_source = protocol == "file"
        getter_func = (
            self._get_unprocessed_timestep_local
            if local_source
            else self._get_unprocessed_timestep_remote
        )

        logger.debug(
            f"Dataset source {protocol=}, using function self.{getter_func.__name__} to get timestep"
        )

        ds = getter_func(year=year, index=index, time=time)

        return ds

    @final
    @lru_cache(maxsize=8)
    def get_timestep(
        self,
        year: Optional[int] = None,
        index: Optional[int] = None,
        time: Optional[pd.Timestamp | str] = None,
        file_exists_handling: Literal["raise", "warn", "ignore"] = "warn",
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

        Raises
        ------
        ValueError
            If the (year+index) or (time) is not provided, OR
            if the requested timestep is out of bounds, OR
            no data was found for the requested timestep
        FileNotFoundError
            If no files were found for the requested timestep
        FileExistsError
            If the timestep is already stored and `file_exists_handling` is set to "raise".
        """

        self.is_timestep_write_safe(year, index, time, handling=file_exists_handling)

        ds = self.get_timestep_unprocessed(
            year=year, index=index, time=time, error_handling="raise"
        )
        ds = self._regrid_data(ds)

        return ds

    @final
    def _is_in_bounds(
        self,
        year: Optional[int] = None,
        index: Optional[int] = None,
        time: Optional[pd.Timestamp | str] = None,
    ) -> tuple[bool, str]:
        """Check if the given year, index, or time is within the bounds of the dataset.
        Uses `self.time_start` and `self.time_end` to determine the bounds.
        """

        t0, t1 = self.date_windows.get_window_edges(time, year, index)
        time = self.date_windows.get_window_center(year=year, index=index, time=time)
        time_end = self.time_end
        time_start = self.time_start

        # if time start and time end are not defined, use default values
        if time_start is None:
            logger.debug(
                "Dataset does not have a time_start defined - setting to 1970-01-01."
            )
            time_start = pd.Timestamp("1970-01-01")
        elif isinstance(time_start, str):
            time_start = pd.Timestamp(time_start)

        if time_end is None:
            time_end = pd.Timestamp.now() - pd.DateOffset(days=8)
            logger.debug(
                f"Dataset does not have a time_end defined - setting to {time_end}."
            )
        elif isinstance(time_end, str):
            time_end = pd.Timestamp(time_end)

        if t0 < time_start:
            in_bounds = False
            message = (
                f"Requested time {time} is before the dataset start time {time_start}"
            )
        elif t1 > time_end:
            in_bounds = False
            message = f"Requested time {time} is after the dataset end time {time_end}"
        else:
            message = f"Dataset is within bounds ({time_start} - {time_end})"
            in_bounds = True

        return in_bounds, message

    @abstractmethod
    def _regrid_data(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Regrid the dataset to the target grid.

        Parameters
        ----------
        ds : xr.Dataset
            The dataset to regrid.

        Returns
        -------
        xr.Dataset
            The regridded dataset.
        """
        ...

        """
        Validate the dataset to ensure it meets the expected properties.

        Parameters
        ----------
        ds : xr.Dataset
            The dataset to validate.
        spatial_res : float
            The expected spatial resolution.
        window_span : str
            The expected temporal resolution.

        Returns
        -------
        bool
            True if the dataset is valid, False otherwise.
        """
        ...

    @final
    def validate_timestep(
        self,
        ds: xr.Dataset,
        checks: tuple[str, ...] = ("fix_timestep", "add_time_bnds", "check_lon_lat"),
    ) -> xr.Dataset:
        """
        Validate the dataset to ensure it meets the expected properties.

        Parameters
        ----------
        ds : xr.Dataset
            The dataset to validate.

        Returns
        -------
        bool
            True if the dataset is valid, False otherwise.
        """
        from .checker import TimestepValidator

        if checks == ():
            logger.warning(
                "No validation checks have been specified, please update the config"
            )

        checker = TimestepValidator(
            spatial_res=self.spatial_res, time_window=self.window_span
        )

        for check in checks:
            func = getattr(checker, check, None)
            if func is None:
                raise ValueError(
                    f"Unknown checking function: {check}, possible values are {checker.__dir__()}"
                )
            else:
                ds = func(ds)

        return ds

    @final
    def write_timestep_to_disk(
        self,
        year: int,
        index: int,
        progress: bool = False,
        validate_data: bool = True,
        file_exists_handling: Literal["raise", "warn", "ignore"] = "warn",
    ) -> None:
        """
        Write a specific timestep to disk in Zarr format.

        Parameters
        ----------
        year : int
            The year of the timestep. Must also pass index
        index : int
            The index of the timestep (index of the sepcific year)
        progress : bool, optional
            Whether to show a progress bar during saving. Defaults to False.
        validate_data : bool, optional
            Whether to validate the data before saving. Defaults to True.
        file_exists_handling : Literal['raise', 'warn', 'ignore'], optional
            How to handle the case where the file already exists.
            'raise' will raise a FileExistsError, 'warn' will log a warning, and 'ignore' will log a debug message.
            Defaults to 'warn'.

        Returns
        -------
        None
            The function does not return anything, it saves the dataset to disk.

        Raises
        ------
        ValueError
            If the year or index is not provided.

        """
        from .zarr_utils import save_vars_to_zarrs

        write_safe = self.is_timestep_write_safe(
            year, index, handling=file_exists_handling
        )

        if write_safe:
            ds = self.get_timestep(year=year, index=index)

            if validate_data:
                ds = self.validate_timestep(ds, self.checks)

            save_vars_to_zarrs(
                ds,
                self.save_path,
                group_by_year=True,
                progress=progress,
                error_handling="warn",
            )

    @final
    def _get_save_path(self, var: str) -> str:
        """
        Get the save path for a specific timestep.

        Parameters
        ----------
        year : int
            The year of the timestep.
        index : int
            The index of the timestep.
        time : pd.Timestamp or str, optional
            The time of the timestep.

        Returns
        -------
        str
            The formatted save path.
        """
        return self.save_path.format(
            var=var,
            window_span=self.window_span,
            spatial_res=f"{str(self.spatial_res).replace('.', '')}",
        )

    @final
    def _is_var_timestep_write_safe(
        self,
        var: str,
        year: Optional[int] = None,
        index: Optional[int] = None,
        time: Optional[pd.Timestamp | str] = None,
    ) -> tuple[bool, str]:
        from .zarr_utils import is_time_safe_for_zarr

        # can't overwrite year, index else raises error in get_window_center
        year_, _ = self.date_windows.get_index(year=year, index=index, time=time)
        sname = str(pathlib.Path(self._get_save_path(var)) / str(year_))

        time = self.date_windows.get_window_center(year=year, index=index, time=time)

        safe_for_zarr, message = is_time_safe_for_zarr(
            time, sname, window_span=self.window_span
        )
        return safe_for_zarr, message

    @final
    def is_timestep_write_safe(
        self,
        year: Optional[int] = None,
        index: Optional[int] = None,
        time: Optional[pd.Timestamp | str] = None,
        handling: Literal["raise", "warn", "ignore"] = "warn",
    ) -> bool:
        """
        Check if a timestep is safe to write to disk.

        To be write-safe, it must adhere to the timestemp:
        - Must not already exist in the dataset
        - Must be in the future relative to the last timestep in the storage location
        - Must be a maximum of `window_span` days old relative to the last timestep in the storage location
        - Must be the first timestep of the year if the file does not exist

        Parameters
        ----------
        var : str
            The variable to check.
        year : int, optional
            The year to check.
        index : int, optional
            The index of the data to check.
        time : pd.Timestamp or str, optional
            The time to check.
        handling : Literal['raise', 'warn', 'ignore'], optional
            How to handle the case where the timestep is not write-safe.
            'raise' will raise a FileExistsError, 'warn' will log a warning, and 'ignore' will log a debug message.
            Defaults to 'raise'.

        Returns
        -------
        bool
            True if the timestep is write-safe, False otherwise.

        """
        vars = set(list(self.vars.values()))
        vars = tuple(vars - self._dim_names)

        write_safe = ()
        message = ""
        for key in vars:
            safe, message = self._is_var_timestep_write_safe(
                var=key, year=year, index=index, time=time
            )
            write_safe += (safe,)

        if not all(write_safe):
            match handling:
                case "ignore":
                    logger.trace(message)
                    return False
                case "warn":
                    logger.warning(message)
                    return False
                case "raise":
                    raise FileExistsError(message)
                case _:
                    raise ValueError(
                        f"Unknown handling method: {handling}, possible values are 'warn', 'raise', 'ignore'"
                    )
        else:
            return True


def get_path_protocol(path: str):
    """
    Get the protocol of a path.

    Parameters
    ----------
    path : str
        The path to get the protocol for.

    Returns
    -------
    str
        The protocol of the path.
    """
    if "://" in path:
        return path.split("://")[0]
    elif "/" not in path:
        return "cmems_id"
    else:
        return "file"
