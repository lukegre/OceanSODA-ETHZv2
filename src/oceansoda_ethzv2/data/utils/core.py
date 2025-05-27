import pathlib
from abc import ABC, abstractmethod
from functools import cached_property, lru_cache
from typing import Literal, Optional, final

import pandas as pd
import xarray as xr
from loguru import logger

from .date_utils import DateWindows


class CoreDataset(ABC):
    spatial_res: float = 0.25
    window_span: str = "8D"
    save_path: str = (
        "../data/{window_span}_{spatial_res}/{{var}}-{window_span}_{spatial_res}.zarr"
    )
    checks: tuple[str, ...] = ()
    local_source: bool = False
    sourece_path: str = ""
    vars: dict[str, str] = {}
    _dim_names: set = {"time", "lat", "lon", "depth"}

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
        file_exists_handling: Literal["raise", "warn", "ignore"] = "warn",
    ) -> xr.Dataset:
        self.is_timestep_write_safe(
            year=year, index=index, time=time, handling=file_exists_handling
        )

        if self.local_source:
            ds = self._get_unprocessed_timestep_local(year=year, index=index, time=time)
        else:
            ds = self._get_unprocessed_timestep_remote(
                year=year, index=index, time=time
            )

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
            If the (year+index) or (time) is not provided.
        FileExistsError
            If the timestep is already stored.
        """

        self.is_timestep_write_safe(year, index, time, handling=file_exists_handling)

        ds = self.get_timestep_unprocessed(
            year=year, index=index, time=time, file_exists_handling="ignore"
        )
        ds = self._regrid_data(ds)

        return ds

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

        year, index = self.date_windows.get_index(year=year, index=index, time=time)
        sname = str(pathlib.Path(self._get_save_path(var)) / str(year))

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
            safe, message = self._is_var_timestep_write_safe(key, year, index, time)
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
