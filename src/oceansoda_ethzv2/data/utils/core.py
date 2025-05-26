from abc import ABC, abstractmethod
from functools import cached_property, lru_cache
from typing import Optional, final

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

    @final
    # @lru_cache(maxsize=8)
    def get_timestep(
        self,
        year: Optional[int] = None,
        index: Optional[int] = None,
        time: Optional[pd.Timestamp | str] = None,
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

        ds = self.get_timestep_unprocessed(year=year, index=index, time=time)
        ds = self._regrid_data(ds)

        return ds

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
        from .checker import TimestepChecker

        if checks == ():
            logger.warning(
                "No validation checks have been specified, please update the config"
            )

        checker = TimestepChecker(
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
    # @lru_cache(8)
    def write_timestep_to_disk(
        self,
        year: int,
        index: int,
        progress: bool = False,
        validate_data: bool = True,
    ) -> xr.Dataset:
        """
        Write a single timestep of the ERA5 dataset to disk.

        Parameters
        ----------
        ds : xr.Dataset
            The xarray Dataset containing the ERA5 data.
        time : pd.Timestamp
            The timestamp for the data to be saved.
        save_path : str, optional
            The path where the data will be saved, defaults to '../data/era5/{time:%Y%m%d}.zarr'.
        """
        from .zarr_utils import save_vars_to_zarrs

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

        return ds

    @abstractmethod
    @lru_cache
    def get_timestep_unprocessed(
        self,
        year: Optional[int] = None,
        index: Optional[int] = None,
        time: Optional[pd.Timestamp | str] = None,
    ) -> xr.Dataset: ...

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
