from functools import cached_property, partial
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

from .utils.core import CoreDataset
from .utils.date_utils import DateWindows


class HTTPDailyDataset(CoreDataset):
    checks = ("fix_timestep", "add_time_bnds", "check_lon_lat")

    def __init__(
        self,
        spatial_res=0.25,
        window_span="8D",
        save_path: str = "../data/{window_span}_{spatial_res}/{{var}}-{window_span}_{spatial_res}.zarr",
        source_path: str = "https://data.ceda.ac.uk/neodc/esacci/sstcci/{version}/sstcci_{time:%Y%m%d}_{spatial_res}deg.nc",
        vars: dict[str, str] = {},
        version: dict[str, dict] = {},
        fsspec_kwargs: dict = {},
        **kwargs,
    ) -> None:
        """
        Initialize the CEDADataset with properties.

        Parameters
        ----------
        spatial_res : float, optional
            The spatial resolution in degrees, by default 0.25.
        window_span : str, optional
            The temporal resolution of the dataset, by default '8D'.
        save_path : str, optional
            The path to save the dataset, by default '../data/{window_span}_{spatial_res}/{{var}}-{window_span}_{spatial_res}.zarr'.
        source_path : str, optional
            The source URL for the dataset, by default 'https://data.ceda.ac.uk/neodc/esacci/sstcci/{version}/sstcci_{time:%Y%m%d}_{spatial_res}deg.nc'.
        """

        self.source_path = source_path
        self.vars = vars
        self.version = version
        self.fsspec_kwargs = fsspec_kwargs
        self.spatial_res = spatial_res
        self.window_span = window_span
        self.save_path = save_path.format(
            window_span=window_span,
            spatial_res=f"{str(spatial_res).replace('.', '')}",
        )

    def _get_unprocessed_timestep_remote(
        self,
        year: Optional[int] = None,
        index: Optional[int] = None,
        time: Optional[pd.Timestamp | str] = None,
    ) -> xr.Dataset:
        from .utils.download import (
            data_url_data_in_parallel,
            get_urls_that_exist,
        )

        dates = self.date_windows.get_window_dates(year=year, index=index, time=time)
        urls = tuple([self._make_url(date) for date in dates])
        urls = get_urls_that_exist(urls)

        logger.debug(
            f"Found {len(urls)} valid URLs over a {len(dates)} day window to"
            f" be clipped down to {self.window_span}"
        )

        ds_list = data_url_data_in_parallel(urls, self._remote_netcdf_reader)
        ds = xr.concat(ds_list, dim="time")
        ds = ds.assign_attrs(requested_time=time)

        return ds

    @cached_property  # using cached_property to avoid recomputing the reader, thus nulling the cache
    def _remote_netcdf_reader(self):
        from .utils.download import netcdf_tempfile_reader

        url_processor = partial(
            netcdf_tempfile_reader, netcdf_opener=self._opener, **self.fsspec_kwargs
        )

        return url_processor

    def _opener(self, fname) -> xr.Dataset:
        from .utils.processors import preprocessor_generic

        ds = xr.open_dataset(fname, chunks={}, engine="h5netcdf")
        ds = preprocessor_generic(ds, vars_rename=self.vars)
        return ds

    def _make_url(self, time: pd.Timestamp) -> str:
        """
        Generate the URL for the SSTCCI dataset based on the provided time.

        Parameters
        ----------
        time : pd.Timestamp
            The timestamp for which to generate the URL.

        Returns
        -------
        str
            The URL for the specified time.
        """
        version = get_ceda_version_from_time(time, self.version)
        return self.source_path.format(
            time=time, version=version, spatial_res=self.spatial_res
        )

    def _regrid_data(self, ds: xr.Dataset) -> xr.Dataset:
        from .utils.processors import coarsen_then_interp

        time = self.date_windows.get_window_center(time=ds.time.to_index()[0])
        ds = coarsen_then_interp(
            ds, spatial_res=self.spatial_res, window_size=self.window_span
        )
        ds = ds.assign_coords(time=[time])

        return ds


def get_ceda_version_from_time(time: pd.Timestamp, versions: dict[str, dict]) -> str:
    """
    Determine the version of the SSTCCI dataset based on the provided time.

    Parameters
    ----------
    time : pd.Timestamp
        The timestamp for which to determine the version.

    Returns
    -------
    str
        The version of the dataset ('CDR3.0' or 'ICDR3.0').
    """

    if versions == {}:
        logger.trace("No versions provided. Returning an empty string.")
        return ""

    for version, date in versions.items():
        t0 = pd.Timestamp(date["time_start"])
        t1 = pd.Timestamp(date["time_end"])

        if t0 <= time <= t1:
            return version

    raise ValueError(
        f"No valid version found for time {time}. Valid versions are: {list(versions.keys())}."
    )
