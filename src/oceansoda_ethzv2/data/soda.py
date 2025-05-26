from functools import cached_property, lru_cache
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

from .utils.core import CoreDataset


class SODADataset(CoreDataset):
    checks = (
        "fix_timestep",
        "add_time_bnds",
        "check_lon_lat",
        "check_land_points",
        "check_missing_lon",
    )

    def __init__(
        self,
        spatial_res=0.25,
        window_span="8D",
        save_path: str = "../data/{window_span}_{spatial_res}/{{var}}-{window_span}_{spatial_res}.zarr",
        url="http://dsrs.atmos.umd.edu/DATA/soda3.15.2/REGRIDED/ocean/soda3.15.2_5dy_ocean_reg_{time:%Y_%m_%d}.nc",
        vars={},
        fsspec_kwargs={},
        **kwargs,
    ):
        """
        Initialize the SODADataset with properties.
        """

        self.url_template = url
        self.vars = vars
        self.fsspec_kwargs = fsspec_kwargs
        self.spatial_res = spatial_res  # Default spatial resolution in degrees
        self.window_span = window_span  # Default window size for temporal resolution
        self.save_path = save_path.format(
            window_span=window_span,
            spatial_res=f"{str(spatial_res).replace('.', '')}",
        )

    @lru_cache
    def get_timestep_unprocessed(
        self,
        year: Optional[int] = None,
        index: Optional[int] = None,
        time: Optional[pd.Timestamp | str] = None,
    ) -> xr.Dataset:
        """
        Get a time stride from the SODA dataset without regridding.

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

        Returns
        -------
        xr.Dataset
            The dataset for the specified year and index.
        """
        from .utils.download import (
            data_url_data_in_parallel,
            get_urls_that_exist,
            make_urls_from_dates,
        )

        time = self.date_windows.get_window_center(time=time, year=year, index=index)

        dates = make_dates_for_extended_window(time=time, window_span=self.window_span)
        urls = make_urls_from_dates(dates, url_template=self.url_template)
        urls = get_urls_that_exist(urls)

        logger.debug(
            f"Found {len(urls)} valid URLs over a {len(dates)} day window to"
            f" be clipped down to {self.window_span}"
        )

        ds_list = data_url_data_in_parallel(urls, self._netcdf_reader)
        ds = xr.concat(ds_list, dim="time")
        ds = ds.assign_attrs(requested_time=time)

        return ds

    @cached_property  # using cached_property to avoid recomputing the reader, thus nulling the cache
    def _netcdf_reader(self):
        from functools import partial

        from .utils.download import netcdf_tempfile_reader

        open_soda_custom = partial(open_soda_file, vars_rename=self.vars)
        url_processor = partial(
            netcdf_tempfile_reader, netcdf_opener=open_soda_custom, **self.fsspec_kwargs
        )

        return url_processor

    def _regrid_data(self, ds) -> xr.Dataset:
        from .utils.processors import make_target_global_grid

        time = ds.attrs.pop("requested_time")
        t0, t1 = self.date_windows.get_window_edges(time=time)

        target_grid = make_target_global_grid(self.spatial_res)

        out = (
            ds.resample(time="1D")
            .interpolate(kind="linear")
            .sel(time=slice(t0, t1))
            .coarsen(time=self.date_windows.ndays, boundary="exact")
            .mean()
            .interp(**target_grid, method="linear")
            .reindex(**target_grid, method="nearest", tolerance=0.5)
            # getting rid of lon interp gap - not pretty but works
            .roll(lon=360, roll_coords=False)
            .interpolate_na(dim="lon", method="linear", limit=2)
            .roll(lon=-360, roll_coords=False)
        )

        return out


def open_soda_file(
    fname,
    vars_rename={
        "xt_ocean": "lon",
        "yt_ocean": "lat",
        "st_ocean": "depth",
        "salt": "sss_soda",
        "ssh": "ssh_soda",
        "mlp": "mld_soda",
    },
):
    """
    The preprocessor does everything that can be done on a single
    file (e.g. renaming variables, dropping dimensions, etc.)
    What it does not do, is time-related operations since this
    can only be done with multiple files.

    The reason we use a preporcessor is that the files may be messy
    with inconsistent coordinates (e.g. lon, lat) which makes
    merging fail.

    The downside is that things need to be hardcoded unless we use
    a parital function.
    """
    from .utils.processors import preprocessor_generic

    logger.trace("reading in SODA file: {}", fname)
    ds = xr.open_dataset(fname, chunks={}, decode_timedelta=False)
    ds = preprocessor_generic(
        ds, vars_rename=vars_rename, depth_idx=0, coords_duplicate_check=["lat", "lon"]
    )

    return ds


def make_dates_for_extended_window(
    time: Optional[pd.Timestamp] = None,
    year: Optional[int] = None,
    index: Optional[int] = None,
    window_span="8D",
) -> pd.DatetimeIndex:
    from .utils.date_utils import DateWindows

    dw = DateWindows(window_span=window_span)
    year, index = dw.get_index(time=time, year=year, index=index)  # type: ignore

    window_dates = dw.get_window_dates(year=year, index=index)

    year_index_lower, year_index_upper = dw.get_adjacent(year=year, index=index)

    window_lower = dw.get_window_dates(
        year=year_index_lower[0], index=year_index_lower[1]
    )
    window_upper = dw.get_window_dates(
        year=year_index_upper[0], index=year_index_upper[1]
    )

    w2 = pd.Timedelta(dw.freq).days // 2
    window_plus_wings = np.hstack(
        [
            window_lower[-w2 - 1 :],  # only last half of the window
            window_dates,
            window_upper[:w2],  # only first half of the window
        ]
    ).tolist()

    return pd.DatetimeIndex(window_plus_wings)
