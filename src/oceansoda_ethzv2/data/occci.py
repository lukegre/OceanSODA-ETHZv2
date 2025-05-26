from functools import lru_cache, partial
from typing import Callable, Optional

import numpy as np
import pandas as pd
import xarray as xr
from icecream import ic as print

from .utils.core import CoreDataset
from .utils.date_utils import DateWindows

OCCCI_URL = "ftp://oceancolour.org/occci-v6.0/geographic/netcdf/8day/chlor_a/{time:%Y}/ESACCI-OC-L3S-CHLOR_A-MERGED-8D_DAILY_4km_GEO_PML_OCx-{time:%Y%m%d}-fv6.0.nc"


class OCCCIDataset(CoreDataset):
    checks = ("fix_timestep", "add_time_bnds", "check_lon_lat")

    def __init__(
        self,
        spatial_res=0.25,
        window_span="8D",
        save_path: str = "../data/{window_span}_{spatial_res}/{{var}}-{window_span}_{spatial_res}.zarr",
        url: str = OCCCI_URL,
        fsspec_kwargs: dict = {},
        vars: dict[str, str] = {},
        **kwargs,
    ):
        """
        Initialize the OCCCIDataset with properties.
        """
        self.url = url
        self.spatial_res = spatial_res  # Default spatial resolution in degrees
        self.window_span = window_span  # Default window size for temporal resolution
        self.fsspec_kwargs = (
            fsspec_kwargs  # Additional fsspec arguments for downloading
        )
        self.vars = vars
        self.save_path = save_path.format(
            window_span=window_span, spatial_res=f"{str(spatial_res).replace('.', '')}"
        )

    @lru_cache
    def get_timestep_unprocessed(
        self,
        year: Optional[int] = None,
        index: Optional[int] = None,
        time: Optional[pd.Timestamp | str] = None,
    ) -> xr.Dataset:
        time = self.date_windows.get_window_center(year=year, index=index, time=time)
        ds = get_occci_timestep(
            time,
            window_span=self.window_span,
            vars_rename=self.vars,
            fsspec_kwargs=self.fsspec_kwargs,
        )

        return ds

    def _regrid_data(self, ds) -> xr.Dataset:
        from .utils.processors import coarsen_then_interp

        time = self.date_windows.get_window_center(time=ds.time.to_index()[0])
        ds = coarsen_then_interp(
            ds, spatial_res=self.spatial_res, window_size=self.window_span
        )
        ds = ds.assign_coords(time=[time])

        return ds


def get_occci_timestep(
    time: pd.Timestamp,
    window_span="8D",
    vars_rename: dict = {
        "chlor_a": "chl_occci",
        "chlor_a_log10_rmsd": "chl_occci_sigma_uncert",
    },
    fsspec_kwargs: dict = {},
):
    """"""
    from .utils.download import download_netcdfs_from_ftp

    urls = make_occci_urls(time, window_span=window_span)
    opener = partial(open_occci_dataset, vars_rename=vars_rename)
    ds_list = download_netcdfs_from_ftp(urls, netcdf_opender=opener, **fsspec_kwargs)

    ds_hr = xr.concat(ds_list, dim="time").chunk({"time": 1, "lat": -1, "lon": -1})
    return ds_hr


def download_netcdfs_from_ftp(urls: tuple, netcdf_opender: Callable, **fsspec_kwargs):
    from functools import partial

    from .utils.download import (
        check_urls_on_ftp_server,
        data_url_data_in_parallel,
        netcdf_tempfile_reader,
    )

    urls_on_ftp_server = check_urls_on_ftp_server(urls, **fsspec_kwargs)
    url_processor = partial(
        netcdf_tempfile_reader, netcdf_opener=netcdf_opender, **fsspec_kwargs
    )

    url_data_in_list = data_url_data_in_parallel(urls_on_ftp_server, url_processor)

    return url_data_in_list


def open_occci_dataset(
    fname: str,
    vars_rename: Optional[dict] = {
        "chlor_a": "chl_occci",
        "chlor_a_log10_rmsd": "chl_occci_sigma_uncert",
    },
):
    from .utils.processors import preprocessor_generic

    ds = xr.open_dataset(fname, chunks={}, decode_timedelta=False)
    ds = preprocessor_generic(
        ds, vars_rename=vars_rename, depth_idx=None, coords_duplicate_check=[]
    )

    return ds


def make_occci_urls(time, window_span="8D"):
    """
    Create URLs for the OCCI dataset based on a given time and window span.

    Args:
        time (pd.Timestamp): The timestamp for which to create the URLs.
        window_span (str): The span of the window, e.g., "8D" for 8 days.

    Returns:
        tuple: A tuple of URLs for the OCCI dataset.
    """
    from .utils.download import make_urls_from_dates

    dates = make_dates_for_window(time=time, window_span=window_span)
    urls = make_urls_from_dates(dates, url_template=OCCCI_URL)
    return urls


def make_dates_for_window(
    time: Optional[pd.Timestamp] = None,
    year: Optional[int] = None,
    index: Optional[int] = None,
    window_span="8D",
) -> pd.DatetimeIndex:
    dw = DateWindows(window_span=window_span)
    year, index = dw.get_index(time=time, year=year, index=index)  # type: ignore

    window_dates = dw.get_window_dates(year=year, index=index)
    return window_dates
