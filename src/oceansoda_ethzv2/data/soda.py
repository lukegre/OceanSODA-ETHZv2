import functools
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from .utils.date_utils import DateWindows

SODA_URL = "http://dsrs.atmos.umd.edu/DATA/soda3.15.2/REGRIDED/ocean/soda3.15.2_5dy_ocean_reg_{time:%Y_%m_%d}.nc"


def validate_soda_data(ds, window_span="8D", spatial_res=0.25):
    from .checker import TimestepChecker

    checker = TimestepChecker(spatial_res=spatial_res, time_window=window_span)
    checker.check_lon_lat(ds)
    checker.check_land_points(ds)
    checker.check_missing_lon(ds)

    ds = checker.fix_timestep(ds)
    ds = checker.add_time_bnds(ds)

    return ds


def get_soda_data_for_window(
    time: pd.Timestamp, window_span="8D", spatial_res=0.25
) -> xr.Dataset:
    from .processors import make_target_global_grid

    ds = get_soda_for_extended_window(time=time, window_span="8D")

    dw = DateWindows(window_span="8D")
    t0, t1 = dw.get_window_edges(time)

    target_grid = make_target_global_grid(spatial_res)

    out = (
        ds.resample(time="1D")
        .interpolate(kind="linear")
        .sel(time=slice(t0, t1))
        .coarsen(time=8)
        .mean()
        .interp(**target_grid, method="linear")
        .reindex(**target_grid, method="nearest", tolerance=0.5)
        .roll(lon=360, roll_coords=False)
        .interpolate_na(dim="lon", method="linear", limit=2)
        .roll(lon=-360, roll_coords=False)
        .pipe(validate_soda_data, window_span=window_span, spatial_res=spatial_res)
    )

    return out


def open_soda_file(
    fname,
    vars_rename={
        "xt_ocean": "lon",
        "yt_ocean": "lat",
        "st_ocean": "depth",
        "salt": "sss",
        "ssh": "ssh",
        "mlp": "mld",
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
    from functools import partial

    from .processors import preprocessor_generic

    ds = xr.open_dataset(fname, chunks={}, decode_timedelta=False)
    ds = preprocessor_generic(
        ds, vars_rename=vars_rename, depth_idx=0, coords_duplicate_check=["lat", "lon"]
    )

    return ds


def get_soda_for_extended_window(time: pd.Timestamp, window_span="8D"):
    from .utils.download import (
        get_data_from_urls,
        get_urls_that_exist,
        netcdf_url_reader,
    )

    url = SODA_URL.format(time=time)

    dates = make_dates_for_extended_window(time=time, window_span="8D")
    urls = make_urls_from_dates(dates, url_template=SODA_URL)
    urls = get_urls_that_exist(urls)
    url_processor = functools.partial(netcdf_url_reader, netcdf_opener=open_soda_file)
    out = get_data_from_urls(urls, url_processor)
    ds = xr.concat(out, dim="time")

    return ds


def make_dates_for_extended_window(
    time: Optional[pd.Timestamp] = None,
    year: Optional[int] = None,
    index: Optional[int] = None,
    window_span="8D",
) -> pd.DatetimeIndex:
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


def make_urls_from_dates(dates: pd.DatetimeIndex, url_template=SODA_URL):
    url_list = [url_template.format(time=date) for date in dates]
    return tuple(url_list)
