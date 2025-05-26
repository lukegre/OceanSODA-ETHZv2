import numpy as np
import pandas as pd
import xarray as xr

from .utils.date_utils import DateWindows

SSTCCI_URL = "https://dap.ceda.ac.uk/neodc/eocis/data/global_and_regional/sea_surface_temperature/CDR_v3/Analysis/L4/v3.0.1/{time:%Y}/{time:%m}/{time:%d}/{time:%Y%m%d}120000-ESACCI-L4_GHRSST-SSTdepth-OSTIA-GLOB_{version}-v02.0-fv01.0.nc"
VALID_VERSIONS = {
    "CDR3.0": {"time_start": "1982-01-01", "time_end": "2021-12-31"},
    "ICDR3.0": {"time_start": "2022-01-01", "time_end": "2024-06-22"},
}


def get_sstcci_timestep(
    time: pd.Timestamp, url: str, versions: dict, window_span="8D", spatial_res=0.25
) -> xr.Dataset:
    """
    Get a specific timestep from the SSTCCI dataset.

    Parameters
    ----------
    time : pd.Timestamp
        The timestamp for which to retrieve the data.
    window_span : str, optional
        The temporal resolution of the dataset, by default '8D'.
    spatial_res : float, optional
        The spatial resolution in degrees, by default 0.25.

    Returns
    -------
    xr.Dataset
        The dataset for the specified timestep.
    """
    from .utils.download import (
        data_url_data_in_parallel,
    )

    urls = make_sstcci_urls(
        time, url_fmt=url, versions=versions, window_span=window_span
    )


def ceda_netcdf_opener(fname, var_renmaes={}): ...


def make_sstcci_urls(
    time: pd.Timestamp, url_fmt: str, versions: dict, window_span="8D"
) -> tuple[str, ...]:
    """
    Generate URLs for the SSTCCI dataset based on the provided time and version.

    Parameters
    ----------
    time : pd.Timestamp
        The timestamp for which to generate the URLs.
    version : str, optional
        The version of the dataset, by default 'CDR3.0'.

    Returns
    -------
    tuple[str, ...]
        A tuple of URLs for the specified time and version.
    """
    from .utils.download import check_if_urls_exist

    version = get_version_from_time(time, versions)

    dw = DateWindows(window_span=window_span)
    dates = dw.get_window_dates(time=time)

    urls = tuple([url_fmt.format(time=t, version=version) for t in dates])
    urls = check_if_urls_exist(urls)

    return tuple(urls)


def get_version_from_time(time: pd.Timestamp, versions: dict[str, dict]) -> str:
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

    for version, date in versions.items():
        t0 = pd.Timestamp(date["time_start"])
        t1 = pd.Timestamp(date["time_end"])

        if t0 <= time <= t1:
            return version

    raise ValueError(
        f"No valid version found for time {time}. Valid versions are: {list(versions.keys())}."
    )
