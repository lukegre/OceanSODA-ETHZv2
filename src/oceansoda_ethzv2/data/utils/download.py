from functools import lru_cache
from typing import Callable

import pooch
import xarray as xr


class DummyProgress(object):
    """
    Dummy progress bar for when no progress is needed.
    """

    def __init__(self, desc=None):
        self.desc = desc

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def update(self, n=1):
        pass


def download_w_pooch(url, dest=None, progress=True, **kwargs):
    """Downloads url and reads in the MBL surface file

    Args:
        noaa_mbl_url (str): the address for the noaa surface file
        dest (str): the destination to which the raw file will be saved

    Returns:
        pd.Series: multindexed series of xCO2 with (time, lat) as coords.
    """

    pooch.utils.get_logger().disabled = not progress

    # download the file
    filename = pooch.retrieve(
        url,
        known_hash=None,
        path=dest,
        progressbar=progress,
        **kwargs,
    )
    return filename


@lru_cache()
def chech_if_url_exists(url: str) -> bool:
    """Check if the URL exists by pinging it"""
    import requests

    try:
        response = requests.head(url)
        return response.status_code == 200
    except requests.RequestException:
        return False


def check_if_urls_exist(urls: tuple) -> tuple:
    from joblib import Parallel, delayed

    results = Parallel(n_jobs=-1, backend="threading")(
        delayed(chech_if_url_exists)(url) for url in urls
    )
    return tuple(results)


def get_urls_that_exist(urls: tuple) -> tuple:
    """
    Check which URLs exist and return the ones that do.
    """
    urls_exist = check_if_urls_exist(urls)
    url_list = [url for url, exists in zip(urls, urls_exist) if exists]
    return tuple(url_list)


def get_data_from_urls(urls: tuple, url_processor: Callable) -> list:
    import dask.bag as db
    import dask.config

    dask.config.set(scheduler="threads")

    return (
        db.from_sequence(urls, npartitions=len(urls))
        .map(lambda url: url_processor(url) if url else None)
        .filter(lambda ds: ds is not None)
        .compute(scheduler="threads")
    )


@lru_cache(maxsize=16)
def netcdf_url_reader(url: str, netcdf_opener: Callable) -> xr.Dataset:
    """
    Read a netCDF file from a URL.
    """
    import tempfile

    with tempfile.TemporaryDirectory(delete=True) as temp_dir:
        fname = download_w_pooch(url, dest=temp_dir, progress=False)
        ds = netcdf_opener(fname).persist()
        return ds
