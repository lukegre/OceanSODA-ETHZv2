from typing import Callable, Optional

import numpy as np
import pandas as pd
import pooch
import xarray as xr
from loguru import logger
from memoization import cached
from tqdm.dask import TqdmCallback


class DummyProgress(object):
    """
    Dummy progress bar for when no progress is needed.
    """

    def __init__(self, *args, **kwargs): ...

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass


def check_urls_on_ftp_server(urls: tuple, **kwargs) -> tuple:
    flist = list_ftp_files_with_fsspec(urls, **kwargs)
    urls_mask = np.isin([u.split("/")[-1] for u in urls], flist)
    urls_on_ftp_server = tuple(np.array(urls)[urls_mask].tolist())
    return urls_on_ftp_server


def list_ftp_files_with_fsspec(urls, **kwargs):
    import fsspec

    fs, url = fsspec.url_to_fs(urls[0], **kwargs)
    dir_list = ["/".join(url.rstrip("/").split("/")[:-1]) for url in urls]
    dir_list = list(set(dir_list))  # Remove duplicates

    flist = []
    for directory in dir_list:
        flist += fs.listdir(directory, detail=False)

    return flist


def download_w_fsspec(url, dest=None, **kwargs):
    import fsspec

    if dest is not None and "cache_storage" not in kwargs:
        kwargs["cache_storage"] = dest
        kwargs["same_names"] = True

    url = f"filecache::{url}"
    fs, url = fsspec.url_to_fs(url, **kwargs)
    name = url.split("/")[-1]
    dest = f"{dest}/{name}"

    fs.get(url, dest)
    return dest


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


def check_if_url_exists(url: str) -> bool:
    """Check if the URL exists by pinging it"""
    import requests

    try:
        response = requests.head(url)
        return response.status_code == 200
    except requests.RequestException:
        return False


def check_if_urls_exist(urls: tuple) -> tuple:
    logger.trace("Checking if URLs exist: {}", urls)
    return parmap_func(check_if_url_exists, urls)


def get_urls_that_exist(urls: tuple) -> tuple[str]:
    """
    Check which URLs exist and return the ones that do.
    """
    urls_exist = check_if_urls_exist(urls)
    url_list = [url for url, exists in zip(urls, urls_exist) if exists]
    return tuple(url_list)


def data_url_data_in_parallel(
    urls: tuple[str], url_processor: Callable, progress=False
) -> list:
    ProgressBar = TqdmCallback if progress else DummyProgress

    with ProgressBar(desc=f"Downloading {len(urls)} files"):
        return parmap_func(url_processor, urls)


@cached(max_size=8)
def netcdf_tempfile_reader(
    url: str, netcdf_opener: Callable, **fsspec_kwargs
) -> xr.Dataset:
    """
    Read a netCDF file from a URL.
    """
    import tempfile

    with tempfile.TemporaryDirectory(delete=True) as temp_dir:
        fname = download_w_fsspec(url, dest=temp_dir, **fsspec_kwargs)
        ds = netcdf_opener(fname).persist()
        return ds


def download_netcdfs_from_ftp(
    urls: tuple[str, ...], netcdf_opender: Callable, **fsspec_kwargs
):
    from functools import partial

    urls_on_ftp_server = check_urls_on_ftp_server(urls, **fsspec_kwargs)
    url_processor = partial(
        netcdf_tempfile_reader, netcdf_opener=netcdf_opender, **fsspec_kwargs
    )
    url_data_in_list = data_url_data_in_parallel(urls_on_ftp_server, url_processor)

    return url_data_in_list


def make_paths_from_dates(
    dates: pd.DatetimeIndex, string_template: str
) -> tuple[str, ...]:
    path_list = [(string_template.format(time=date)) for date in dates]
    return tuple(path_list)


def parmap_func(
    func, iterable, n_jobs: Optional[int] = None, sheduler="threads", **kwargs
):
    """
    Apply a function in parallel to an iterable.
    """
    from dask import bag as db

    if n_jobs is None or n_jobs > len(iterable):
        n_jobs = len(iterable)

    bag = db.from_sequence(iterable, npartitions=n_jobs)
    results = bag.map(func, **kwargs).compute(scheduler=sheduler)

    return results
