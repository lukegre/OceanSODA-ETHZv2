from functools import lru_cache
from typing import Callable

import numpy as np
import pandas as pd
import pooch
import xarray as xr
from memoization import cached


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

    kwargs = dict(njobs=-1, backend="threading")

    results = Parallel(**kwargs)(  # type: ignore
        delayed(chech_if_url_exists)(url) for url in urls
    )
    return tuple(results)


def get_urls_that_exist(urls: tuple) -> tuple[str]:
    """
    Check which URLs exist and return the ones that do.
    """
    urls_exist = check_if_urls_exist(urls)
    url_list = [url for url, exists in zip(urls, urls_exist) if exists]
    return tuple(url_list)


def data_url_data_in_parallel(urls: tuple[str], url_processor: Callable) -> list:
    import dask.bag as db
    from tqdm.dask import TqdmCallback as ProgressBar

    with ProgressBar(desc=f"Downloading {len(urls)} files"):
        return (
            db.from_sequence(urls, npartitions=len(urls))
            .map(lambda url: url_processor(url) if url else None)
            .compute(scheduler="threads")
        )


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


def download_netcdfs_from_ftp(urls: tuple, netcdf_opender: Callable, **fsspec_kwargs):
    from functools import partial

    urls_on_ftp_server = check_urls_on_ftp_server(urls, **fsspec_kwargs)
    url_processor = partial(
        netcdf_tempfile_reader, netcdf_opener=netcdf_opender, **fsspec_kwargs
    )

    url_data_in_list = data_url_data_in_parallel(urls_on_ftp_server, url_processor)

    return url_data_in_list


def make_urls_from_dates(dates: pd.DatetimeIndex, url_template: str) -> tuple[str, ...]:
    url_list = [(url_template.format(time=date)) for date in dates]
    return tuple(url_list)
