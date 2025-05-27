import pathlib
from functools import lru_cache, partial
from typing import Callable, Optional

import numpy as np
import pandas as pd
import xarray as xr

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
        source_path: str = OCCCI_URL,
        fsspec_kwargs: dict = {},
        vars: dict[str, str] = {},
        **kwargs,
    ):
        """
        Initialize the OCCCIDataset with properties.
        """
        self.source_path = source_path
        self.spatial_res = spatial_res  # Default spatial resolution in degrees
        self.window_span = window_span  # Default window size for temporal resolution
        self.fsspec_kwargs = (
            fsspec_kwargs  # Additional fsspec arguments for downloading
        )
        self.vars = vars
        self.save_path = save_path.format(
            window_span=window_span, spatial_res=f"{str(spatial_res).replace('.', '')}"
        )

    def _get_unprocessed_timestep_remote(
        self,
        year: Optional[int] = None,
        index: Optional[int] = None,
        time: Optional[pd.Timestamp | str] = None,
    ) -> xr.Dataset:
        from .utils.download import download_netcdfs_from_ftp, make_paths_from_dates

        time = self.date_windows.get_window_center(year=year, index=index, time=time)
        dates = self.date_windows.get_window_dates(time=time)
        urls = make_paths_from_dates(dates, string_template=self.source_path)

        ds_list = download_netcdfs_from_ftp(
            urls=urls, netcdf_opender=self._opener, **self.fsspec_kwargs
        )
        ds = xr.concat(ds_list, dim="time").chunk({"time": 1, "lat": -1, "lon": -1})

        return ds

    def _get_unprocessed_timestep_local(
        self,
        year: int | None = None,
        index: int | None = None,
        time: Optional[pd.Timestamp | str] = None,
    ) -> xr.Dataset:
        from .utils.download import make_paths_from_dates

        time = self.date_windows.get_window_center(year=year, index=index, time=time)
        dates = self.date_windows.get_window_dates(time=time)

        paths = make_paths_from_dates(dates, self.source_path)
        paths = [path for path in paths if pathlib.Path(path).exists()]

        ds_list = [self._opener(path) for path in paths]
        ds = xr.concat(ds_list, dim="time")
        ds = ds.chunk({"time": 1, "lat": -1, "lon": -1})
        return ds

    def _regrid_data(self, ds) -> xr.Dataset:
        from .utils.processors import coarsen_then_interp

        time = self.date_windows.get_window_center(time=ds.time.to_index()[0])
        ds = coarsen_then_interp(
            ds, spatial_res=self.spatial_res, window_size=self.window_span
        )
        ds = ds.assign_coords(time=[time])

        return ds

    def _opener(self, fname: str) -> xr.Dataset:
        """
        Open the dataset from a file name.
        """
        from .utils.processors import preprocessor_generic

        ds = xr.open_dataset(fname, chunks={}, decode_timedelta=False)
        ds = preprocessor_generic(
            ds=ds, vars_rename=self.vars, depth_idx=None, coords_duplicate_check=[]
        )

        return ds
