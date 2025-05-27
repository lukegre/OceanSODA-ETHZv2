from functools import lru_cache

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import BaseModel, Field

from .utils.zarr_utils import ZarrDataset

TODAY = pd.Timestamp.now().floor("D")


class ERA5Entry(BaseModel):
    source_path: str = Field(..., description="URL to the ERA5 dataset Zarr store")
    time_start: str | None = Field(
        None,
        description="Start time of the dataset, will clip the dataset, None means no clipping",
    )
    time_end: str | None = Field(
        None,
        description="End time of the dataset, will clip the dataset, None means no clipping",
    )
    vars: dict[str, str] | None = Field(
        {},
        description="Variables to rename in the dataset, e.g. {'2m_temperature': 't2m'}",
    )


class ERA5Dataset(ZarrDataset):
    checks = ("fix_timestep", "add_time_bnds", "check_lon_lat")

    @lru_cache(maxsize=1)
    def _open_full_dataset(self) -> xr.Dataset:
        """
        Open the ERA5 dataset based on the properties defined in this instance.

        Returns
        -------
        xr.Dataset
            The full ERA5 dataset as an xarray Dataset.
        """
        from .utils.processors import rename_and_drop

        entry = self.entries[0]
        ds = self._open_era5_gcs_zarr(entry["source_path"])

        # Select the time range specified in the entry
        t0 = entry.get("time_start", None)
        t1 = entry.get("time_end", None)
        t1 = t1 if t1 is not None else TODAY

        ds = ds.sel(time=slice(t0, t1))

        ds_sub = ds.pipe(rename_and_drop, entry.get("vars", {}))
        ds_sub["flag_era5nrt"] = make_era5_nrt_flags(ds_sub)

        ds_sub = ds_sub.chunk({"time": 24, "lat": -1, "lon": -1})  # for lazy loading

        return ds_sub

    def _regrid_data(self, ds: xr.Dataset) -> xr.Dataset:
        from .utils.processors import standard_regridding

        if "u10_era5" in ds and "v10_era5" in ds:
            wind_moments = calc_wind_moments(ds)
            ds = xr.merge([ds, wind_moments])

        ds = standard_regridding(
            ds_in=ds, spatial_res=self.spatial_res, window_size=self.temporal_res
        )

        if "u10_era5" in ds and "v10_era5" in ds:
            ds["wind_speed"] = np.sqrt(ds["wind_2nd_moment_era5"])
            ds = ds.drop_vars(
                [
                    "wind_3rd_moment_era5_sigma_regrid",
                    "v10_era5_sigma_regrid",
                    "u10_era5_sigma_regrid",
                ]
            )

        return ds

    def _validate_entry(self, entry: dict):
        ERA5Entry(**entry)

    @lru_cache(maxsize=1)
    def _open_era5_gcs_zarr(self, url):
        import xarray

        ds = xarray.open_zarr(
            url,
            chunks=None,  # type: ignore
            storage_options=dict(token="anon"),
        )

        return ds


def make_era5_nrt_flags(ds: xr.Dataset) -> xr.DataArray:
    """
    Create a flag variable for ERA5 datasets along the specified dimension.

    Parameters
    ----------
    ds : xr.Dataset
        The xarray Dataset containing the ERA5 data.
    flag_dim : str, optional
        The dimension along which to create the flag, defaults to 'time'.
    flag_name : str, optional
        The name of the flag variable, defaults to 'flag'.

    Returns
    -------
    xr.DataArray
        An xarray DataArray containing the flag values.
    """

    time = ds["time"].to_index()

    start_of_era5t = pd.Timestamp(ds.attrs.get("valid_time_start_era5t", TODAY))
    era5t_mask = np.array(time >= start_of_era5t)

    flags = np.zeros_like(time, dtype=np.bool_)  # 0 = validated ERA5 data
    flags[era5t_mask] = True  # 2 = NRT ERA5 data

    flags = xr.DataArray(
        data=flags,
        dims=["time"],
        coords={"time": time},
        attrs={
            "flag_values": [1, 2],
            "flag_meanings": "validated nrt",
            "valid_min": 1,
            "valid_max": 2,
            "description": "ERA5 data flags: 1 = validated, 2 = NRT",
        },
    )
    return flags


def calc_wind_moments(ds: xr.Dataset, u_var="u10_era5", v_var="v10_era5") -> xr.Dataset:
    """
    Calculate wind moments (mean and standard deviation) from the u and v components.

    Parameters
    ----------
    ds : xr.Dataset
        The xarray Dataset containing the u and v wind components.
    u_var : str, optional
        The variable name for the u component, defaults to 'u10'.
    v_var : str, optional
        The variable name for the v component, defaults to 'v10'.

    Returns
    -------
    xr.Dataset
        The dataset with added wind moments.
    """

    u = ds[u_var]
    v = ds[v_var]

    wind_moments = xr.Dataset()
    wind_moments["wind_2nd_moment_era5"] = u**2 + v**2
    wind_moments["wind_3rd_moment_era5"] = u**3 + v**3

    return wind_moments
