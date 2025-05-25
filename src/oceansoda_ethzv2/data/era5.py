import functools

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import BaseModel, Field

from .utils.zarr_utils import ZarrDataset

ERA5_URL = "gcs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
TODAY = pd.Timestamp.now().floor("D")


class ERA5Entry(BaseModel):
    product: str = Field("era5", description="ERA5 dataset product name")
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
    def __init__(
        self,
        entries: list[dict],
        spatial_res: float = 0.25,
        temporal_res: str = "8D",
        save_path: str = "../data/{temporal_res}_{spatial_res}/{{var}}-{temporal_res}_{spatial_res}.zarr",
    ):
        """
        Initialize the ERA5Dataset with properties.
        Properties should match the ERA5Entry model.

        Parameters
        ----------
        entries : list[dict]
            A list of dictionaries containing dataset entries.
        """
        assert len(entries) == 1, "ERA5Dataset should only contain one entry."
        self.entries = entries
        self.spatial_res = spatial_res  # Default spatial resolution in degrees
        self.temporal_res = temporal_res  # Default window size for temporal resolution
        self.save_path = save_path.format(
            temporal_res=temporal_res,
            spatial_res=f"{str(spatial_res).replace('.', '')}",
        )

    def _open_full_dataset(self) -> xr.Dataset:
        """
        Open the ERA5 dataset based on the properties defined in this instance.

        Returns
        -------
        xr.Dataset
            The full ERA5 dataset as an xarray Dataset.
        """
        return open_era5(self.entries[0])

    def _regrid_data(self, ds: xr.Dataset) -> xr.Dataset:
        from .processors import standard_regridding

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

    @classmethod
    def init_with_defaults(
        cls,
        spatial_res: float = 0.25,  # Default spatial resolution in degrees
        temporal_res: str = "8D",  # Default temporal resolution
        time_start="1982-01-01",
        time_end=None,
        vars: dict[str, str] | None = {
            "latitude": "lat",
            "longitude": "lon",
            "mean_sea_level_pressure": "msl_era5",
            "10m_u_component_of_wind": "u10_era5",
            "10m_v_component_of_wind": "v10_era5",
        },
    ):
        """
        Load the ERA5 dataset with default parameters.

        Parameters
        ----------
        time_start : str, optional
            Start time of the dataset, defaults to '1982-01-01'.
        time_end : str, optional
            End time of the dataset, defaults to None (no clipping).
        vars : dict[str, str] | None, optional
            Variables to rename in the dataset, defaults to an empty dictionary.

        Returns
        -------
        ERA5Dataset
            An instance of ERA5Dataset with the loaded entries.
        """
        entry = ERA5Entry(
            product="era5",
            time_start=time_start,
            time_end=time_end,
            vars=vars,
        ).model_dump()  # Convert to dictionary

        return cls(entries=[entry], spatial_res=spatial_res, temporal_res=temporal_res)

    def write_timestep_to_disk(
        self,
        year: int,
        index: int,
        progress: bool = False,
        check_data_formatting: bool = True,
    ):
        """
        Write a single timestep of the ERA5 dataset to disk.

        Parameters
        ----------
        ds : xr.Dataset
            The xarray Dataset containing the ERA5 data.
        time : pd.Timestamp
            The timestamp for the data to be saved.
        save_path : str, optional
            The path where the data will be saved, defaults to '../data/era5/{time:%Y%m%d}.zarr'.
        """
        from .checker import TimestepChecker
        from .utils.zarr_utils import save_vars_to_zarrs

        ds = self.get_time_stride_regridded(year=year, index=index)

        if check_data_formatting:
            checker = TimestepChecker(
                spatial_res=self.spatial_res, time_window=self.temporal_res
            )
            checker.check_lon_lat(ds)
            ds = checker.fix_timestep(ds)
            ds = checker.add_time_bnds(ds)

        save_vars_to_zarrs(
            ds,
            self.save_path,
            group_by_year=True,
            progress=progress,
            error_handling="warn",
        )


@functools.lru_cache(maxsize=1)
def _open_era5_zarr_from_gcs(url: str = ERA5_URL):
    import xarray

    ds = xarray.open_zarr(
        url,
        chunks=None,  # type: ignore
        storage_options=dict(token="anon"),
    )

    return ds


def open_era5(entry: dict):
    """
    Open a single ERA5 dataset based on the provided entry dictionary.

    Parameters
    ----------
    entry : dict
        A dictionary containing the dataset entry information.

    Returns
    -------
    xr.Dataset
        The opened xarray Dataset.
    """
    from .processors import rename_and_drop

    ds = _open_era5_zarr_from_gcs()

    # Select the time range specified in the entry
    t0 = entry.get("time_start", None)
    t1 = entry.get("time_end", None)
    t1 = t1 if t1 is not None else TODAY
    ds = ds.sel(time=slice(t0, t1))

    ds_sub = ds.pipe(rename_and_drop, entry.get("vars", {}))
    ds_sub["flag_era5nrt"] = make_era5_nrt_flags(ds_sub)

    ds_sub = ds_sub.chunk({"time": 1, "lat": -1, "lon": -1})  # for lazy loading

    return ds_sub


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
