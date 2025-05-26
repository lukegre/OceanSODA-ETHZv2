import pathlib
from typing import Optional

import munch
import numpy as np
import pandas as pd
import xarray as xr


def standard_regridding(
    ds_in: xr.Dataset,
    spatial_res=0.25,
    window_size="8D",
    lon_name="lon",
    dtype="float32",
) -> xr.Dataset:
    """
    standard regridding function for CMEMS datasets.
    Renames variables and adjusts longitude coordinates to 180W-180E.
    """
    ds_in = ds_in.chunk({"time": 1, "lat": -1, "lon": -1})

    ds = lon_180W_180E(ds_in, lon_name=lon_name)

    # Coarsen and interpolate daily data if needed
    ds = coarsen_then_interp(ds, spatial_res=spatial_res, window_size=window_size)

    # Convert all data variables to the standard dtype
    for var in ds.data_vars:
        ds[var] = ds[var].astype(dtype)

    if "depth" in ds.dims:
        ds = ds.sel(depth=0, method="nearest", drop=True)

    return ds


def preprocessor_generic(ds, vars_rename={}, depth_idx=None, coords_duplicate_check=[]):
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
    ds = rename_and_drop(ds, vars_rename)

    if depth_idx is not None:
        ds = ds.isel(depth=depth_idx, drop=True)

    sizes = {}
    for coord in coords_duplicate_check:
        ds[coord] = ds[coord].round(3)
        sizes[coord] = ds[coord].size
    ds = ds.drop_duplicates(coords_duplicate_check)
    if any(sizes[coord] != ds[coord].size for coord in coords_duplicate_check):
        ds = ds.assign_attrs(processing_coords_duplicate_check=str(sizes))

    ds = lon_180W_180E(ds)

    for coord in ["lon", "lat"]:
        ds = sort_coord(ds, coord)

    return ds


def sort_coord(ds, coord_name):
    ser = ds[coord_name].to_series()
    if not ser.is_monotonic_increasing:
        ds = ds.sortby(coord_name)
        ds = ds.assign_attrs({f"processing_{coord_name}_sorted": True})
    return ds


def rename_and_drop(ds: xr.Dataset, rename: dict):
    if rename == {}:
        return ds
    else:
        keep = list(rename)
        ds = ds[keep].rename(rename)
        ds = ds.assign_attrs(processing_renamed_vars=str(rename))
        return ds


def lon_180W_180E(ds, lon_name="lon"):
    from numpy import isclose

    lon = ds[lon_name].values

    lon180 = (lon - 180) % 360 - 180
    if isclose(lon, lon180).all():
        return ds

    ds = (
        ds.assign_coords(**{lon_name: lon180})
        .sortby(lon_name)
        .assign_attrs(processing_lon360_to_180=True)
    )

    return ds


def make_target_global_grid(res, lon_0_360=False) -> munch.Munch:
    half = res / 2

    if lon_0_360:
        x0 = 0 + half
        x1 = 360
    else:
        x0 = -180 + half
        x1 = 180

    y0 = -90 + half
    y1 = 90

    da = munch.Munch(lat=np.arange(y0, y1, res), lon=np.arange(x0, x1, res))
    return da


def get_timesteps_per_window(ds, window_size: str = "8D", time_dim="time"):
    """
    Returns the number of timesteps per window.
    """
    window_td = pd.Timedelta(window_size)

    if time_dim not in ds.dims:
        raise ValueError(f"Dataset does not contain dimension '{time_dim}'")

    if ds[time_dim].size == 1:
        return 1
    elif ds[time_dim].size == 0:
        raise ValueError(f"Dataset dimension '{time_dim}' is empty")
    else:
        time_diff = ds[time_dim].diff(time_dim).median()
        n_timesteps = int(window_td / time_diff)
        return n_timesteps


def coarsen_then_interp(ds, spatial_res=0.25, window_size: str = "8D"):
    import warnings

    warnings.simplefilter("ignore")

    # make target grid if not provided
    target_temporal = get_timesteps_per_window(ds, window_size=window_size)
    target_spatial = make_target_global_grid(spatial_res)
    target = target_spatial | {"time": target_temporal}

    # coarsening to match target grid resolution
    coarsened = coarsen_toward_target_grid(ds, target)

    # interpolating exactly to target grid
    target.pop("time")
    out = coarsened.interp(**target)

    return out


def coarsen_toward_target_grid(
    ds: xr.Dataset, target_grid: dict, exclude_keys=["sigma", "flag", "uncert"]
) -> xr.Dataset:
    """
    Coarsens the dataset to match the target grid resolution.
    If target_grid is None, it defaults to a 0.25 degree grid.
    """
    import re
    import warnings

    import munch

    warnings.simplefilter("ignore")

    # filter out variables that contain exclude keys
    std_vars = []
    for key in ds.data_vars:
        pattern = r"(" + r"|".join(exclude_keys) + r")"
        match = re.findall(pattern, str(key))
        if match:
            continue
        std_vars.append(key)

    # if the grid size of the input is smaller than the target, raise error to not coarsen
    if ds.lat.size < target_grid["lat"].size:
        raise ValueError(
            "Dataset latitude resolution is finer than target "
            "grid resolution. Rather use interpolation. "
        )

    coarse_dict = munch.Munch(
        boundary="pad",
        time=target_grid["time"],
        lat=ds.lat.size // target_grid["lat"].size,
        lon=ds.lon.size // target_grid["lon"].size,
    )

    coarse = ds.coarsen(**coarse_dict)
    avg = coarse.mean()
    std = coarse.std()[std_vars].rename({k: f"{k}_sigma_regrid" for k in std_vars})
    out = xr.merge([avg, std])

    out = out.assign_attrs(
        processing_coarsen_grid=str(dict(coarse_dict)),
    )

    return out


def make_file_list(catalog_entry, year):
    flist_0 = catalog_entry.glob(year=year - 1).tolist()[-1:]
    flist_1 = catalog_entry.glob(year=year).tolist()
    flist_2 = catalog_entry.glob(year=year + 1).tolist()[:1]
    flist = np.array(flist_0 + flist_1 + flist_2)

    return flist


def save_var_to_zarr(
    ds,
    dest="../data/{res_days}_{res_deg_str}/{var}_{product}-{res_days}_{res_deg_str}.zarr",
    **kwargs,
):
    assert "{var}" in dest, "`dest` must contain a {var} placeholder"

    ds = ds.chunk({"time": 1, "lat": 360, "lon": 360})

    for var in ds.data_vars:
        fname = dest.format(**kwargs, var=var)
        fname = pathlib.Path(fname)
        fname.parent.mkdir(exist_ok=True, parents=True)

        ds[[var]].to_zarr(fname, mode="w", group=kwargs["year"])
