import xarray as xr


def open_cmems_dataset(props):
    import copernicusmarine

    from .processors import rename_and_drop

    ds_all = copernicusmarine.open_dataset(props["id"])

    t0 = props["time_start"]
    t1 = props["time_end"]
    ds_sub = ds_all.sel(time=slice(t0, t1))

    ds_sub = ds_sub.pipe(rename_and_drop, props["vars"])

    if "flag" in props:
        flag_values = [props["flag"]] * ds_sub.time.size
        ds_sub["flag"] = xr.DataArray(flag_values, dims=["time"])

    return ds_sub
