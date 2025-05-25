from . import mauna_loa, noaa_mbl


def get_co2atm():
    """Get the CO2 data from Mauna Loa and NOAA MBL.

    Returns:
        xr.Dataset: A dataset containing the CO2 data from Mauna Loa and NOAA MBL.
    """
    from xarray import merge

    from ..utils.date_utils import DateWindows
    from ..utils.smoothing import smooth_iter
    from .noaa_mbl import fix_timestep_offset, regress_seasonal_resid

    dw = DateWindows()

    da_mlo = mauna_loa.get_co2_mauna_loa()
    da_mbl = noaa_mbl.get_xco2_noaa_mbl()

    years = list(set(da_mlo.time.dt.year.values.tolist()))
    times = dw.get_bin_centers(years)

    da_mlo_8D = da_mlo.interp(time=times, method="linear")
    da_mbl_8D = da_mbl.interp(time=times, method="linear")

    x = smooth_iter(da_mlo_8D, iters=3)
    y = da_mbl_8D

    da_mbl_8D_regress = regress_seasonal_resid(x, y)
    da_mbl_8D_regress = fix_timestep_offset(da_mbl_8D, da_mbl_8D_regress)
    da_mbl_8d_filled = da_mbl_8D.fillna(da_mbl_8D_regress).assign_attrs(
        da_mbl_8D_regress.attrs
    )

    ds = merge(
        [
            da_mlo_8D.rename("xco2atm_mauna_loa"),
            da_mbl_8d_filled.rename("xco2mbl_noaa"),
            da_mbl_8D_regress.rename("xco2mbl_noaa_regress"),
            da_mbl_8D.rename("xco2mbl_noaa_raw"),
        ]
    ).astype("float32")

    return ds
