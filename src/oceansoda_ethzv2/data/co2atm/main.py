from functools import cache

from . import mauna_loa, noaa_mbl


@cache
def get_xco2atm(window_span="8D"):
    """Get the CO2 data from Mauna Loa and NOAA MBL.

    Returns:
        xr.Dataset: A dataset containing the CO2 data from Mauna Loa and NOAA MBL.
    """
    from xarray import merge

    from ..utils.date_utils import DateWindows
    from .noaa_mbl import fix_timestep_offset, regress_seasonal_resid
    from .smoothing import smooth_iter

    dw = DateWindows(window_span=window_span)

    da_mlo = mauna_loa.get_co2_mauna_loa()
    da_mbl = noaa_mbl.get_xco2_noaa_mbl()

    years = list(set(da_mlo.time.dt.year.values.tolist()))
    times = dw.get_bin_centers(years)

    da_mlo_ws = da_mlo.interp(time=times, method="linear")
    da_mbl_ws = da_mbl.interp(time=times, method="linear")

    x = smooth_iter(da_mlo_ws, iters=3)
    y = da_mbl_ws

    da_mbl_ws_regress = regress_seasonal_resid(x, y)
    da_mbl_ws_regress = fix_timestep_offset(da_mbl_ws, da_mbl_ws_regress)
    da_mbl_ws_filled = da_mbl_ws.fillna(da_mbl_ws_regress).assign_attrs(
        da_mbl_ws_regress.attrs
        | dict(
            description=(
                "xCO2mbl regressed against Mauna Loa xCO2 for each pixel. "
                "The missing times are then computed with the regression. "
                "This dataset contains the orginal dataset + the regressed "
                "missing data at the end of the time series. "
            )
        )
    )

    ds = merge(
        [
            da_mlo_ws.rename("xco2atm_mauna_loa"),
            da_mbl_ws_filled.rename("xco2mbl_noaa"),
            da_mbl_ws_regress.rename("xco2mbl_noaa_regress"),
            da_mbl_ws.rename("xco2mbl_noaa_raw"),
        ]
    ).astype("float32")

    return ds
