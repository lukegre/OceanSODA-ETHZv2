import pandas as pd
import xarray as xr

MAUNA_LOA_URL = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_weekly_mlo.csv"


def get_co2_mauna_loa(url: str = MAUNA_LOA_URL) -> xr.DataArray:
    df = pd.read_csv(
        url,
        comment="#",
        # index_col='time',
        na_values="-999.99",
    )

    date_cols = ["year", "month", "day"]

    df = (
        df.assign(time=lambda x: pd.to_datetime(x[date_cols]))
        .set_index("time")
        .drop(columns=date_cols)
    )

    da = (
        df.rename(columns=lambda s: s.replace(" ", "_"))
        .increase_since_1800.rename("xco2atm_mauna_loa")
        .interpolate()
        .to_xarray()
        .assign_attrs(
            source=url,
            units="ppm",
        )
    )

    return da
