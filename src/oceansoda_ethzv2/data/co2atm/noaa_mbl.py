from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

# date fetched = "16.05.2025"
NOAA_MBL_URL = (
    "https://gml.noaa.gov/ccgg/mbl/tmp/co2_GHGreference.683788075_surface.txt"
)
LATITUDES = np.arange(-90 + 0.125, 90, 0.25)


def get_xco2_noaa_mbl(
    url=NOAA_MBL_URL, lat: Optional[np.ndarray] = LATITUDES, dest: Optional[str] = None
) -> xr.DataArray:
    """Downloads url and reads in the MBL surface file

    Data is downloaded to a temporary location using pooch

    Args:
        url (str): the address for the noaa surface file (default: NOAA_MBL_URL)
        lat (np.ndarray): latitude to which the data should be interpolated
        dest (str): the destination to which the raw file will be saved

    Returns:
        pd.Series: multindexed series of xCO2 with (time, lat) as coords.
    """
    import numpy as np
    import pandas as pd

    from ..utils.download import download_w_pooch

    # save to temporary location with pooch
    fname = download_w_pooch(url, dest=dest, progress=False)
    df = parse_noaa_mbl_txt(fname)
    da = parse_noaa_mbl_table_to_xarray(df, lat=lat)

    da.attrs["source"] = url
    da.attrs["date_url_fetched"] = "2025-05-16"

    return da


def parse_noaa_mbl_table_to_xarray(df, lat=None):
    """Reads in the MBL surface file and returns a xarray dataset

    Args:
        fname (str): the address for the noaa surface file
        lat (float): latitude to which the data should be interpolated

    Returns:
        xr.Dataset: xarray dataset with xco2mbl_noaa as variable
    """
    import numpy as np
    import pandas as pd

    # renaming indexes (have to stack for that)
    df = df.stack()
    index = df.index.set_names(["time", "lat"])
    df = df.set_axis(index)

    da = df.to_xarray()

    if lat is not None:
        da = da.interp(lat=lat).rename("xco2mbl_noaa")

    return da


def parse_noaa_mbl_txt(fname) -> pd.DataFrame:
    start_line = get_start_line(fname)

    # read fixed width file CO2
    df = pd.read_fwf(fname, skiprows=start_line, header=None, index_col=0)
    df.index.name = "date"
    # every second line is uncertainty
    df = df.iloc[:, ::2]
    # latitude is given as sin(lat)
    df.columns = np.rad2deg(np.arcsin(np.linspace(-1, 1, 41)))

    # resolve time properly
    year = (df.index.values - (df.index.values % 1)).astype(int)
    day_of_year = ((df.index.values - year) * 365 + 1).astype(int)
    date_strings = ["{}-{:03d}".format(*a) for a in zip(year, day_of_year)]

    date = pd.to_datetime(date_strings, format="%Y-%j")

    df = df.set_index(date)

    return df


def get_start_line(fname) -> int:
    """Finds the start line of the MBL surface file

    Args:
        fname (str): the address for the noaa surface file

    Returns:
        int: the start line of the MBL surface file
    """
    import re

    start_line = 0
    is_mbl_surface = False
    for start_line, line in enumerate(open(fname)):
        if re.findall("MBL.*SURFACE", line):
            is_mbl_surface = True
        if not line.startswith("#"):
            break

    if not is_mbl_surface:
        raise Exception(
            "The file at the provided url is not an MBL SURFACE file. "
            "Please check that you have provided the surface url. "
        )

    return start_line


def regress_seasonal_resid(
    x: xr.DataArray, y: xr.DataArray, dim="time", season_agg_dim="time.dayofyear"
) -> xr.DataArray:
    """
    Perform a linear regression of x against y over dimension

    We assume that the seasonal cycle is present in y, but not in x.
    This means that the residuals will carry the seasonal cycle.
    After making predictions, we add a seasonal climatology of the residuals
    to the predictions, resulting in a prediction with the mean
    seasonal cycle added. A bit of a dirty shortcut

    Parameters
    ----------
    x: xr.DataArray
        Must have dimension in 'dim'
    y: xr.DataArray
        Must have dimension in 'dim' but can also have additional dims
    dim: str
        dimension over which data is regressed
    season_agg_dim: str
        the seasonal key used to aggregate the data (groupby)

    Returns
    -------
    xr.DataArray:
        has dims from Y but with length of 'dim' from X

    """
    import xskillscore as xs
    from sklearn.metrics import r2_score

    from ...model.metrics import r2_score
    from .smoothing import smooth_seas_cycle

    # compute the seasonal cycle
    m = xs.linslope(x, y, dim=dim, skipna=True)
    # and the intercept
    b = y.mean(dim=dim) - m * x.mean(dim=dim)
    # doing prediction
    yhat = (m * x) + b
    yhat = yhat.transpose(dim, ...)

    # computing the seasonal cycle of the residuals
    resid = y - yhat
    resid_seasonal = resid.groupby(season_agg_dim).mean()
    resid_seasonal = smooth_seas_cycle(resid_seasonal)

    # add the seasonal cycle from the residuals to the predictions
    yhat_seasonal = yhat.groupby(season_agg_dim) + resid_seasonal

    # compute some metrics for the comparison
    attrs = dict(
        description=f"Regression of `{x.name}` against `{y.name}` with seasonal cycle added",
        r2_excl_seas_resid=np.around(r2_score(y, yhat), 3),
        r2_incl_seas_resid=np.around(r2_score(y, yhat_seasonal), 3),
    )
    # assign the attributes to the seasonal cycle
    yhat_seasonal = yhat_seasonal.assign_attrs(attrs).drop_vars("dayofyear")

    return yhat_seasonal


def fix_timestep_offset(da_orig, da_filler):
    """
    When interpolating, there can be a slight missmatch in the xco2 between
    the last timestep of the original data and the first timestep of the
    interpolated data. This function fixes this by adding the difference
    between the two timesteps to the interpolated data.
    """

    t0 = da_orig.dropna("time").time[-1]
    da_orig_last = da_orig.sel(time=t0)
    da_fill_first = da_filler.sel(time=slice(t0, None)).isel(time=0)

    offset = da_orig_last - da_fill_first
    da_adjusted = da_filler + offset

    da_adjusted = da_adjusted.assign_attrs(
        da_filler.attrs
        | {"interpolated_from": da_fill_first.time.dt.strftime("%Y-%m-%d").item()}
    )

    return da_adjusted
