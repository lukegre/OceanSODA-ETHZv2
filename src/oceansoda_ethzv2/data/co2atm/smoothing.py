from typing import Optional

import xarray as xr


def smooth_seas_cycle(
    da: xr.DataArray | xr.Dataset, dim="dayofyear", window=3, iters=1, func="mean"
) -> xr.DataArray:
    """
    Smooth a seasonal cycle in an xarray DataArray by applying a moving window
    function iteratively.

    Notes
    -----
    - The function extends the input data along the specified dimension by
      concatenating it with itself to avoid edge effects during smoothing.
    - After smoothing, the function extracts the original range of the data
      and reassigns the original coordinates.

    Parameters
    ----------
    da : xarray.DataArray
        The input data array containing the seasonal cycle to be smoothed.
    dim : str, optional
        The dimension along which to apply the smoothing. Default is 'dayofyear'.
    window : int, optional
        The size of the moving window for smoothing. Default is 3.
    iters : int, optional
        The number of iterations to apply the smoothing function. Default is 1.
    func : str, optional
        The smoothing function to apply. Options include 'mean' or other
        functions supported by the `smooth_iter` function. Default is 'mean'.

    Returns
    -------
    xarray.DataArray
        The smoothed data array with the same dimensions as the input array.

    Example
    -------
    >>> smoothed_da = smooth_seas_cycle(da, dim='dayofyear', window=5, iters=2, func='mean')
    """

    da3 = xr.concat([da, da, da], dim=dim)
    da3 = da3.assign_coords(dayofyear=range(da3[dim].size))

    da3_smooth = smooth_iter(da3, dim=dim, window=window, iters=iters, func=func)

    i0 = da[dim].size
    i1 = i0 * 2
    da_smooth = da3_smooth.isel({dim: slice(i0, i1)})
    da_smooth = da_smooth.assign_coords({dim: da[dim]})

    return da_smooth


def smooth_iter(
    da: xr.DataArray, dim="time", window=3, iters=3, func="mean"
) -> xr.DataArray:
    """
    Smooths a DataArray by applying a rolling operation iteratively.

    Parameters:
    -----------
    da : xarray.DataArray
        The input data array to be smoothed.
    dim : str, optional
        The dimension along which to apply the rolling operation. Default is 'time'.
    window : int, optional
        The size of the rolling window. Default is 3.
    iters : int, optional
        The number of iterations to apply the rolling operation. Default is 3.
    func : str, optional
        The aggregation function to apply on the rolling window.
        Must be a method of the rolling object (e.g., 'mean', 'sum', etc.). Default is 'mean'.

    Returns:
    --------
    xarray.DataArray
        The smoothed DataArray after applying the rolling operation iteratively.
    """

    da_smooth = da
    for i in range(iters):
        da_rolling = da_smooth.rolling(**{dim: window}, center=True, min_periods=1)
        da_smooth = getattr(da_rolling, func)()

    return da_smooth
