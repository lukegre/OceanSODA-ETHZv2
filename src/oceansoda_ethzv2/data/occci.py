import pathlib
from functools import lru_cache, partial
from typing import Callable, Optional

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

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
            ds=ds, vars_rename=self.vars, coords_duplicate_check=[]
        )

        return ds


#########################
## Chlorophyll filling ##
#########################


class ChlorophyllFillingSVD:
    def __init__(
        self,
        clim: xr.DataArray,
        n_iter: int = 13,
        n_components: int = 8,
    ):
        """
        Initialize the Chlorophyll Filling SVD class.

        Parameters
        ----------
        n_iter : int
            Number of iterations for the SVD reconstruction.
        n_components : int
            Number of principal components to use in the reconstruction.
        """
        self.n_iter = n_iter
        self.n_components = n_components
        if self._is_chl_log10(clim):
            raise ValueError("CHL climatology data must be in log10 scale.")
        self.clim = clim

    def __call__(self, da: xr.DataArray) -> xr.Dataset:
        """
        Fill gaps in the provided DataArray using SVD reconstruction.

        Parameters
        ----------
        da : xr.DataArray
            Input data array with gaps to be filled. It is expected to have a 'time' dimension.

        Returns
        -------
        xr.DataArray
            A new xarray DataArray with missing values filled using SVD reconstruction.
        """
        if self.clim is None:
            raise ValueError("Climatology data must be provided for gap filling.")

        if self.clim.shape != da.shape:
            raise ValueError(
                f"Input data must match climatology size. Input data has "
                f"shape {da.sizes}, but climatology has shape {self.clim.sizes}"
            )

        if not self._is_chl_log10(da):
            raise ValueError(
                "Input data must be in log10 scale. Convert with self.convert_log10"
            )

        return make_chl_filled(
            chl_year=da,
            clim=self.clim,
            dinsvd_kwargs={"n_iter": self.n_iter, "n_components": self.n_components},
        )

    @staticmethod
    def _is_chl_log10(da: xr.DataArray) -> bool:
        result = (da.min() < -0.1) & (da.max() < 10)
        return np.array(result).item()

    def convert_log10(self, da: xr.DataArray) -> xr.DataArray:
        """
        Convert the data to log10 scale if it is not already in that scale.

        Parameters
        ----------
        da : xr.DataArray
            Input data array to be converted.

        Returns
        -------
        xr.DataArray
            Data array in log10 scale.
        """
        if self._is_chl_log10(da):
            return da
        elif da.min() < 0:
            raise ValueError(
                "Input data contains negative values, cannot convert to log10 scale."
            )
        else:
            name = da.name
            units = da.attrs.get("units", "unknown")
            return (
                da.clip(min=5e-3)
                .pipe(np.log10)
                .reanme(f"{name}_log10")  # type: ignore
                .assign_attrs(units=f"log10({units})", processing_log10=True)
            )


def svd_reconstruction(
    arr: np.ndarray,
    first_guess: float | np.ndarray = 0,
    n_components: int = 8,
    n_iter: int = 10,
    svd_kwargs: dict = dict(n_iter=2),
) -> tuple[np.ndarray, list]:
    """
    Fills gappy data using Singular Value Decomposition (SVD) reconstruction.

    Based on the idea that the principle components can be used to iteratively
    reconstruct the data with less uncertainty when applied iteratively.

    Parameters
    ----------
    arr : np.ndarray [2D]
        The array to be filled structured as (time, stacked space-coords).
        The missing values should be NaNs and these will be filled.
    first_guess : np.ndarray [2D] or float
        A first guess of the missing values. If a float is provided, all missing
        values will be filled with this value. If an array is provided, it has to
        be the same shape as `arr` and will be used as the first guess. An example
        of a good first guess is the climatology. NaNs in the first guess will be
        treated as land values and will not be filled in the final output.
    n_components : int
        The number of principle components to use in the reconstruction.
    n_iter : int
        The number of iterations to perform.
    svd_kwargs : dict
        Keyword arguments to pass to the `randomized_svd` function from sklearn.
    verbosity : bool
        If True, prints the iteration number and the current error.

    Returns
    -------
    filled : np.ndarray [2D]
        The filled array. If first_guess was provided with NaNs, these values will
        be filled with NaNs in the final output.
    """
    from sklearn.utils.extmath import randomized_svd

    logger.debug(f"SVD reconstruction of Chl-a [{n_iter} iters]")
    X = np.array(arr)
    m = np.isnan(X)

    if isinstance(first_guess, (np.ndarray, pd.DataFrame)):
        land_mask = np.isnan(first_guess)
        filler = np.nan_to_num(first_guess, nan=0, neginf=0, posinf=0)[m]
    else:
        land_mask = None
        filler = first_guess

    X[m] = filler

    error = []
    for i in range(n_iter):
        u, s, v = randomized_svd(X, n_components, **svd_kwargs)
        out = u @ np.diag(s) @ v
        X[m] = out[m]
        error += (np.nansum(abs(X[~m] - out[~m])),)
        logger.trace(f"{i:02d}: {error[-1]:.2f}")

    filled = X
    if land_mask is not None:
        filled[land_mask] = np.nan

    return filled, error


def fill_lat_lon_nearest(da: xr.DataArray) -> xr.DataArray:
    """
    Fill missing values in an xarray DataArray by propagating the nearest
    available values along the 'time', 'lat', and 'lon' dimensions.
    The function performs forward fill (ffill) and backward fill (bfill)
    operations sequentially along each dimension ('time', 'lat', 'lon')
    with a limit of 1, ensuring that missing values are filled with the
    nearest available data.
    Parameters:
    -----------
    da : xr.DataArray
        The input xarray DataArray containing missing values to be filled.
    Returns:
    --------
    xr.DataArray
        A new xarray DataArray with missing values filled using the nearest
        available values along the specified dimensions.
    """
    da_filled = (
        da.ffill(dim="time", limit=1)
        .bfill(dim="time", limit=1)
        .ffill(dim="lat", limit=1)
        .bfill(dim="lat", limit=1)
        .ffill(dim="lon", limit=1)
        .bfill(dim="lon", limit=1)
    )

    return da_filled


def fill_gaps_with_clim_dinsvd(
    da: xr.DataArray, clim: xr.DataArray, n_iter: int = 13, n_components: int = 8
) -> tuple[xr.DataArray, list]:
    """
    Fill gaps in a data array using climatology and a
    Data Interpolating Singular Value Decomposition (DInSVD)

    Parameters:
    -----------
    da : xr.DataArray
        Input data array with gaps to be filled. It is expected to have a 'time' dimension.
    clim : xr.DataArray
        Climatology data array used as the first guess for the gap-filling process.
        Must have the same shape and coordinates as `da`.
    n_iter : int, optional
        Number of iterations for the SVD-based reconstruction. Default is 13.
    n_components : int, optional
        Number of principal components to retain during the reconstruction. Default is 8.
    Returns:
    --------
    tuple[xr.DataArray, list]
        A tuple containing:
        - `chl_dineof` (xr.DataArray): The gap-filled data array with the same shape and coordinates as `da`.
        - `dineof_err` (list): A list of reconstruction errors for each iteration.
    Notes:
    ------
    - This function is inspired by the DINEOF approach to iteratively reconstruct missing values in the input data array.
    - The process may take some time depending on the size of the input data and the number of iterations.
    """

    logger.debug("CHL gap filling (SVD approach) may take some time")
    t = da.time.size

    chl_dineof, dineof_err = svd_reconstruction(
        arr=da.values.reshape(t, -1),
        first_guess=clim.values.reshape(t, -1),
        n_iter=n_iter,
        n_components=n_components,
    )

    chl_dineof = xr.DataArray(
        data=chl_dineof.reshape(da.shape), coords=da.coords, dims=da.dims
    )

    return chl_dineof, dineof_err


def make_chl_filled(
    chl_year: xr.DataArray, clim: xr.DataArray, dinsvd_kwargs: dict[str, int] = {}
) -> xr.Dataset:
    from tqdm.dask import TqdmCallback as ProgressBar

    name = chl_year.name

    with ProgressBar():
        chl_nearest = fill_lat_lon_nearest(chl_year).persist()
        chl_dineof, _ = fill_gaps_with_clim_dinsvd(
            chl_nearest, clim=clim, **dinsvd_kwargs
        )

    chl_filled = chl_year.fillna(chl_nearest).fillna(chl_dineof)

    mask_year = chl_year.notnull()
    mask_nearest = chl_nearest.notnull()
    mask_dineof = chl_dineof.notnull()

    chl_flag = (
        (mask_year * 1)
        + (mask_nearest & ~mask_year) * 2
        + (mask_dineof & ~mask_year & ~mask_nearest) * 3
    )

    ds = xr.merge(
        [chl_filled.rename(f"{name}_filled"), chl_flag.rename(f"{name}_filled_flag")]
    )

    return ds
