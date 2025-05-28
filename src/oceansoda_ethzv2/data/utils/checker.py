from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger


class TimestepValidator:
    """
    DataChecker class for checking that the data has the correct grid
    with the appropriate metadata attributes.

    Methods
    -------
    - add_time_bnds(ds: xr.Dataset): Adds bounds to the dataset if they are missing (time)
    - check_lon_lat(ds: xr.Dataset): Checks if the longitude and latitude coordinates are correctly formatted
    - check_time(ds: xr.Dataset): Checks if the time coordinate is correctly formatted
    - check_missing_lon(ds: xr.Dataset): Checks if there are any missing values in the dataset
    """

    def __init__(
        self,
        spatial_res=0.25,
        time_window="8D",
        start_doy=1,
        dtype="float32",
        lon_0_360=False,
    ):
        from .date_utils import DateWindows
        from .processors import make_target_global_grid

        self.spatial_res = spatial_res
        self.time_window = time_window
        self.start_doy = start_doy
        self.dtype = dtype
        self.target_grid = make_target_global_grid(spatial_res, lon_0_360=lon_0_360)
        self.date_windows = DateWindows(
            start_day_of_year=self.start_doy, window_span=self.time_window
        )

        self.land_points = dict(
            sahara=(20, 25),
            australia=(-25, 135),
            mexico=(20, -100),
            russia=(60, 100),
            south_africa=(-33, 20),
        )

    def add_time_bnds(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Adds bounds to the dataset if they are missing (time).

        Parameters
        ----------
        ds : xr.Dataset
            The dataset to check.

        Returns
        -------
        xr.Dataset
            The dataset with time bounds added if they were missing.
        """
        from .date_utils import DateWindows

        if "time_bnds" in ds.coords:
            logger.debug("Time bounds already exist in the dataset.")
            return ds

        logger.warning("Time bounds are missing, adding them.")

        time = ds["time"].to_index()
        assert time.size == 1, "Time coordinate must have exactly one value."
        time = time[0]

        window_edges = self.date_windows.get_window_edges(time=time)

        time_bnds = xr.DataArray(
            data=np.array([window_edges]),
            dims=["time", "bnds"],
            coords={"time": ds["time"], "bnds": ["start", "end"]},
            name="time_bnds",
            attrs={
                "long_name": "Time bounds",
                "units": "days since 1970-01-01 00:00:00",
                "calendar": "standard",
            },
        )

        ds = ds.assign_coords(time_bnds=time_bnds)

        return ds

    def check_lon_lat(self, ds) -> xr.Dataset:
        """
        Checks if the longitude and latitude coordinates are correctly formatted.

        Parameters
        ----------
        ds : xr.Dataset
            The dataset to check.

        Returns
        -------
        bool
            True if the dataset has valid longitude and latitude coordinates, False otherwise.
        """
        if "lon" not in ds.coords or "lat" not in ds.coords:
            raise ValueError("Dataset must contain 'lon' and 'lat' coordinates.")

        lon = ds["lon"].values
        lat = ds["lat"].values

        target_lon = self.target_grid.lon
        target_lat = self.target_grid.lat

        if not (lon.shape == target_lon.shape and lat.shape == target_lat.shape):
            raise ValueError(
                "Longitude and latitude coordinates do not match the target grid shape."
            )

        if not (lon.min() >= target_lon.min() and lon.max() <= target_lon.max()):
            raise ValueError("Longitude values are out of bounds of the target grid.")

        if not (lat.min() >= target_lat.min() and lat.max() <= target_lat.max()):
            raise ValueError("Latitude values are out of bounds of the target grid.")

        if (lon != target_lon).all():
            raise ValueError("Longitude values do not match the target grid.")

        if (lat != target_lat).all():
            raise ValueError("Latitude values do not match the target grid.")

        return ds

    def check_land_points(
        self,
        ds: xr.Dataset,
        land_should_be_nan=True,
        check_first_var_only=True,
        error_handling="raise",
    ) -> xr.Dataset:
        """
        Makes sure that the continents are in the right place.
        We do this by picking a bunch of random points on the map that should be nan
        These are typically on large continents (e.g., sahara, australia, USA, Russia, South Africa)
        """

        if error_handling not in ["raise", "warn"]:
            raise ValueError("error_handling must be either 'raise' or 'warn'.")

        for key in ds:
            if "lon" not in ds[key].dims or "lat" not in ds[key].dims:
                continue
            else:
                da = ds[key].load()  # load the dataset into memory

                for place, (lat, lon) in self.land_points.items():
                    data_at_point = da.sel(lat=lat, lon=lon, method="nearest")
                    data_is_null = data_at_point.isnull().all().item()

                    lat = data_at_point.lat.item()
                    lon = data_at_point.lon.item()

                    success = True if (data_is_null and land_should_be_nan) else False

                    place = place.capitalize()
                    if not success:
                        if land_should_be_nan:
                            message = f"Oceanic dataset expected no data at a point in {place} (lat={lat}, lon={lon}), but found data: {data_at_point.values}."
                        else:
                            message = f"Land dataset expected data at a point in {place} (lat={lat}, lon={lon}), but found no data: {data_at_point.values}."
                        if error_handling == "raise":
                            raise ValueError(message)
                        elif error_handling == "warn":
                            logger.warning(message)
                    else:
                        logger.trace(
                            f"Ocean/Land check passed for variable '{key}' at point {place} (lat={lat}, lon={lon})."
                        )
                if check_first_var_only:
                    logger.debug(
                        f"Ocean/Land check passed for the first variable [{key}] of the dataset."
                    )
                    break
        return ds

    def fix_timestep(
        self, ds: xr.Dataset, fix_threshold=pd.Timedelta("1D")
    ) -> xr.Dataset:
        """
        Checks if the time coordinate matches the expected grid based on the time window

        Parameters
        ----------
        ds : xr.Dataset
            The dataset to check.
        fix_threshold : pd.Timedelta, optional
            The threshold for fixing the time coordinate if it is within this range.
            Defaults to '1D'. If set to None, no fixing will be attempted and will raise
        an error if the time coordinate does not match the expected value.


        Returns
        -------
        xr.Dataset
            If all tests have passed will return the dataset,
        """
        from .date_utils import DateWindows

        assert isinstance(fix_threshold, (pd.Timedelta, type(None))), (
            "fix_threshold must be a pd.Timedelta or None."
        )

        if "time" not in ds.coords:
            raise ValueError("Dataset must contain 'time' coordinate.")

        dw = DateWindows(start_day_of_year=self.start_doy, window_span=self.time_window)

        time = ds["time"].to_index()
        assert time.size == 1, "Time coordinate must have exactly one value."
        time = time[0]

        year, index = dw.get_index(time=time)
        correct_time = dw.get_bin_centers(year)[index]

        dt = abs(pd.Timedelta(time - correct_time))
        if time != correct_time:
            if fix_threshold is None:
                raise ValueError(
                    f"Time coordinate {time} does not match expected time {correct_time}."
                )
            elif dt > fix_threshold:
                raise ValueError(
                    f"Time coordinate {time} does not match expected time {correct_time}, and is outside of the fix threshold {fix_threshold}."
                )
            else:
                # If the time is within the fix threshold, we will fix it
                ds = ds.assign_coords(time=[correct_time])
                logger.warning(
                    f"Time fixed. Time coordinate {time} did not match expected time {correct_time}. Adjusted to {correct_time}."
                )
        else:
            logger.debug(
                f"Time coordinate {time} matches expected time {correct_time}."
            )

        return ds

    def check_missing_lon(
        self, ds: xr.Dataset, progressbar=False, check_first_variable_only=True
    ) -> xr.Dataset:
        """
        Mistakes happen - sometimes, longitude is not corrected before interpolation
        resulting in half the image being missing

        Warning, this is not a lazy operation, it will load the entire dataset into memory.
        """
        if progressbar:
            from tqdm.dask import TqdmCallback as ProgressBar
        else:
            from .download import DummyProgress as ProgressBar

        for key in ds.data_vars:
            if "lon" in ds[key].dims:
                with ProgressBar(
                    desc=f"Checking missing longitudes for variable '{key}'"
                ):
                    da = ds[
                        key
                    ].load()  # load, so that we don't have to load later again

                lon_missing = da.count("lat").isel(time=0) == 0

                if lon_missing.any():
                    missing_lons = da["lon"].sel(lon=lon_missing).values
                    logger.warning(
                        f"Missing longitude values in variable '{key}': {missing_lons}"
                    )
                    raise ValueError(
                        f"Missing longitude values in variable '{key}': {missing_lons}"
                    )
                elif check_first_variable_only:
                    logger.debug(
                        f"No missing longitude values found in the first variable [{key}] of the dataset."
                    )
                    break

        logger.debug(
            f"No missing longitude values found along longitude for all variables of the dataset."
        )
        return ds


class ZarrYearValidator:
    """
    Checks if the year group in a zarr store has the full number of timesteps and
    matches the expected grid shape.
    """

    def __init__(
        self,
        spatial_res=0.25,
        time_window="8D",
        error_handling: Literal["raise", "warn", "ignore"] = "raise",
    ):
        from .date_utils import DateWindows
        from .processors import make_target_global_grid

        self.spatial_res = spatial_res
        self.time_window = time_window
        self.target_grid = make_target_global_grid(spatial_res)
        self.date_windows = DateWindows(window_span=self.time_window)
        self.error_handling = error_handling

    def __call__(
        self,
        ds: xr.Dataset,
        error_handling: Literal["raise", "warn", "ignore", None] = None,
    ) -> bool:
        """
        Validates the dataset for the year group in a zarr store.

        Parameters
        ----------
        ds : xr.Dataset
            The dataset to validate.
        error_handling : Literal['raise', 'warn', 'ignore', None], optional
            How to handle errors. If 'raise', raises an error on failure.
            If 'warn', logs a warning on failure. If 'ignore', logs a trace message on failure.
            If None, uses the default error handling set in the class.

        Returns
        -------
        bool
            True if the dataset is valid, False otherwise.
        """
        return self.is_valid(ds, error_handling=error_handling)

    def is_valid(
        self,
        ds: xr.Dataset,
        error_handling: Literal["raise", "warn", "ignore", None] = None,
    ) -> bool:
        """
        Validates the dataset for the year group in a zarr store.

        Parameters
        ----------
        ds : xr.Dataset
            The dataset to validate.
        error_handling : Literal['raise', 'warn', 'ignore', None], optional
            How to handle errors. If 'raise', raises an error on failure.
            If 'warn', logs a warning on failure. If 'ignore', logs a trace message on failure.
            If None, uses the default error handling set in the class.

        Returns
        -------
        xr.Dataset
            The validated dataset.
        """
        checks = [
            self.is_single_year,
            self.is_full_year,
            self.is_correct_shape,
        ]

        if error_handling is None:
            error_handling = self.error_handling  # type: ignore #

        results = []
        for check in checks:
            success, message = check(ds)
            # do complex matching, where error_handling must be x and success must be false
            match error_handling:
                case "raise" if not success:
                    raise ValueError(message)
                case "warn" if not success:
                    logger.warning(message)
                    results.append(False)
                case "ignore" if not success:
                    logger.trace(message)
                    results.append(True)
                case _ if success:
                    logger.trace(message)
                    results.append(True)
                case _:
                    raise ValueError(
                        f"Invalid error_handling option: {error_handling}. Must be 'raise', 'warn', or 'ignore'."
                    )

        return all(results)

    def is_single_year(self, ds: xr.Dataset) -> tuple[bool, str]:
        year_arr = ds.time.dt.year.values
        n_years = np.unique(year_arr).size
        if n_years != 1:
            return False, f"Dataset contains {n_years} years, expected 1 year."
        else:
            return True, f"Dataset contains {n_years} year, as expected."

    def is_full_year(self, ds: xr.Dataset) -> tuple[bool, str]:
        year = int(ds.time.dt.year.values[0])

        n_expected = self.date_windows.get_bin_centers(year).size
        n_actual = ds.sizes["time"]

        if n_actual != n_expected:
            return (
                False,
                f"Dataset contains {n_actual} timesteps, expected {n_expected} for year {year}.",
            )
        else:
            return (
                True,
                f"Dataset contains {n_actual} timesteps, as expected for year {year}.",
            )

    def is_correct_shape(self, ds: xr.Dataset) -> tuple[bool, str]:
        lon = ds["lon"].values
        lat = ds["lat"].values

        target_lon = self.target_grid.lon
        target_lat = self.target_grid.lat

        if lon.shape != target_lon.shape or lat.shape != target_lat.shape:
            return (
                False,
                f"Dataset shape does not match expected shape. Expected {target_lon.shape}x{target_lat.shape}, got {lon.shape}x{lat.shape}.",
            )
        else:
            return (
                True,
                f"Dataset shape matches expected shape: {lon.shape}x{lat.shape}.",
            )
