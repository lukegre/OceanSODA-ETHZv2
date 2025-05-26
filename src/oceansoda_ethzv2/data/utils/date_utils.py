import pathlib
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr


class DateWindows:
    """
    Class to generate date windows for a given year.

    This class allows you to create date windows for a specified year or
    a range of years. The windows are defined by a specified frequency (e.g., '8D').

    Methods include:
    - get_bin_edges: Get the edges of the date bins.
    - get_bin_centers: Get the centers of the date bins.
    - get_window_dates: Get the date window for a specific time or year and index.
    - get_index: Get the time step index for a specified date.
    - get_adjacent: Get the previous and next time steps for a given year and index.
    - get_most_recent(): Get the most recent time step based on the current date.

    """

    def __init__(
        self, start_day_of_year: int = 1, window_span: str = "8D", time_offset="12h"
    ):
        """Initiate the object from which
        periods can be generated

        Parameters
        ----------
        start_day : int or iterable, optional
            the start day for each year, by default 1
        freq : str, optional
            how large are windows, by default '8D'.
            uses pd.TimeDelta strings
        """
        self.freq = window_span
        self.time_offset = time_offset
        self.today = pd.Timestamp.today()

        self.ndays = pd.Timedelta(window_span).days
        self.nhours = pd.Timedelta(window_span).seconds // 3600

    def get_bin_edges(self, year: Union[int, list[int]]) -> pd.DatetimeIndex:
        return get_date_bins(year, self.freq)[0]

    def get_bin_centers(self, year: Union[int, list[int]]) -> pd.DatetimeIndex:
        return get_date_bins(year, self.freq)[1]

    def get_window_center(
        self,
        time: Optional[pd.Timestamp | str] = None,
        year: Optional[int] = None,
        index: Optional[int] = None,
    ) -> pd.Timestamp:
        self._check_optionals_groupings((time,), (year, index))
        if time is not None:
            year, index = self.get_index(time=time)
        elif year is None or index is None:
            raise ValueError("Either time or (year + index) must be specified")

        bin_centers = self.get_bin_centers(year)
        window_center = bin_centers[index]

        return window_center

    def get_window_edges(
        self,
        time: Optional[pd.Timestamp | str] = None,
        year: Optional[int] = None,
        index: Optional[int] = None,
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        """
        Get the minimum and maximum dates for the date window of a specified year or years.

        Args:
            year (int or list of int): The year or years for which to get the date window.

        Returns:
            tuple[pd.Timestamp, pd.Timestamp]: A tuple containing the minimum and maximum dates of the window.
        """
        self._check_optionals_groupings((time,), (year, index))
        if time is not None:
            year, index = self.get_index(time=time)
        elif year is None or index is None:
            raise ValueError("Either time or (year + index) must be specified")

        bin_edges = self.get_bin_edges(year)

        if (index + 1) >= bin_edges.size:
            raise ValueError(
                f"index {index} is out of bounds for year {year}. Must be < {bin_edges.size - 1}"
            )

        left = bin_edges[index]
        right = bin_edges[index + 1] - pd.Timedelta(seconds=1)
        return left, right

    def get_window_dates(
        self,
        time: Optional[pd.Timestamp | str] = None,
        year: Optional[int] = None,
        index: Optional[int] = None,
    ) -> pd.DatetimeIndex:
        """
        Args:
            time (pd.Timestamp, optional): Date. Defaults to None. Must be specified on its own
            year (int, optional): Year. Defaults to None. Must be specified with index
            index (int, optional): Time step of year. Defaults to None. Must be specified with year

        Returns:
            pd.DatetimeIndex: The date window for the specified time or year and index.

        Raises:
            ValueError: If neither (time) nor (year, index) are specified.

        """
        self._check_optionals_groupings((time,), (year, index))

        if time is not None:
            return get_window_from_datetime(
                time, offset=self.time_offset, freq=self.freq
            )
        elif year is not None and index is not None:
            return get_window_from_year_index(
                year, index, offset=self.time_offset, freq=self.freq
            )
        else:
            raise ValueError("Either time or (year + index) must be specified")

    def get_index(
        self,
        time: Optional[pd.Timestamp | str] = None,
        year: Optional[int] = None,
        dayofyear: Optional[int] = None,
        index: Optional[int] = None,
    ) -> tuple[int, int]:
        """
        Get the time_step index for a specified date.

        Parameters
        ----------
        time : pandas.Timestamp, optional
            A timestamp from which to extract the year and day-of-year.
        year : int, optional
            The calendar year of the date. Must be provided if `time` is not.
        dayofyear : int, optional
            The day of the year (1-366). Must be provided if `time` is not.

        Returns
        -------
        tuple[int, int]
            A tuple (year, index) where `index` is the zero-based bin index
            into the array of day-of-year bin edges for the specified year.

        Raises
        ------
        ValueError
            If neither `time` nor both `year` and `dayofyear` are specified.
        ValueError
            If `dayofyear` is not in the range 1 to 366.
        """

        if index is not None and year is not None:
            return year, index

        self._check_optionals_groupings(
            (time,),
            (
                year,
                dayofyear,
            ),
        )

        if time is not None:
            if isinstance(time, str):
                time = pd.Timestamp(time)
            year = time.year
            dayofyear = time.dayofyear
        elif year is None or dayofyear is None:
            raise ValueError("Either time or (year + dayofyear) must be specified")
        elif (dayofyear == 0) or (dayofyear > 366):
            raise ValueError("dayofyear must be > 0, <= 366")

        edges = self.get_bin_edges(year).dayofyear[:-1]
        index = np.where(dayofyear >= edges)[0].max().item()

        return year, index

    def get_adjacent(
        self, year: int, index: int
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """
        Get the previous and next time steps for a given year and index.

        Args:
            year (int): The year for which to get the adjacent time steps.
            index (int): The index of the time step within the year.

        Returns:
            tuple[tuple[int,int], tuple[int,int]]:
                A tuple containing two tuples:
                - The first tuple is the previous time step (year, index).
                - The second tuple is the next time step (year, index).

        Raises:
            ValueError: If the index is out of bounds for the year.
        """
        n = self.get_bin_centers(year).size

        # if index is > n, then we are in the next year
        if (index >= n) or (index < 0):
            raise ValueError(
                f"index must be < num steps in year ({n}), but got {index}"
            )
        elif index == (n - 1):  # if index is n, then we are at the end of the year
            prev = (year, index - 1)
            next = (year + 1, (n - 1) - index)
        elif index == 0:  # if index is 0, then we are at the start of the year
            prev = (year - 1, n - 1)
            next = (year, index + 1)
        else:  # if index is in the current year
            prev = (year, index - 1)
            next = (year, index + 1)

        return prev, next

    def get_most_recent(self, buffer="8D") -> tuple[int, int]:
        """
        Get the most recent time step based on the current date.

        Args:
            buffer (str, optional): The time buffer to subtract from the current date.
                Defaults to '8D'.

        Returns:
            tuple[int,int]: A tuple (year, index) representing the most recent time step.
        """
        today = pd.Timestamp.today() - pd.Timedelta(buffer)
        year, index = self.get_index(time=today)

        return year, index

    def _check_optionals_groupings(self, *args) -> None:
        """
        Ensure exactly one group of optional parameters is provided to the caller.
        This helper inspects the call stack to find the parent function's name and its
        keyword argument order, then checks the provided argument groups to confirm
        that exactly one group contains only non-None values. Other groups must be
        entirely None or omitted.
        Parameters:
            *args: tuple of iterables
                Each iterable represents a set of related optional arguments from the
                parent function. A group is considered "provided" if all values in that
                iterable are non-None.
        Raises:
            ValueError:
                If inspection of the call stack fails (must be called by a parent function)
                If not exactly one argument group is fully non-None.
                The error message will mention the parent function name and list
                of valid argument groupings.
        """

        import inspect

        # get the name of the parent function
        call_frame = inspect.currentframe()
        if call_frame is None:
            raise ValueError("No parent function found")
        parent_function = call_frame.f_back
        if parent_function is None:
            raise ValueError("No parent function found")

        parent_name = parent_function.f_code.co_name
        kwarg_names = parent_function.f_code.co_varnames[1:]

        kwarg_groups = []
        groups_passed = []
        for i, group in enumerate(args):
            all_in_group_passed = all([True if v is not None else False for v in group])
            groups_passed.append(all_in_group_passed)

            # groupw kwarg names like the args
            group_names = ()
            for j in range(len(group)):
                i += j
                group_names += (kwarg_names[i],)
            kwarg_groups.append(group_names)
        kwarg_groups = str(kwarg_groups).replace("'", "")[1:-1]

        num_groups_passed = sum(groups_passed)
        if num_groups_passed != 1:
            raise ValueError(
                f"Only one group of optional arguments can be passed "
                f"at a time - check `{parent_name}` inputs: [ {kwarg_groups} ]"
            )


def _get_date_bins_for_year(
    year: int, freq="8D", clip_to_today=True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate date bins and their corresponding centers for a given year.
    This function creates evenly spaced date bins within a specified year,
    based on the provided frequency. It also calculates the center points
    of each bin.
    Parameters:
    -----------
    year : int
        The year for which the date bins are to be generated.
    freq : str, optional
        The frequency string (e.g., '8D' for 8 days) to define the spacing
        of the bins. Default is '8D'.
    Returns:
    --------
    tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - bin_edges: An array of datetime objects representing the edges of the bins. Edges are left-inclusive, right-exclusive.
        - bin_centers: An array of datetime objects representing the centers of the bins.
    Raises:
    -------
    AssertionError
        If the size of bin_edges is not exactly one more than the size of bin_centers.
    Notes:
    ------
    The function ensures that the bins cover the entire year, with an additional
    edge at the start of the following year to close the last bin.
    """
    time_8D_y0 = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq=freq).to_numpy()
    time_8D_y1 = pd.Timestamp(f"{year + 1}-01-01").to_numpy()

    bin_edges = np.concatenate([time_8D_y0, [time_8D_y1]])
    if clip_to_today:
        bin_edges = bin_edges[bin_edges <= pd.Timestamp.today()]

    diff = bin_edges[1:] - bin_edges[:-1]
    bin_centers = bin_edges[:-1] + diff / 2

    assert bin_edges.size == (bin_centers.size + 1)

    return bin_edges, bin_centers


def get_date_bins(
    years: Union[int, list[int]], freq="8D"
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """
    Generate date bins with specified frequency for given years.

    This function creates date bin edges and centers for the specified years
    using the given frequency. It supports both single year (int) and multiple
    years (list of ints) as input.

    Args:
        years (Union[int, list[int]]): A single year (int) or a list of years (list of ints)
            for which the date bins are to be created.
        freq (str, optional): Frequency string (e.g., '8D' for 8 days) to define
            the bin intervals. Defaults to '8D'.
        limit (pd.Timestamp, optional): The maximum date for the bins.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - bin_edges (np.ndarray): Array of bin edges.
            - bin_centers (np.ndarray): Array of bin centers.

    Raises:
        TypeError: If `years` is not an int or a list of ints.

    Notes:
        - The function ensures that the last edge of the final year is included
          in the bin edges.
        - The size of `bin_edges` will always be one more than the size of
          `bin_centers`.
    """
    if isinstance(years, int):
        years = [years]
    elif isinstance(years, list):
        years = years
    else:
        raise TypeError("years must be int or list of ints")

    bin_edges = []
    bin_centers = []
    for y in years:
        edges, centers = _get_date_bins_for_year(y, freq, clip_to_today=True)
        bin_edges.append(edges[:-1])
        bin_centers.append(centers)

        last_edge = edges[-1]

    bin_edges.append([last_edge])  # final year needs the last edge

    bin_edges = np.concatenate(bin_edges)
    bin_centers = np.concatenate(bin_centers)

    assert bin_edges.size == (bin_centers.size + 1), (
        "bin_edges and bin_centers sizes do not match"
    )

    bin_edges = pd.DatetimeIndex(bin_edges)
    bin_centers = pd.DatetimeIndex(bin_centers)

    return bin_edges, bin_centers


def get_window_from_year_index(
    year: int, index: int, offset="12h", freq="8D"
) -> pd.DatetimeIndex:
    """
    Get the date window for a specific year and index, where
    the window is the range of dates between two bin edges.

    Args:
        year (int): The year for which to get the date window.
        index (int): The index of the date window within the year.

    Returns:
        pd.DatetimeIndex: The dates for the window for the specified year and index.
    """
    bin_edges, _ = get_date_bins(year, freq=freq)
    t0 = bin_edges[index]
    t1 = bin_edges[index + 1]

    window = pd.date_range(t0, t1, inclusive="left")
    window_with_offset = window + pd.Timedelta(offset)

    return window_with_offset


def get_window_from_datetime(
    date: pd.Timestamp, offset="12h", freq="8D"
) -> pd.DatetimeIndex:
    """
    Get the date window for a specific date, where
    the window is the range of dates between two bin edges.

    Args:
        date (pd.Timestamp): The date for which to get the date window.

    Returns:
        pd.DatetimeIndex: The dates for the window for the specified date.
    """
    year = date.year
    bin_edges, _ = get_date_bins(year, freq=freq)
    index = np.where(date >= bin_edges)[0].max()

    window = get_window_from_year_index(year, index, offset=offset)

    return window
