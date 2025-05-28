import xarray as xr

from . import open_zarr_groups
from .utils import ZarrYearValidator


def calc_seasonal_climatology(
    zarr_group_name: str,
    aggregator_dim="time.dayofyear",
    validator=ZarrYearValidator(),
):
    """
    Compute the seasonal climatology of a dataset along the specified dimension.

    Parameters:
        ds (xr.Dataset): The input dataset.
        aggregator_dim (str): The dimension to aggregate over, typically 'time.dayofyear'.

    Returns:
        xr.Dataset: The dataset with seasonal climatology computed.
    """
    ds = open_zarr_groups(
        zarr_group_name, concat_dim="dayofyear", group_validator=validator
    )
    ds = ds.groupby(aggregator_dim).mean(dim="time")
    ds = ds.assign_attrs(processing_seasonal_climatology=True)
    return ds


def calc_longterm_mean(ds: xr.Dataset, dim="time"):
    """
    Compute the long-term mean of a dataset along the specified dimension.

    Parameters:
        ds (xr.Dataset): The input dataset.
        dim (str): The dimension to compute the mean over, typically 'time'.

    Returns:
        xr.Dataset: The dataset with long-term mean computed.
    """
    return ds.mean(dim=dim).assign_attrs(processing_longterm_mean=True)


def calc_climatological_seasonal_anomaly(
    ltm: xr.DataArray,
    clim: xr.DataArray,
    aggregator_dim="time.dayofyear",
):
    """
    Compute the seasonal anomaly of a dataset relative to a climatology.

    Parameters:
        ds (xr.Dataset): The input dataset.
        climatology (xr.Dataset): The climatology dataset to compute anomalies against.
        aggregator_dim (str): The dimension to aggregate over, typically 'time.dayofyear'.

    Returns:
        xr.Dataset: The dataset with seasonal anomalies computed.
    """
    return (clim - ltm).assign_attrs(processing_climatological_seasonal_anomaly=True)


def successive_filling_with_flagging(
    *args: xr.DataArray,
    flag_name: str | None = None,
) -> xr.Dataset:
    """
    given multiple DataArrays, fill the first with the second.
    Then take the output from the first step and fill it with the third, and so on.

    Parameters:
        *args (xr.DataArray): DataArrays to be filled successively.
        flag_name (str): Name of the flag variable to indicate filled values.

    Returns:
        xr.Dataset: A dataset containing the filled DataArrays with flags.
        Flags are 1 for first DataArray, 2 for second, etc.
    """

    if len(args) < 2:
        raise ValueError("At least two DataArrays are required for successive filling.")

    name = getattr(args[0], "name", "original_data")
    flag_name = flag_name or f"{name}_fill_flags"

    filled = args[0].copy()

    flags = None
    flag_labels = {1: name}
    for i in range(1, len(args)):
        next_data = args[i]
        filled, flags = fill_a_with_b_with_flags(filled, next_data, flags=flags)
        flag_labels[i + 1] = next_data.name if next_data.name else f"filled_{i + 1}"

    ds = xr.Dataset()
    ds[f"{name}_filled"] = filled.assign_attrs(
        processing_successive_filling_with_flagging=True
    )
    ds[flag_name] = flags.assign_attrs(flag_labels=str(flag_labels))  # type: ignore

    return ds


def fill_a_with_b_with_flags(
    a: xr.DataArray,
    b: xr.DataArray,
    flags: None | xr.DataArray = None,
) -> tuple[xr.DataArray, xr.DataArray]:
    mask = a.isnull() & b.notnull()
    filled = a.where(~mask, b)

    if isinstance(flags, xr.DataArray):
        flags_max = flags.max()
        flags = flags.where(~mask, flags_max + 1)
    else:
        flags = a.notnull().astype("float32")
        flags = flags.where(~mask, 2)

    return filled, flags
