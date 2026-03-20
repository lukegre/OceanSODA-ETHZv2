import copernicusmarine
import xarray as xr
from loguru import logger
from pydantic import BaseModel, Field

from oceansoda_ethzv2.data.cmems import make_flag_along_dim


class CMEMSEntry(BaseModel):
    id: str = Field(..., description="CMEMS dataset ID")
    product: str = Field(..., description="CMEMS product name")
    time_start: str | None = Field(None, description="Start time of fetching")
    time_end: str | None = Field(None, description="End time of fetching")
    flag: int | None = Field(
        None,
        description="Flag value to be added to the dataset along the time dimension",
    )
    vars: dict[str, str] = Field({}, description="renames variables in the dataset")
    # check that login dictionary has "username" and "password" keys that both have strings as values


def open_cmems_dataset(entry: CMEMSEntry) -> xr.Dataset:

    logger.info(f"Opening CMEMS dataset [{entry.id}]")
    logger.debug("Opening CMEMS dataset with properties: \n{}", entry)

    # Suppress copernicusmarine logging
    ds_all = copernicusmarine.open_dataset(entry["id"])

    ds_sub = ds_all.sel(time=slice(entry.time_start, entry.time_end))

    ds_sub = ds_sub.rename(entry.vars)
    ds_sub = ds_sub[list(entry.vars.values())]

    if "flag" in entry:
        key = f"flag_{entry['product']}"
        ds_sub[key] = make_flag_along_dim(ds_sub, entry["flag"])

    return ds_sub
