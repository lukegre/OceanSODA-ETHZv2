import functools
import pathlib
import pprint
from dataclasses import dataclass
from typing import Optional, Union

import munch
import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger
from memoization import cached
from pydantic import BaseModel, Field

from .utils.zarr_utils import ZarrDataset


class CMEMSEntry(BaseModel):
    id: str = Field(..., description="CMEMS dataset ID")
    product: str = Field(..., description="CMEMS product name")
    time_start: str | None = Field(
        None,
        description="Start time of the dataset, will clip the dataset, None means no clipping",
    )
    time_end: str | None = Field(
        None,
        description="End time of the dataset, will clip the dataset, None means no clipping",
    )
    flag: int | None = Field(
        None,
        description="Flag value to be added to the dataset along the time dimension",
    )
    vars: dict[str, str] | None = Field(
        {},
        description="Variables to rename in the dataset, e.g. {'analysed_sst': 'sst'}",
    )


class CMEMSDataset(ZarrDataset):
    def __init__(
        self, entries: list[dict], spatial_res: float = 0.25, temporal_res: str = "8D"
    ):
        """
        Initialize the CMEMSDataset with properties.
        Properties should match the CMEMSEntry model.
        """
        for entry in entries:
            is_valid_entry(entry, CMEMSEntry)

        self.entries = entries
        self.spatial_res = spatial_res  # Default spatial resolution in degrees
        self.temporal_res = temporal_res  # Default window size for temporal resolution

    def _open_full_dataset(self) -> xr.Dataset:
        """
        Open the CMEMS dataset based on the properties defined in this instance.
        """
        return open_cmems_datasets(self.entries)

    def _regrid_data(self, ds) -> xr.Dataset:
        from .processors import standard_regridding

        return standard_regridding(
            ds_in=ds,
            spatial_res=self.spatial_res,
            window_size=self.temporal_res,
        )


class CMEMSCatalog:
    def __init__(
        self,
        datasets: dict[str, CMEMSDataset],
        save_path: str = "../data/{temporal_res}_{spatial_res}/{{var}}-{temporal_res}_{spatial_res}.zarr",
        spatial_res: float = 0.25,  # Default spatial resolution in degrees
        temporal_res: str = "8D",  # Default window size for temporal resolution
        **kwargs,  # additional keyword arguments for filename formatting - available in locals()
    ):
        """
        Initialize the CMEMSCatalog with a dictionary of datasets.
        The keys are dataset names and the values are CMEMSDataset instances.
        """
        self.datasets = munch.Munch(datasets)
        self.spatial_res = spatial_res  # Default spatial resolution in degrees
        self.temporal_res = temporal_res  # Default window size for temporal resolution
        spatial_res = str(spatial_res).replace(".", "")  # type: ignore - convert to string without dot for filename
        self.save_path = save_path.format(**locals())

    @classmethod
    def from_yaml(cls, yaml_file: str) -> "CMEMSCatalog":
        """
        Load a CMEMSCatalog from a YAML file.
        The YAML file should contain a dictionary of dataset entries.
        """
        import yaml

        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f)

        datasets = {name: CMEMSDataset(entries) for name, entries in data.items()}
        return cls(datasets)

    def write_timestep_to_disk(self, year: int, index: int, progress=False) -> str:
        """
        Write the dataset for a specific year and index to disk.
        The dataset is saved in Zarr format at the specified save path.
        """
        from .utils.zarr_utils import save_vars_to_zarrs

        ds = self._get_timestep((year, index))
        if ds is None:
            raise ValueError(f"No data available for year {year} and index {index}.")

        # Save the dataset to Zarr
        save_vars_to_zarrs(
            ds,
            self.save_path,
            group_by_year=True,
            progress=progress,
            error_handling="warn",
        )

        return self.save_path

    def get(self, year: int, index: int, progress=False) -> xr.Dataset:
        """
        Get a dataset by name or by year and index.
        This method is a convenience wrapper around the __getitem__ method.
        """

        try:
            self.write_timestep_to_disk(year, index, progress=progress)
        except:
            pass
        finally:
            return self._get_timestep(
                (
                    year,
                    index,
                )
            )

    @functools.lru_cache(maxsize=2)
    def _get_timestep(self, key: tuple[int, int]) -> xr.Dataset:
        """
        Internal method to get a specific timestep from a dataset.
        This is a convenience method for accessing datasets by year and index.
        """
        year, index = key
        ds_all = []
        for product, dataset in self.datasets.items():
            ds = dataset.get_time_stride_regridded(year=year, index=index)
            ds_size = np.array(list(ds.sizes.values())).prod()

            if ds_size == 0:
                logger.warning(f"{product} : [{year}, {index}] is empty. Skipping.")
                continue
            else:
                ds_all += [ds]
        ds = xr.merge(ds_all)
        return ds

    def __getitem__(self, key: Union[str, tuple[int, int]]) -> xr.Dataset:
        """
        Allow access to datasets by name or by year and index.
        If key is a string, return the dataset with that name.
        If key is a tuple (year, index), return the dataset for that year and index.
        """
        if isinstance(key, tuple) and len(key) == 2:
            return self._get_timestep(key)
        else:
            raise NotImplementedError(
                "Only tuple keys of the form (year, index) are supported for CMEMSCatalog."
            )


@cached(max_size=7)
def open_cmems_dataset(entry: dict) -> xr.Dataset:
    from copernicusmarine import open_dataset

    from .processors import rename_and_drop

    is_valid_entry(entry, CMEMSEntry)

    logger.debug(
        "Opening CMEMS dataset with properties: \n{}",
        pprint.pformat(entry, sort_dicts=False),
    )
    ds_all = open_dataset(entry["id"])

    t0 = entry.get("time_start", None)
    t1 = entry.get("time_end", None)
    ds_sub = ds_all.sel(time=slice(t0, t1))

    ds_sub = ds_sub.pipe(rename_and_drop, entry.get("vars", {}))

    if "flag" in entry:
        key = f"flag_{entry['product']}"
        ds_sub[key] = make_flag_along_dim(ds_sub, entry["flag"])

    return ds_sub


def open_cmems_datasets(entry_list: list[dict]) -> xr.Dataset:
    if len(entry_list) == 0:
        raise ValueError("entry_list cannot be empty")
    ds_list = [open_cmems_dataset(entry) for entry in entry_list]
    ds = xr.concat(ds_list, dim="time")
    return ds


def make_flag_along_dim(
    ds: xr.Dataset, flag_value, flag_dim="time", flag_name="flag"
) -> xr.DataArray:
    """Create a flag variable along a specified dimension."""
    flag_values = [flag_value] * ds[flag_dim].size
    return xr.DataArray(flag_values, dims=[flag_dim])


# check if entry is a valid CMEMSEntry
def is_valid_entry(entry, model):
    try:
        model(**entry)
    except Exception as e:
        missing_entry = str(e).splitlines()[1]
        expected = pprint.pformat(CMEMSEntry.model_fields, sort_dicts=False)
        raise ValueError(
            f"Invalid or missing entry: [{missing_entry}] \n\nPlease check the "
            "CMEMS entry dictionary to make sure it matches the following model:\n"
            f"{expected}"
        )
