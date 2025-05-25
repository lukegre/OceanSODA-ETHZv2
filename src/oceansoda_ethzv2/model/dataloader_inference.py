"""
Contains torch data loaders for inference dataset (gridded satellite data)
"""

import pathlib
from typing import Optional

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset


class GriddedDataset(Dataset):
    """
    Custom dataset for loading gridded satellite data.
    """

    def __init__(
        self,
        data_path: str,
        clim_path: str,
        seas_path: str,
        transform: Optional[callable] = None,
    ):
        """
        Args:
            data_path (string): Path to the gridded satellite data file.
            clim_path (string): Path to the climatology file.
            seas_path (string): Path to the seasonality file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_path = data_path
        self.clim_path = clim_path
        self.seas_path = seas_path
        self.transform = transform

        self._data = None
        self._clim = None
        self._seas = None

    def load_data(self, year):
        """
        Load the data from the specified file.
        """
        # Load the gridded data
        data = xr.open_zarr(self.data_path, consolidated=True)

    def calc_dt(self):
        """
        calculate the different between t and (t-1)
        """
