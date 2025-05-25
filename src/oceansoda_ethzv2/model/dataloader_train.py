"""Contains torch data loaders for training dataset (colocated SOCAT)"""

import pathlib

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class ColocatedDataset(Dataset):
    """
    Custom dataset for loading colocated SOCAT data.
    """

    def __init__(self, data_dir, subset, split="train", transform=None):
        """
        Args:
            data_dir (string): Directory with all the data files.
            split (string): One of 'train', 'val', or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.subset = subset
        self.split = split
        self.transform = transform

        # Load the data
        self.data = self.load_data()

    def load_data(self):
        """
        Load the data from the specified directory and subset.
        """
        data_path = pathlib.Path(self.data_dir) / self.split / f"{self.subset}.csv"
        if not data_path.exists():
            raise FileNotFoundError(f"Data file {data_path} does not exist.")

        # Load the data into a DataFrame
        data = pd.read_csv(data_path)

        # Convert DataFrame to numpy array
        data = data.to_numpy(dtype=np.float32)

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        """
        sample = self.data[idx]

        # Apply transform if specified
        if self.transform:
            sample = self.transform(sample)

        return sample
