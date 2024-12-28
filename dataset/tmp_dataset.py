import torch
import torch.utils.data
import numpy as np
import pickle
from dataset.dataset import read_data, binary


class AdderDataset(torch.utils.data.dataset.Dataset):
    # from state to next output
    def __init__(self, data_dir, split, nbits=64):
        super().__init__()
        self.next = 3
        self.data = read_data(data_dir, nbits=nbits).reshape(-1, 3)
        scaled_split = [int(len(self.data) * p) for p in split]
        self.size = scaled_split[1] - scaled_split[0] - self.next
        self.data = self.data[scaled_split[0]:scaled_split[1], :]
        print(f"Load data from {data_dir}, split {split}, total data {self.size}")

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        assert 0 <= item < self.size
        return torch.tensor(binary(self.data[item, :2], nbits=64), dtype=torch.uint8), torch.tensor(
            binary(self.data[item, 2], nbits=64), dtype=torch.uint8)
