import torch
import torch.utils.data
import numpy as np
import pickle
import math


def read_data(data_chunk, nbits=12):
    if nbits == 8:
        data = np.fromfile(data_chunk, dtype=np.uint8)
    elif nbits == 12:
        data = np.fromfile(data_chunk, dtype=np.uint8)
        fst_uint8, mid_uint8, lst_uint8 = np.reshape(data, (data.shape[0] // 3, 3)).astype(np.uint16).T
        fst_uint12 = (fst_uint8 << 4) + (mid_uint8 >> 4)
        snd_uint12 = ((mid_uint8 % 16) << 8) + lst_uint8
        return np.reshape(np.concatenate((fst_uint12[:, None], snd_uint12[:, None]), axis=1), 2 * fst_uint12.shape[0])
    elif nbits == 16:
        data = np.fromfile(data_chunk, dtype=np.uint16)
    elif nbits == 32:
        data = np.fromfile(data_chunk, dtype=np.uint32)
    elif nbits == 64:
        data = np.fromfile(data_chunk, dtype=np.uint64)
    elif nbits == 128:
        # currently numpy does not support it, so we use pickle
        with open(data_chunk, "rb") as f:
            data = pickle.load(f)

    else:
        assert 0, "nbits must be 8, 12, 16, 32 or 64"
    return data


def binary(x, nbits=8):
    if isinstance(x, list):
        return np.array([binary(i, nbits) for i in x])
    elif isinstance(x, int):
        # in reverse order, so that it is convenient for rnn based models
        return [1 if digit == '1' else 0 for digit in bin(x)[2:].zfill(nbits)][::-1]
        # binary_array = np.array([1 if digit == '1' else 0 for digit in bin(x)[2:]])
        # return np.pad(binary_array, (nbits - len(binary_array), 0), 'constant', constant_values=0)
    else:
        assert isinstance(x, (np.ndarray, np.generic)), type(x)
        shape = x.shape
        if len(x.shape) == 0:
            x = x.reshape(-1)
        # return np.unpackbits(x.view(np.uint8), bitorder='little')[..., ::-1].copy()
        data = np.unpackbits(x.view(np.uint8), bitorder='little').reshape(*shape, -1)
        # return data[..., ::-1].copy()
        # in reverse order, so that it is convenient for rnn based models
        return data.copy()


class Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, data_dir, split, seqlen=30, nbits=8):
        super().__init__()
        self.size = 0
        self.index = []
        self.split = split
        self.nbits = nbits
        self.seqlen = seqlen
        if not hasattr(self, 'next'):
            self.next = seqlen
        self.data = self.fetch_data(data_dir)
        self.split_data()
        self.set_index()
        print(f"Data loaded from {data_dir}, split {split}, total data {self.size}")

    def set_index(self):
        self.index = np.arange(self.seqlen)

    def fetch_data(self, data_dir):
        return read_data(data_dir, self.nbits)

    def split_data(self):
        scaled_split = [int(len(self.data) * p) for p in self.split]
        self.data = self.data[scaled_split[0]:scaled_split[1]]
        self.size = scaled_split[1] - scaled_split[0] - self.next

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        assert 0 <= item < self.size
        return torch.tensor(binary(self.data[item + self.index], self.nbits)), torch.tensor(
            binary(self.data[item + self.next], self.nbits))


class MTGPTDataset(Dataset):
    def __init__(self, data_dir, split, seqlen=1248):
        self.next = 1248
        super().__init__(data_dir, split, seqlen, nbits=16)

    def fetch_data(self, data_dir):
        return read_data(data_dir, 16)

    def set_index(self):
        # self.index = [i for i in range(self.seqlen)]
        self.index = np.arange(self.seqlen)

    def __getitem__(self, item):
        assert 0 <= item < self.size
        x = self.data[item + self.index].astype(int)
        y = binary(self.data[item + self.next - 1: item + self.next + 1], self.nbits).reshape(-1)
        return torch.tensor(x, dtype=torch.long), torch.tensor(y)


class MTPreloadDataset(Dataset):
    def __init__(self, data_dir, split, seqlen=624):
        self.next = 624
        super().__init__(data_dir, split, seqlen, nbits=32)
        # reload data so that it is pre computed
        self.data = binary(self.data, 32)
        print(self.data.shape)

    def fetch_data(self, data_dir):
        return read_data(data_dir, 32)

    def set_index(self):
        if self.seqlen > 398:
            self.index = [i for i in range(self.seqlen)]
        else:
            self.index = [i for i in range(self.seqlen - 1)]
            self.index.append(397)
        self.index = np.array(self.index)

    def __getitem__(self, item):
        assert 0 <= item < self.size
        return torch.tensor(2. * self.data[item + self.index] - 1., dtype=torch.float), torch.tensor(
            self.data[item + self.next])


class MTDataset(Dataset):
    def __init__(self, data_dir, split, seqlen=624):
        self.next = 624
        super().__init__(data_dir, split, seqlen, nbits=32)

    def fetch_data(self, data_dir):
        return read_data(data_dir, 32)

    def set_index(self):
        if self.seqlen > 398:
            self.index = [i for i in range(self.seqlen)]
        else:
            self.index = [i for i in range(self.seqlen - 1)]
            self.index.append(397)
        self.index = np.array(self.index)

    def __getitem__(self, item):
        assert 0 <= item < self.size
        return torch.tensor(binary(self.data[item + self.index], self.nbits), dtype=torch.float), torch.tensor(
            binary(self.data[item + self.next], self.nbits), dtype=torch.float)


class TemperDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, data_dir, register_data_dir, split):
        super().__init__()
        self.data = np.fromfile(data_dir, dtype=np.uint32)
        self.register_data = np.fromfile(register_data_dir, dtype=np.uint32)
        scaled_split = [int(len(self.data) * p) for p in split]
        self.size = scaled_split[1] - scaled_split[0]
        self.data = self.data[scaled_split[0]:scaled_split[1]]
        self.register_data = self.register_data[scaled_split[0]:scaled_split[1]]
        print(f"Load data from {data_dir}, split {split}, total data {self.size}")
        print(f"Load data from {register_data_dir}, split {split}, total data {self.size}")

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        assert 0 <= item < self.size
        return torch.tensor(binary(self.data[item])), torch.tensor(binary(self.register_data[item]))


class LehmerDataset(Dataset):
    def __init__(self, data_dir, state_data_dir, split, seqlen=3, nbits=128):
        self.next = 3
        super().__init__(data_dir, split, 3, nbits=64)

    def __getitem__(self, item):
        assert 0 <= item < self.size
        # return torch.tensor(binary(self.data[item:item + self.next], nbits=64), dtype=torch.uint8), torch.tensor(
        #     binary(self.data[item + self.next], nbits=64), dtype=torch.uint8)
        return binary(self.data[item:item + self.next], nbits=64), binary(self.data[item + self.next], nbits=64)
        # return torch.tensor(np.log2(self.data[item + self.index]), dtype=torch.double) - 62.5, torch.tensor(
        #     np.log2(self.data[item + self.next]), dtype=torch.double) - 62.5, torch.tensor(
        #     binary(self.data[item + self.next], nbits=128), dtype=torch.uint8)


class LehmerForwardDataset(torch.utils.data.dataset.Dataset):
    # from state to next output
    def __init__(self, data_dir, state_data_dir, split, seqlen=1, nbits=128):
        super().__init__()
        self.next = 3
        self.data = read_data(data_dir, nbits=64)
        self.state_data = read_data(state_data_dir, nbits=128)
        scaled_split = [int(len(self.data) * p) for p in split]
        self.size = scaled_split[1] - scaled_split[0] - self.next
        self.data = self.data[scaled_split[0]:scaled_split[1]]
        self.state_data = self.state_data[scaled_split[0]:scaled_split[1]]

        print(f"Load data from {data_dir}, split {split}, total data {self.size}")
        print(f"Load data from {state_data_dir}, split {split}, total data {self.size}")

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        assert 0 <= item < self.size
        # return torch.tensor(binary(self.state_data[item:item + self.next], nbits=128), dtype=torch.uint8), torch.tensor(
        #     binary(self.data[item + self.next]))
        return torch.tensor(binary(self.state_data[item], nbits=128), dtype=torch.uint8), torch.tensor(
            binary(self.state_data[item + 1], nbits=128), dtype=torch.uint8)
        # return torch.tensor((self.state_data[item])/2**128, dtype=torch.double), torch.tensor(
        #     (self.state_data[item + 1])/2**128, dtype=torch.double), torch.tensor(
        #     binary(self.state_data[item + 1], nbits=128), dtype=torch.uint8)


class LehmerBackwardDataset(torch.utils.data.dataset.Dataset):
    # from state to next output
    def __init__(self, data_dir, state_data_dir, split):
        super().__init__()
        self.next = 0
        self.data = read_data(data_dir, nbits=64)
        self.state_data = read_data(state_data_dir, nbits=128)
        scaled_split = [int(len(self.data) * p) for p in split]
        self.size = scaled_split[1] - scaled_split[0] - 3
        self.data = self.data[scaled_split[0]:scaled_split[1]]
        self.state_data = self.state_data[scaled_split[0]:scaled_split[1]]
        self.index = np.arange(3)
        print(f"Load data from {data_dir}, split {split}, total data {self.size}")
        print(f"Load data from {state_data_dir}, split {split}, total data {self.size}")

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        # we mask the lower 64 bits because it is trivial to predict
        assert 0 <= item < self.size
        return torch.tensor(binary(self.data[item + self.index])), torch.tensor(
            # binary(self.state_data[item + self.next]))
            binary(self.state_data[item + self.next], nbits=128)[64:], dtype=torch.uint8)


class HalfCrackerDataset(torch.utils.data.dataset.Dataset):
    MAX_SEQLEN = 624

    def __init__(self, data_dir, register_data_dir, split):
        super().__init__()
        self.data = np.fromfile(data_dir, dtype=np.uint32)
        self.register_data = np.fromfile(register_data_dir, dtype=np.uint32)
        scaled_split = [int(len(self.data) * p) for p in split]
        self.size = scaled_split[1] - scaled_split[0] - self.MAX_SEQLEN
        self.data = self.data[scaled_split[0]:scaled_split[1]]
        self.register_data = self.register_data[scaled_split[0]:scaled_split[1]]
        print(f"Load data from {data_dir}, split {split}, total data {self.size}")
        print(f"Load data from {register_data_dir}, split {split}, total data {self.size}")

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        assert 0 <= item < self.size
        return torch.tensor(binary(self.data[item:item + self.MAX_SEQLEN])), torch.tensor(
            binary(self.register_data[item + self.MAX_SEQLEN]))
