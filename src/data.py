# src/data.py
import numpy as np
import torch
from torch.utils.data import IterableDataset

class BinTokenWindowDataset(IterableDataset):
    def __init__(self, bin_path: str, seq_len: int, seed: int = 0, max_tokens: int | None = None):
        super().__init__()
        self.bin_path = bin_path
        self.seq_len = seq_len
        self.seed = seed

        self.tokens = np.memmap(self.bin_path, dtype=np.uint16, mode="r")
        T = len(self.tokens)
        self.T = min(T, max_tokens) if max_tokens is not None else T
        if self.T < seq_len + 2:
            raise ValueError("bin too small for requested seq_len")

    def __iter__(self):
        rng = np.random.default_rng(self.seed + (torch.utils.data.get_worker_info().id if torch.utils.data.get_worker_info() else 0))
        T = self.T
        L = self.seq_len

        while True:
            i = int(rng.integers(0, T - (L + 1)))
            x = torch.from_numpy(np.asarray(self.tokens[i : i + L], dtype=np.int64))
            y = torch.from_numpy(np.asarray(self.tokens[i + 1 : i + L + 1], dtype=np.int64))
            yield x, y
