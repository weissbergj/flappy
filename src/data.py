# src/data.py
import numpy as np
import torch
from torch.utils.data import IterableDataset

class BinTokenWindowDataset(IterableDataset):
    def __init__(self, bin_path: str, seq_len: int, seed: int = 0, rank: int = 0, offset_tokens: int = 0, max_tokens: int | None = None):
        super().__init__()
        self.bin_path = bin_path
        self.seq_len = seq_len
        self.seed = seed
        self.rank = rank

        self.tokens = np.memmap(self.bin_path, dtype=np.uint16, mode="r")
        T = len(self.tokens)
        self.start = max(0, min(offset_tokens, T))
        self.end = min(T, self.start + max_tokens) if max_tokens is not None else T
        if self.end - self.start < seq_len + 2:
            raise ValueError("slice too small for requested seq_len (need offset_tokens + max_tokens to give at least seq_len + 2)")

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        rng = np.random.default_rng(self.seed + 1000 * self.rank + wid)
        L = self.seq_len

        while True:
            i = int(rng.integers(self.start, self.end - (L + 1)))
            x = torch.from_numpy(np.asarray(self.tokens[i : i + L], dtype=np.int64))
            y = torch.from_numpy(np.asarray(self.tokens[i + 1 : i + L + 1], dtype=np.int64))
            yield x, y
