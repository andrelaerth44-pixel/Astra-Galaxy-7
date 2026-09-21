from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TokenDataset(Dataset):
    """Memory-mapped next-token prediction dataset backed by uint32 tokens."""

    def __init__(self, path, seq_len):
        self.path = Path(path)
        self.seq_len = seq_len
        self.tokens = np.memmap(self.path, dtype=np.uint32, mode="r")
        self.length = max(0, (len(self.tokens) - 1) // self.seq_len)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise IndexError(idx)

        start = idx * self.seq_len
        chunk = self.tokens[start : start + self.seq_len + 1]

        x = torch.from_numpy(np.asarray(chunk[:-1], dtype=np.int64).copy())
        y = torch.from_numpy(np.asarray(chunk[1:], dtype=np.int64).copy())
        return x, y
