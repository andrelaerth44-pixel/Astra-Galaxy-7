from pathlib import Path
import torch
from torch.utils.data import Dataset


class TokenDataset(Dataset):
    """Loads a flat token stream stored as whitespace-separated integers."""

    def __init__(self, path, seq_len):
        self.tokens = list(map(int, Path(path).read_text().split()))
        self.seq_len = seq_len
        self.length = max(0, (len(self.tokens) - 1) // seq_len)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.tokens[start : start + self.seq_len + 1]
        if len(chunk) != self.seq_len + 1:
            raise IndexError(idx)

        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y
