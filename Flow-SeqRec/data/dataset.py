import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class SequentialRecDataset(Dataset):
    """Dataset for sequential recommendation.

    Each sample consists of:
        - input_seq: historical item interaction sequence (padded)
        - target: the next item to predict
        - seq_len: actual length of the input sequence
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        max_seq_len: int = 50,
    ):
        assert split in ("train", "valid", "test")
        self.max_seq_len = max_seq_len
        self.split = split

        seq_path = os.path.join(data_dir, f"{split}_sequences.pkl")
        meta_path = os.path.join(data_dir, "metadata.pkl")

        with open(seq_path, "rb") as f:
            self.sequences: List[List[int]] = pickle.load(f)

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        self.num_items: int = meta["num_items"]
        self.num_users: int = meta["num_users"]

        self.sparse_features: Optional[np.ndarray] = None
        sparse_path = os.path.join(data_dir, "item_sparse_features.npy")
        if os.path.exists(sparse_path):
            self.sparse_features = np.load(sparse_path, allow_pickle=True)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        target = seq[-1]
        input_items = seq[:-1]

        seq_len = min(len(input_items), self.max_seq_len)
        if len(input_items) > self.max_seq_len:
            input_items = input_items[-self.max_seq_len:]

        padded = [0] * (self.max_seq_len - seq_len) + input_items

        sample = {
            "input_seq": torch.tensor(padded, dtype=torch.long),
            "target": torch.tensor(target, dtype=torch.long),
            "seq_len": torch.tensor(seq_len, dtype=torch.long),
        }

        if self.sparse_features is not None:
            sample["sparse_features"] = torch.tensor(
                self.sparse_features[target], dtype=torch.float32
            )

        return sample


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate a list of samples into a batched dictionary."""
    keys = batch[0].keys()
    collated = {}
    for k in keys:
        collated[k] = torch.stack([sample[k] for sample in batch], dim=0)
    return collated
