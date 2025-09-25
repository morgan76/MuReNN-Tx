from __future__ import annotations

import csv
import os
from typing import Optional

import torch
import torchaudio
import torchaudio.functional as AF
import soundfile as sf
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from murenn_tx.data.registry import register_datamodule


class ESC50Dataset(Dataset):
    """ESC-50 dataset loader using the official folder layout.

    Expects:
    root/
    audio/*.wav
    meta/esc50.csv (columns: filename, fold, target, category, esc10, src_file, take)
    """

    def __init__(self, root: str, split: str, fold: int, sr: int = 16000, seconds: float = 5.0):
        assert split in {"train", "val", "test"}
        self.root = root
        self.split = split
        self.sr = sr
        self.T = int(seconds * sr)
        meta_path = os.path.join(root, "meta", "esc50.csv")
        self.items = []
        with open(meta_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                fnum = int(row["fold"]) if row["fold"] else None
                if split == "train" and fnum == fold:
                    continue
                if split in {"val", "test"} and fnum != fold:
                    continue
                path = os.path.join(root, "audio", row["filename"])
                label = int(row["target"])  # 0..49
                self.items.append((path, label))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        #x, sr = torchaudio.load(path)  # (C,T)
        wav, sr = sf.read(path, dtype="float32", always_2d=True)
        x = torch.from_numpy(wav).transpose(0, 1).contiguous()
        #x, sr = torchaudio.load_with_torchcodec(path)
        if x.size(0) > 1:
            x = x.mean(dim=0, keepdim=True)
        else:
            x = x[:1, :]
        if sr != self.sr:
            x = AF.resample(x, orig_freq=sr, new_freq=self.sr)
        T = x.size(1)
        if T < self.T:
            pad = self.T - T
            x = torch.nn.functional.pad(x, (0, pad))
        else:
            x = x[:, : self.T]
        return x, torch.tensor(label, dtype=torch.long)

@register_datamodule("esc50")
class ESC50DM(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.data
        # stash simple scalars (avoid passing cfg around downstream)
        self.root = d.root
        self.fold = getattr(d, "fold", 1)
        self.batch_size = d.batch_size
        self.num_workers = d.num_workers
        self.sr = d.sample_rate
        self.seconds = d.segment_seconds

    def setup(self, stage: Optional[str] = None):
        self.ds_train = ESC50Dataset(
            root=self.root, split="train", fold=self.fold, sr=self.sr, seconds=self.seconds
        )
        self.ds_val = ESC50Dataset(
            root=self.root, split="val", fold=self.fold, sr=self.sr, seconds=self.seconds
        )

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
