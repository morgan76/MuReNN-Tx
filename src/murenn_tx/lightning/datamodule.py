from __future__ import annotations

from typing import Optional

import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset


class FolderAudioDataset(Dataset):
    """A minimalist dataset expecting a CSV with columns: path,label
    and relative/absolute paths to audio files.
    """

    def __init__(self, csv_path: str, sr: int = 16000, seconds: float = 1.0):
        import csv

        self.items = []
        with open(csv_path) as f:
            for row in csv.reader(f):
                if len(row) < 2:
                    continue
                self.items.append((row[0], int(row[1])))
        self.sr = sr
        self.T = int(seconds * sr)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        x, sr = sf.read(path, dtype="float32", always_2d=False)
        if sr != self.sr:
            import resampy

            x = resampy.resample(x, sr, self.sr)
        if x.ndim > 1:
            x = x.mean(axis=1)
        x = torch.from_numpy(x)
        if x.numel() < self.T:
            x = torch.nn.functional.pad(x, (0, self.T - x.numel()))
        else:
            x = x[: self.T]
        return x.unsqueeze(0), torch.tensor(label, dtype=torch.long)


class SimpleDataModule(torch.utils.data.Dataset):
    pass


class AudioDataModule(torch.utils.data.Dataset):
    pass


class LightningAudioDataModule(torch.utils.data.Dataset):
    pass


from lightning.pytorch import LightningDataModule


class AudioDM(LightningDataModule):
    def __init__(
        self,
        train_csv: str,
        val_csv: str,
        batch_size: int = 8,
        num_workers: int = 4,
        sr: int = 16000,
        seconds: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):
        hp = self.hparams
        self.ds_train = FolderAudioDataset(hp.train_csv, sr=hp.sr, seconds=hp.seconds)
        self.ds_val = FolderAudioDataset(hp.val_csv, sr=hp.sr, seconds=hp.seconds)

    def train_dataloader(self):
        hp = self.hparams
        return DataLoader(
            self.ds_train,
            batch_size=hp.batch_size,
            shuffle=True,
            num_workers=hp.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        hp = self.hparams
        return DataLoader(
            self.ds_val,
            batch_size=hp.batch_size,
            shuffle=False,
            num_workers=hp.num_workers,
            pin_memory=True,
        )
