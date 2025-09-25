from __future__ import annotations

import csv
import os
from typing import Optional, Tuple, List, Dict

import torch
import torchaudio.functional as AF
import soundfile as sf
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

import random
import numpy as np



def _get_case_insensitive(row: dict, *candidates: str):
    """Return the first matching key from `candidates` (case/spacing insensitive)."""
    keys = {k.lower().replace(" ", "").replace("_", ""): k for k in row.keys()}
    for cand in candidates:
        ck = cand.lower().replace(" ", "").replace("_", "")
        if ck in keys:
            return row[keys[ck]]
    raise KeyError(f"None of {candidates} found in CSV header: {list(row.keys())}")



class TinySOLDataset(Dataset):
    """
    TinySOL dataset loader for **instrument classification** using the official metadata.

    Expects:
      root/
        TinySOL_metadata.csv
        Brass/...
        Strings/...
        Woodwinds/...
        (etc., paths in CSV are relative like 'Brass/BTb/BTb-ord-A#1-ff-N.wav')

    CSV columns (per TinySOL docs) include: Path, Fold ID, Family, Instrument abbreviation,
    Instrument name in full, Technique..., Pitch, Dynamics, etc. We only need Path, Fold, Instrument.
    See: https://zenodo.org/records/3632287  (and newer record variants).  # citation
    """

    def __init__(self, root: str, split: str, fold: int, sr: int = 16000, seconds: float = 3.0):
        assert split in {"train", "val", "test"}
        self.root = root
        print(self.root)
        self.split = split
        self.sr = sr
        self.T = int(seconds * sr)

        meta_path = os.path.join(root, "TinySOL_metadata.csv")
        if not os.path.isfile(meta_path):
            # Some releases may name it slightly differently; try a couple fallbacks.
            for alt in ["TinySOL_metadata_v2.csv", "metadata.csv", "tinysol_metadata.csv"]:
                cand = os.path.join(root, alt)
                if os.path.isfile(cand):
                    meta_path = cand
                    break
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"Could not find TinySOL metadata CSV under {root}")

        # First pass: collect instruments and build a stable class map
        items: List[Tuple[str, str, int]] = []  # (path, instrument_fullname, fold_id)
        instruments_set = set()

        with open(meta_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    relpath = _get_case_insensitive(row, "Path", "File", "WAV path", "wav")
                    fold_id = int(_get_case_insensitive(row, "Fold", "fold", "fold id"))
                    
                except KeyError:
                    # Some older mirrors split path into components; reconstruct if needed
                    relpath = row.get("path") or row.get("Path") or row.get("filepath")
                    if relpath is None:
                        raise
                    fold_id = int(row.get("Fold") or row.get("Fold ID") or row.get("fold id") or 0) 
                    

                # Prefer the full instrument name; fall back to family/abbr if needed
                inst_full = (
                    row.get("Instrument name in full")
                    or row.get("instrument (full)")
                    or row.get("Instrument")
                    or row.get("Instrument full")
                    or row.get("instrument_full")
                )
                if inst_full is None:
                    # Last resort: compose from family + abbr
                    family = row.get("Family") or row.get("family")
                    abbr = row.get("Instrument abbreviation") or row.get("Instrument (abbr.)") or row.get("abbr")
                    inst_full = f"{family or ''} {abbr or ''}".strip() or "Unknown"
                    

                instruments_set.add(inst_full)
                items.append((relpath, inst_full, fold_id))
                

        # Stable label ordering (alphabetical ensures reproducibility)
        instruments: List[str] = sorted(instruments_set)
        self.label_map: Dict[str, int] = {name: i for i, name in enumerate(instruments)}
        print(self.label_map)
        # Second pass: apply split filter and materialize items
        
        self.items: List[Tuple[str, int]] = []
        for relpath, inst_full, fnum in items:
            if self.split == "train" and fnum == fold:
                continue
            if self.split in {"val", "test"} and fnum != fold:
                continue
            fullpath = os.path.join(self.root, relpath)
            self.items.append((fullpath, self.label_map[inst_full]))


    def __len__(self):
        return len(self.items)
    

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        
        wav, sr = sf.read(path, dtype="float32", always_2d=True)  # (T, C)
        x = torch.from_numpy(wav).transpose(0, 1).contiguous()    # (C, T)
        
        if x.size(0) > 1:
            x = x.mean(dim=0, keepdim=True)
        else:
            x = x[:1, :]
        if sr != self.sr:
            x = AF.resample(x, orig_freq=sr, new_freq=self.sr)
        T = x.size(1)
        if T < self.T:
            x = torch.nn.functional.pad(x, (0, self.T - T))
        else:
            
            offset = random.choice(np.arange(T-self.T))
            x = x[:, offset : offset + self.T]
        
        return x, torch.tensor(label, dtype=torch.long)
    


class TinySOLDM(LightningDataModule):
    def __init__(
        self,
        root: str,
        fold: int = 0,
        batch_size: int = 32,
        num_workers: int = 4,
        sr: int = 16000,
        seconds: float = 3.0,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):
        hp = self.hparams
        self.ds_train = TinySOLDataset(
            hp.root, split="train", fold=hp.fold, sr=hp.sr, seconds=hp.seconds
        )
        self.ds_val = TinySOLDataset(
            hp.root, split="val", fold=hp.fold, sr=hp.sr, seconds=hp.seconds
        )

    def train_dataloader(self):
        hp = self.hparams
        return DataLoader(
            self.ds_train,
            batch_size=hp.batch_size,
            shuffle=True,
            num_workers=hp.num_workers,
            pin_memory=True,
            persistent_workers=hp.num_workers > 0,
        )

    def val_dataloader(self):
        hp = self.hparams
        return DataLoader(
            self.ds_val,
            batch_size=hp.batch_size,
            shuffle=False,
            num_workers=hp.num_workers,
            pin_memory=True,
            persistent_workers=hp.num_workers > 0,
        )