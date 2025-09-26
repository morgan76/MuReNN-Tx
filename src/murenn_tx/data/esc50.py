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
from collections import Counter


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

        # ----- First pass: build stable class list from CSV -----
        cls_by_target = {}  # {int target -> str category}
        with open(meta_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    t = int(row["target"])
                except Exception:
                    continue
                cat = row.get("category") or row.get("Category") or f"class_{t}"
                # keep first seen category name for that target (stable)
                cls_by_target.setdefault(t, cat)

        # ESC-50 has 50 classes 0..49; fill any missing with a placeholder
        max_t = max(cls_by_target.keys()) if cls_by_target else 49
        n_cls = max(50, max_t + 1)
        classes = [f"class_{i}" for i in range(n_cls)]
        for t, name in cls_by_target.items():
            classes[t] = name
        self.classes = classes  # index == target id

        # ----- Second pass: materialize items for the requested split -----
        self.items = []  # list[(path, target)]
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

    def class_counts(self) -> dict[str, int]:
        """Return {class_name: count} for this split."""
        counts = Counter(lbl for _, lbl in self.items)
        return {self.classes[i]: counts.get(i, 0) for i in range(len(self.classes))}

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
        if T <= self.T:
            x = torch.nn.functional.pad(x, (0, self.T - T))
        else:
            if self.split == "train":
                # inclusive range: [0, T - self.T]
                offset = torch.randint(0, T - self.T + 1, ()).item()
            else:
                # val/test: deterministic center crop
                offset = max(0, (T - self.T) // 2)
            x = x[:, offset: offset + self.T]
        
        return x, torch.tensor(label, dtype=torch.long)

@register_datamodule("esc50")
class ESC50DM(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.data
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

        # ---- pretty print class distributions (like TinySOL) ----
        self._print_split_stats("train", self.ds_train)
        self._print_split_stats("val",   self.ds_val)

    def _print_split_stats(self, split_name: str, ds: ESC50Dataset):
        dist = ds.class_counts()  # {class_name: count}
        total = len(ds)
        present = sum(1 for c in dist.values() if c > 0)
        width = max(10, max(len(k) for k in dist.keys()))
        print(f"\n[{split_name}] total={total}, classes_present={present}/{len(dist)}")
        print(f"{'class':{width}} | count")
        print("-" * (width + 9))
        for name, cnt in sorted(dist.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"{name:{width}} | {cnt}")

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
