from __future__ import annotations

import torch
from lightning.pytorch import LightningModule
from torchmetrics.classification import MulticlassAccuracy

from ..modules.model import MuReNNTx, MuReNNTxConfig


class LitMuReNNTx(LightningModule):
    def __init__(self, cfg: MuReNNTxConfig, lr: float = 3e-4, wd: float = 0.05):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])  # keep cfg separately
        self.model = MuReNNTx(cfg)
        self.acc = MulticlassAccuracy(num_classes=cfg.n_classes)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        return opt

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        #self.log("train/loss", loss, prog_bar=True)
        acc = self.acc(logits, y)
        self.log_dict({"train/loss": loss, "train/acc": acc}, prog_bar=True)
        #print(self.global_step)
        #if self.global_step % 10 == 0:
        #    stats = self.model.scale_pe_stats()
        #    print("scale_pe/mean_offdiag", stats["mean_offdiag"])
        #    print("scale_pe/min_offdiag",  stats["min_offdiag"])
        #    print("scale_pe/max_offdiag",  stats["max_offdiag"])
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        acc = self.acc(logits, y)
        self.log_dict({"val/loss": loss, "val/acc": acc}, prog_bar=True)
