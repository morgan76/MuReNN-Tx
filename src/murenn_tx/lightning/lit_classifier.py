import torch
from torch import nn
from lightning import LightningModule
from torchmetrics.classification import MulticlassAccuracy
from murenn_tx.models.registry import build_model

class LitClassifier(LightningModule):
    def __init__(self, cfg, arch: str = "tx", lr: float = 3e-4, wd: float = 0.05, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        self.model = build_model(arch, cfg=cfg)  # <-- key line
        self.acc = MulticlassAccuracy(num_classes=cfg.n_classes)

    def forward(self, x): return self.model(x)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        acc  = self.acc(logits, y)
        self.log_dict({"train/loss": loss, "train/acc": acc}, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        acc  = self.acc(logits, y)
        self.log_dict({"val/loss": loss, "val/acc": acc}, on_step=False, on_epoch=True, prog_bar=True)