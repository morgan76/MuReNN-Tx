from __future__ import annotations

import hydra
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig, OmegaConf

from .lightning.datamodule import AudioDM
from .lightning.esc50_datamodule import ESC50DM
from .lightning.tinysol_datamodule import TinySOLDM
from .lightning.model import LitMuReNNTx
from .modules.model import MuReNNTxConfig


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    lr = OmegaConf.select(cfg, "optim.lr", default=3e-4)
    wd = OmegaConf.select(cfg, "optim.wd", default=5.0e-2)

    # Model / LitModule
    model_cfg = MuReNNTxConfig(**cfg.model)
    lit = LitMuReNNTx(model_cfg, lr=lr, wd=wd)

    # Data
    if cfg.data.get("kind", "csv") == "esc50":
        dm = ESC50DM(
            root=cfg.data.root,
            fold=cfg.data.fold,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            sr=cfg.data.sample_rate,
            seconds=cfg.data.seconds,
        )
    elif cfg.data.get("kind", "csv") == "tinysol":
        dm = TinySOLDM(
            root=cfg.data.root,
            fold=cfg.data.fold,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            sr=cfg.data.sample_rate,
            seconds=cfg.data.seconds,
        )
    else:
        dm = AudioDM(
            train_csv=cfg.data.train_csv,
            val_csv=cfg.data.val_csv,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            sr=cfg.data.sample_rate,
            seconds=cfg.data.seconds,
        )

    # Trainer
    logger = CSVLogger(save_dir="logs", name=cfg.experiment.name)
    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
        ModelCheckpoint(monitor="val/acc", mode="max", save_last=True, save_top_k=1),
    ]
    trainer = Trainer(**cfg.trainer, callbacks=callbacks, logger=logger)
    trainer.fit(lit, datamodule=dm)


if __name__ == "__main__":
    main()
