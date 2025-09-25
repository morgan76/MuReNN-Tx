import argparse, json, os, sys
from pathlib import Path
from datetime import datetime
from copy import deepcopy
import yaml
import importlib

from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger


from murenn_tx.lightning.lit_classifier import LitClassifier
from murenn_tx.data.registry import build_datamodule

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../MuReNN-Tx
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from conf.config import MuReNNTxConfig


def _deep_merge(a: dict, b: dict) -> dict:
    out = deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def load_conf(exp: str, run: str) -> dict:
    with open(f"conf/{exp}/base.yaml") as f:
        base = yaml.safe_load(f) or {}
    with open(f"conf/{exp}/{run}.yaml") as f:
        over = yaml.safe_load(f) or {}
    return _deep_merge(base, over)

def apply_overrides(cfg_dict: dict, overrides: list[str]) -> dict:
    for kv in overrides:
        key, val = kv.split("=", 1)
        node = cfg_dict
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        sval = val.strip(); low = sval.lower()
        if low in {"true","false"}: node[parts[-1]] = (low == "true")
        else:
            try: node[parts[-1]] = int(sval)
            except ValueError:
                try: node[parts[-1]] = float(sval)
                except ValueError:
                    node[parts[-1]] = sval
    return cfg_dict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="dataset key, e.g. esc50, tinysol")
    ap.add_argument("--run", required=True, help="run yaml in conf/<exp>/, e.g. tx or cnn")
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--devices", default=None)
    ap.add_argument("--precision", default=None)
    ap.add_argument("--accumulate", type=int, default=None)
    ap.add_argument("--grad_clip", type=float, default=None)
    ap.add_argument("--ckpt", type=str, default=None)
    args = ap.parse_args()

    importlib.import_module(f"murenn_tx.data.{args.exp}")
    cfg_dict = apply_overrides(load_conf(args.exp, args.run), args.override)
    cfg = MuReNNTxConfig(**cfg_dict)

    seed_everything(getattr(cfg, "seed", 1337), workers=True)

    # build model via registry inside LitClassifier
    model = LitClassifier(cfg, arch=cfg.arch, lr=cfg.lr, wd=cfg.wd)
    
    # build the dataset-specific DataModule via the **datamodule registry**
    datamodule = build_datamodule(args.exp, cfg)

    run_tag = f"{args.exp}-{args.run}-{cfg.arch}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = TensorBoardLogger(save_dir="runs", name=run_tag)
    os.makedirs(logger.log_dir, exist_ok=True)
    with open(os.path.join(logger.log_dir, "resolved_config.json"), "w") as f:
        json.dump(cfg_dict, f, indent=2)

    tkw = dict(logger=logger, max_epochs=cfg.epochs, log_every_n_steps=10, accelerator="auto")
    if args.devices is not None:  tkw["devices"] = args.devices
    if args.precision is not None: tkw["precision"] = args.precision
    if args.accumulate is not None: tkw["accumulate_grad_batches"] = args.accumulate
    if args.grad_clip is not None:  tkw["gradient_clip_val"] = args.grad_clip
    trainer = Trainer(**tkw)

    trainer.fit(model, datamodule=datamodule, ckpt_path=args.ckpt)

if __name__ == "__main__":
    main()