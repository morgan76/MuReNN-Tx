# MuReNN‑Tx

A modular PyTorch implementation of MuReNN‑Tx with Hydra configs and Lightning training loops.

## Quickstart
```bash
# 1) Create conda environment
conda create -n murenn-tx python=3.10 -y
conda activate murenn-tx

# (alternative: python -m venv .venv && source .venv/bin/activate)

# 2) Install
pip install -U pip
pip install -e .
pre-commit install

# 3) Run a tiny sanity training on ESC‑50 (expects CSVs & wavs; see conf/data/esc50_debug.yaml)
python -m murenn_tx.train +experiment=esc50_debug
```

## Project layout
```
murenn-tx/
├─ pyproject.toml
├─ README.md
├─ Makefile
├─ .pre-commit-config.yaml
├─ src/
│ └─ murenn_tx/
│ ├─ __init__.py
│ ├─ modules/
│ │ ├─ frontend.py
│ │ ├─ tokenizers.py
│ │ ├─ local_transformer.py
│ │ ├─ fusion.py
│ │ └─ model.py
│ └─ lightning/
│ ├─ datamodule.py
│ └─ model.py
│ └─ train.py
├─ conf/
│ ├─ config.yaml # top-level defaults + global knobs
│ ├─ trainer.yaml # Lightning Trainer config
│ ├─ model/
│ │ └─ base.yaml
│ ├─ data/
│ │ ├─ esc50_debug.yaml
│ │ └─ generic_folder.yaml
│ └─ experiment/
│ ├─ esc50_debug.yaml
│ └─ toy_noise.yaml
└─ tests/
├─ test_smoke.py
└─ test_shapes.py
```
