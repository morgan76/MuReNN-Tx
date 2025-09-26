# MuReNN‑Tx

PyTorch implementation of MuReNN‑Tx (transformers).

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
MuReNN-Tx/
├─ pyproject.toml # package metadata (murenn_tx), deps
├─ README.md
├─ .gitignore
├─ .gitattributes
├─ conf/
│ ├─ config.yaml # Hydra defaults / global knobs
│ ├─ trainer.yaml # Lightning Trainer params
│ ├─ data/
│ │ ├─ esc50_debug.yaml
│ │ └─ generic_folder.yaml
│ ├─ model/
│ │ └─ base.yaml # model hyperparams (frontend, tokenizer, tx)
│ └─ experiment/
│ ├─ esc50_debug.yaml # end-to-end experiment configs
│ └─ toy_noise.yaml
├─ src/
│ └─ murenn_tx/
│ ├─ init.py
│ ├─ train.py # entry point: python -m murenn_tx.train +experiment=...
│ ├─ modules/
│ │ ├─ frontend.py # DTCWT + learnable psi bank front-end
│ │ ├─ tokenizers.py
│ │ ├─ local_transformer.py
│ │ ├─ fusion.py
│ │ └─ model.py # MuReNN-Tx model assembly
│ └─ lightning/
│ ├─ datamodule.py # data loading (e.g., ESC-50)
│ └─ model.py # LightningModule wrapper
└─ tests/
├─ test_smoke.py
└─ test_shapes.py
```
