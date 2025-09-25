from typing import Callable, Dict, Any
from lightning import LightningDataModule

_DATAMODULES: Dict[str, Callable[[Any], LightningDataModule]] = {}

def register_datamodule(name: str):
    def deco(ctor: Callable[[Any], LightningDataModule]):
        if name in _DATAMODULES:
            raise ValueError(f"Datamodule '{name}' already registered")
        _DATAMODULES[name] = ctor
        return ctor
    return deco

def build_datamodule(name: str, cfg) -> LightningDataModule:
    if name not in _DATAMODULES:
        raise KeyError(f"Unknown datamodule '{name}'. Available: {list(_DATAMODULES)}")
    return _DATAMODULES[name](cfg)