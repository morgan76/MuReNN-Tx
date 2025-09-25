from typing import Callable, Dict

_MODEL_REGISTRY: Dict[str, Callable] = {}

def register_model(name: str):
    def deco(fn: Callable):
        if name in _MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' already registered")
        _MODEL_REGISTRY[name] = fn
        return fn
    return deco

def build_model(name: str, **kwargs):
    if name not in _MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {list(_MODEL_REGISTRY)}")
    return _MODEL_REGISTRY[name](**kwargs)