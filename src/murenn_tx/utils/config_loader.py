import yaml
from copy import deepcopy

def _deep_merge(a, b):
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