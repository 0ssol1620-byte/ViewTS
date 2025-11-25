# utils/sandbox.py
from __future__ import annotations
from typing import Any, Dict

SAFE_BUILTINS: Dict[str, Any] = {
    "bool": bool, "int": int, "float": float, "str": str,
    "len": len, "abs": abs, "max": max, "min": min, "sum": sum,
    "sorted": sorted, "range": range,
}

def safe_eval(expr: str, env: Dict[str, Any] | None = None) -> Any:
    """아주 제한된 eval().  SAFE_BUILTINS + env 만 접근 가능."""
    g: Dict[str, Any] = {"__builtins__": SAFE_BUILTINS}
    if env:
        g.update(env)
    return eval(expr, g, {})
