"""Simple, fixed input limits for the TIPS MT-GBM web calculator.

These rules constrain WEB INPUTS only; they do not alter the fitted model.
"""
from __future__ import annotations
import math

MINIMUMS = {
    "Age": 18.0,
    "PLT": 10.0,
    "Hb": 40.0,
    "Alb": 15.0,
    "TBIL": 0.01,
    "Scr": 0.01,
    "INR": 0.01,
    "PT_prolongation": -5.0,
}
MAXIMUMS = {"Age": 75.0}


def check_inputs(values: dict[str, float]) -> list[str]:
    """Return validation errors; an empty list means the calculator may predict."""
    errors = []
    for name, minimum in MINIMUMS.items():
        try:
            value = float(values[name])
        except (ValueError, TypeError, KeyError):
            errors.append(f"{name} 必须填写有效数值")
            continue
        if not math.isfinite(value):
            errors.append(f"{name} 必须填写有限数值")
        elif value < minimum:
            errors.append(f"{name} 不能低于 {minimum:g}")
        elif name in MAXIMUMS and value > MAXIMUMS[name]:
            errors.append(f"{name} 不能高于 {MAXIMUMS[name]:g}")
    return errors
