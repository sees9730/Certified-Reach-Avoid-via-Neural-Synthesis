from __future__ import annotations

import ast
from collections.abc import Mapping

import numpy as np


UNSAFE_REGION_NAMES = (
    "unsafe_down1",
    "unsafe_down2",
    "unsafe_lb",
    "unsafe_rb",
    "unsafe_tb",
    "unsafe_bb",
)


def _eval_region_expr(expr: object) -> float:
    if isinstance(expr, (int, float)):
        return float(expr)
    if not isinstance(expr, str):
        raise TypeError(f"Unsupported region entry type: {type(expr)!r}")

    normalized = expr.strip().replace("\\pi", "pi")
    tree = ast.parse(normalized, mode="eval")

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id == "pi":
            return float(np.pi)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = _eval(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            return left / right
        raise ValueError(f"Unsupported region expression: {expr!r}")

    return _eval(tree)


def _region_value_to_array(value: object) -> np.ndarray:
    nested = [
        [_eval_region_expr(entry) for entry in row]
        for row in value
    ]
    return np.asarray(nested, dtype=np.float32)


def load_region_arrays(example_config: Mapping) -> dict[str, np.ndarray]:
    regions_cfg = example_config["regions"]
    regions = {
        name: _region_value_to_array(value)
        for name, value in regions_cfg.items()
    }
    regions["unsafe_ranges"] = np.stack(
        [regions[name] for name in UNSAFE_REGION_NAMES],
        axis=0,
    )
    return regions


def region_array_to_box(region: np.ndarray) -> dict[str, float]:
    arr = np.asarray(region, dtype=float)
    if arr.shape != (2, 2):
        raise ValueError(f"Expected region shape (2, 2), got {arr.shape}")
    return {
        "x1_min": float(arr[0, 0]),
        "x1_max": float(arr[0, 1]),
        "x2_min": float(arr[1, 0]),
        "x2_max": float(arr[1, 1]),
    }


def load_region_boxes(example_config: Mapping) -> dict[str, object]:
    regions = load_region_arrays(example_config)
    return {
        "init": region_array_to_box(regions["init_range"]),
        "goal": region_array_to_box(regions["goal_range"]),
        "full": region_array_to_box(regions["full_range"]),
        "unsafe": [
            region_array_to_box(regions[name])
            for name in UNSAFE_REGION_NAMES
        ],
    }


def load_controller_hidden_dim(example_config: Mapping, default: int = 32) -> int:
    value = example_config.get("controller_hidden_dim", default)
    return int(value)


def load_dynamics_params(example_config: Mapping) -> dict[str, float]:
    """
    Single source of truth for the inverted-pendulum physical constants
    (g, L, m, b, M_torque, sigma), read from config.json's "dynamics" section.
    """
    dynamics_cfg = example_config["dynamics"]
    return {key: float(value) for key, value in dynamics_cfg.items()}
