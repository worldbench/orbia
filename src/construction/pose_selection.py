"""Signed OpenCV camera-component gates for registered endpoint selection."""
from typing import Mapping
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation

TEMPLATE_POSE_GATE_OPTION_KEYS = frozenset(
    {
        "minimum_translation_m",
        "maximum_translation_m",
        "minimum_rotation_deg",
        "maximum_rotation_deg",
        "minimum_forward_component_m",
        "minimum_lateral_component_m",
        "maximum_lateral_component_m",
        "minimum_absolute_forward_component_m",
        "maximum_absolute_forward_component_m",
        "minimum_absolute_right_component_m",
        "maximum_absolute_right_component_m",
        "maximum_absolute_vertical_component_m",
        "minimum_forward_ratio",
        "maximum_absolute_forward_ratio",
        "minimum_absolute_right_ratio",
        "maximum_absolute_right_ratio",
        "maximum_absolute_vertical_ratio",
        "minimum_direction_angle_deg",
        "maximum_direction_angle_deg",
    }
)

def _registered_pose_baseline(
    first_record: object,
    second_record: object,
) -> tuple[float, float]:
    """Compute the fixed-pair baseline without decoding unrelated frames."""

    first = np.asarray(getattr(first_record, "T_world_camera"), dtype=np.float64)
    second = np.asarray(getattr(second_record, "T_world_camera"), dtype=np.float64)
    if first.shape != (4, 4) or second.shape != (4, 4):
        raise ValueError("registered fixed pair must contain 4x4 poses")
    translation = float(np.linalg.norm(first[:3, 3] - second[:3, 3]))
    relative = first[:3, :3].T @ second[:3, :3]
    rotation = float(np.rad2deg(Rotation.from_matrix(relative).magnitude()))
    return translation, rotation

def _iteration_pose_gate(
    q0_record: object,
    qe_record: object,
    *,
    template_name: str,
    gate: Mapping[str, float],
) -> dict[str, float | bool]:
    """Evaluate one cheap registered-pose pair before any RGB-D decode."""

    q0 = np.asarray(getattr(q0_record, "T_world_camera"), dtype=np.float64)
    qe = np.asarray(getattr(qe_record, "T_world_camera"), dtype=np.float64)
    if q0.shape != (4, 4) or qe.shape != (4, 4):
        raise ValueError("registered pair poses must be finite 4x4 matrices")
    translation, rotation = _registered_pose_baseline(q0_record, qe_record)
    delta = qe[:3, 3] - q0[:3, 3]
    # T_world_camera follows OpenCV camera-to-world axes: +x right, +y down,
    # +z forward.  The three signed components stay separate so that vertical
    # displacement is never counted as lateral motion.
    right_axis = q0[:3, 0]
    down_axis = q0[:3, 1]
    forward_axis = q0[:3, 2]
    right_component = float(np.dot(delta, right_axis))
    vertical_component = float(np.dot(delta, down_axis))
    forward_component = float(np.dot(delta, forward_axis))
    horizontal_translation = float(
        np.hypot(forward_component, right_component)
    )
    lateral_component = right_component
    direction_angle = float(
        np.degrees(
            np.arctan2(abs(right_component), forward_component)
        )
        if horizontal_translation > 1e-12
        else 90.0
    )
    denominator = max(translation, 1e-12)
    forward_ratio = float(forward_component / denominator)
    absolute_forward_ratio = float(abs(forward_component) / denominator)
    absolute_right_ratio = float(abs(right_component) / denominator)
    absolute_vertical_ratio = float(abs(vertical_component) / denominator)
    translation_valid = bool(
        float(gate["minimum_translation_m"])
        <= translation
        <= float(gate["maximum_translation_m"])
    )
    rotation_valid = bool(
        float(gate["minimum_rotation_deg"])
        <= rotation
        <= float(gate["maximum_rotation_deg"])
    )
    forward_valid = bool(
        "minimum_forward_component_m" not in gate
        or forward_component >= float(gate["minimum_forward_component_m"])
    )
    lateral_valid = bool(
        "minimum_lateral_component_m" not in gate
        or abs(right_component) >= float(gate["minimum_lateral_component_m"])
    )
    component_checks = {
        "minimum_absolute_forward_component_valid": (
            "minimum_absolute_forward_component_m" not in gate
            or abs(forward_component)
            >= float(gate["minimum_absolute_forward_component_m"])
        ),
        "maximum_absolute_forward_component_valid": (
            "maximum_absolute_forward_component_m" not in gate
            or abs(forward_component)
            <= float(gate["maximum_absolute_forward_component_m"])
        ),
        "minimum_absolute_right_component_valid": (
            "minimum_absolute_right_component_m" not in gate
            or abs(right_component)
            >= float(gate["minimum_absolute_right_component_m"])
        ),
        "maximum_absolute_right_component_valid": (
            (
                "maximum_absolute_right_component_m" not in gate
                or abs(right_component)
                <= float(gate["maximum_absolute_right_component_m"])
            )
            and (
                "maximum_lateral_component_m" not in gate
                or abs(right_component)
                <= float(gate["maximum_lateral_component_m"])
            )
        ),
        "maximum_absolute_vertical_component_valid": (
            "maximum_absolute_vertical_component_m" not in gate
            or abs(vertical_component)
            <= float(gate["maximum_absolute_vertical_component_m"])
        ),
        "minimum_forward_ratio_valid": (
            "minimum_forward_ratio" not in gate
            or forward_ratio >= float(gate["minimum_forward_ratio"])
        ),
        "maximum_absolute_forward_ratio_valid": (
            "maximum_absolute_forward_ratio" not in gate
            or absolute_forward_ratio
            <= float(gate["maximum_absolute_forward_ratio"])
        ),
        "minimum_absolute_right_ratio_valid": (
            "minimum_absolute_right_ratio" not in gate
            or absolute_right_ratio
            >= float(gate["minimum_absolute_right_ratio"])
        ),
        "maximum_absolute_right_ratio_valid": (
            "maximum_absolute_right_ratio" not in gate
            or absolute_right_ratio
            <= float(gate["maximum_absolute_right_ratio"])
        ),
        "maximum_absolute_vertical_ratio_valid": (
            "maximum_absolute_vertical_ratio" not in gate
            or absolute_vertical_ratio
            <= float(gate["maximum_absolute_vertical_ratio"])
        ),
    }
    minimum_direction = float(gate.get("minimum_direction_angle_deg", 0.0))
    maximum_direction = float(gate.get("maximum_direction_angle_deg", 180.0))
    direction_valid = bool(
        minimum_direction <= direction_angle <= maximum_direction
    )
    passed = bool(
        translation_valid
        and rotation_valid
        and forward_valid
        and lateral_valid
        and all(component_checks.values())
        and direction_valid
    )
    return {
        "direction_angle_deg": direction_angle,
        "direction_angle_valid": direction_valid,
        "forward_component_m": forward_component,
        "lateral_component_m": lateral_component,
        "right_component_m": right_component,
        "vertical_component_m": vertical_component,
        "down_component_m": vertical_component,
        "forward_ratio": forward_ratio,
        "absolute_forward_ratio": absolute_forward_ratio,
        "absolute_right_ratio": absolute_right_ratio,
        "absolute_vertical_ratio": absolute_vertical_ratio,
        "horizontal_translation_m": horizontal_translation,
        "passed": passed,
        "rotation_deg": rotation,
        "rotation_valid": rotation_valid,
        "template": template_name,
        "translation_m": translation,
        "translation_valid": translation_valid,
        "forward_component_valid": forward_valid,
        "lateral_component_valid": lateral_valid,
        "maximum_direction_angle_deg": maximum_direction,
        "minimum_direction_angle_deg": minimum_direction,
        **component_checks,
    }

def evaluate_pose_pair(q0, qe, template, gate):
    required = {"minimum_translation_m", "maximum_translation_m", "minimum_rotation_deg", "maximum_rotation_deg"}
    if set(gate)-TEMPLATE_POSE_GATE_OPTION_KEYS or required-set(gate):
        raise ValueError("pose gate has missing or unsupported fields")
    if any(not np.isfinite(v) or v < 0 for v in gate.values()):
        raise ValueError("pose gate values must be finite and nonnegative")
    return _iteration_pose_gate(SimpleNamespace(T_world_camera=q0),
        SimpleNamespace(T_world_camera=qe), template_name=template, gate=gate)
