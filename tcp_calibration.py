#!/usr/bin/env python3
"""Solve a flange-to-TCP translation from fixed-point pose samples.

The samples must be Nova ``GetPose(user=0,tool=0)`` poses recorded while the
same physical fingertip point touches a fixed reference point in several
different flange orientations. The solver is offline and never connects to
Nova or writes a Tool definition.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


Pose = tuple[float, float, float, float, float, float]
Vector3 = tuple[float, float, float]
Matrix3 = tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class TcpCalibrationResult:
    tcp_flange_mm: Vector3
    fixed_point_user_mm: Vector3
    residuals_mm: tuple[float, ...]
    rms_residual_mm: float
    max_residual_mm: float
    sample_count: int


def _matmul(left: Matrix3, right: Matrix3) -> Matrix3:
    return tuple(
        tuple(sum(left[row][k] * right[k][column] for k in range(3)) for column in range(3))
        for row in range(3)
    )


def rotation_matrix_xyz(rx_deg: float, ry_deg: float, rz_deg: float) -> Matrix3:
    """Return Dobot's fixed-axis X->Y->Z Euler rotation.

    The TCP reference states that rx, ry, rz rotate about fixed user-frame
    axes in X->Y->Z order, which is Rz * Ry * Rx in matrix form.
    """
    rx, ry, rz = (math.radians(value) for value in (rx_deg, ry_deg, rz_deg))
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rx_matrix = ((1.0, 0.0, 0.0), (0.0, cx, -sx), (0.0, sx, cx))
    ry_matrix = ((cy, 0.0, sy), (0.0, 1.0, 0.0), (-sy, 0.0, cy))
    rz_matrix = ((cz, -sz, 0.0), (sz, cz, 0.0), (0.0, 0.0, 1.0))
    return _matmul(_matmul(rz_matrix, ry_matrix), rx_matrix)


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    augmented = [row[:] + [vector[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-10:
            raise ValueError("TCP samples do not provide enough orientation diversity")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor:
                augmented[row] = [
                    current - factor * pivot_value
                    for current, pivot_value in zip(augmented[row], augmented[column])
                ]
    return [augmented[index][-1] for index in range(size)]


def _mat_vec(matrix: Matrix3, vector: Vector3) -> Vector3:
    return tuple(sum(matrix[row][k] * vector[k] for k in range(3)) for row in range(3))


def solve_pivot(samples: Sequence[Pose]) -> TcpCalibrationResult:
    """Solve TCP translation and the fixed contact point by least squares."""
    if len(samples) < 4:
        raise ValueError("At least four pose samples are required")
    normal = [[0.0 for _ in range(6)] for _ in range(6)]
    normal_rhs = [0.0 for _ in range(6)]
    rotations: list[Matrix3] = []
    positions: list[Vector3] = []
    for pose in samples:
        if len(pose) != 6 or any(not math.isfinite(float(value)) for value in pose):
            raise ValueError("Each pose must contain six finite values")
        position = tuple(float(value) for value in pose[:3])
        rotation = rotation_matrix_xyz(*pose[3:])
        rotations.append(rotation)
        positions.append(position)
        for axis in range(3):
            row = [rotation[axis][0], rotation[axis][1], rotation[axis][2]]
            row.extend(-1.0 if axis == index else 0.0 for index in range(3))
            rhs = -position[axis]
            for left in range(6):
                normal_rhs[left] += row[left] * rhs
                for right in range(6):
                    normal[left][right] += row[left] * row[right]
    solution = _solve_linear_system(normal, normal_rhs)
    tcp = tuple(solution[:3])
    fixed = tuple(solution[3:])
    residuals = []
    for position, rotation in zip(positions, rotations):
        predicted = tuple(position[index] + _mat_vec(rotation, tcp)[index] for index in range(3))
        residuals.append(math.sqrt(sum((predicted[index] - fixed[index]) ** 2 for index in range(3))))
    rms = math.sqrt(sum(value * value for value in residuals) / len(residuals))
    return TcpCalibrationResult(
        tcp_flange_mm=tcp,
        fixed_point_user_mm=fixed,
        residuals_mm=tuple(residuals),
        rms_residual_mm=rms,
        max_residual_mm=max(residuals),
        sample_count=len(samples),
    )


def load_samples(path: Path) -> list[Pose]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("samples") if isinstance(payload, dict) else payload
    if not isinstance(values, list):
        raise ValueError("Sample file must contain a JSON list or {\"samples\": [...]}")
    samples = []
    for item in values:
        pose = item.get("pose") if isinstance(item, dict) else item
        if not isinstance(pose, list) or len(pose) != 6:
            raise ValueError("Each sample must provide pose=[x,y,z,rx,ry,rz]")
        samples.append(tuple(float(value) for value in pose))
    return samples


def _format_vector(values: Iterable[float]) -> str:
    return "[" + ", ".join(f"{value:.3f}" for value in values) + "]"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path, help="JSON file of GetPose samples")
    args = parser.parse_args(argv)
    result = solve_pivot(load_samples(args.samples))
    print(f"samples: {result.sample_count}")
    print(f"tcp_translation_mm: {_format_vector(result.tcp_flange_mm)}")
    print(f"fixed_point_user_mm: {_format_vector(result.fixed_point_user_mm)}")
    print(f"rms_residual_mm: {result.rms_residual_mm:.3f}")
    print(f"max_residual_mm: {result.max_residual_mm:.3f}")
    print("\nCandidate geometry YAML (review before copying):")
    print("tcp_translation_mm: " + _format_vector(result.tcp_flange_mm))
    print("tcp_rotation_deg: null  # solver only estimates translation")


if __name__ == "__main__":
    main()
