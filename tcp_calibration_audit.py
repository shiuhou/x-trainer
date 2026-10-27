"""Offline pivot diagnostics, with NumPy SVD as an independent numerical reference.

Outputs are derived diagnostics, never accepted Tool settings. Input samples
remain private; the report stores their hash rather than raw poses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from tcp_calibration import load_samples, rotation_matrix_xyz, solve_pivot


def audit(path: Path) -> dict:
    samples = load_samples(path)
    fitted = solve_pivot(samples)
    rotations = [np.array(rotation_matrix_xyz(*p[3:])) for p in samples]
    a = np.vstack([np.c_[r, -np.eye(3)] for r in rotations])
    b = -np.array([p[:3] for p in samples]).ravel()
    solution, _, rank, singular = np.linalg.lstsq(a, b, rcond=None)
    # Leave-one-out re-fitting measures sensitivity, not independent metrology.
    loo: list[float | None] = []
    if len(samples) >= 5:
        for i, pose in enumerate(samples):
            try:
                fit = solve_pivot(samples[:i] + samples[i + 1 :])
                error = (
                    np.array(pose[:3])
                    + rotations[i] @ fit.tcp_flange_mm
                    - fit.fixed_point_user_mm
                )
                loo.append(float(np.linalg.norm(error)))
            except ValueError:
                loo.append(None)
    return {
        "schema": 1,
        "scope": "offline pivot consistency, not physical accuracy or TCP orientation",
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "fit": asdict(fitted),
        "design_rank": int(rank),
        "design_singular_values": singular.tolist(),
        "design_condition": float(np.linalg.cond(a)),
        "svd_tcp_translation_mm": solution[:3].tolist(),
        "normal_equation_vs_svd_translation_mm": float(
            np.linalg.norm(solution[:3] - fitted.tcp_flange_mm)
        ),
        "leave_one_out_errors_mm": loo,
        "independent_physical_error_mm": None,
        "calibration_accepted": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.samples)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
