"""ApolloCar3D ground-truth loader: intrinsics, per-frame car poses.

Apollo pose = [r0, r1, r2, tx, ty, tz]: the first three are an axis-angle /
rotation vector (Rodrigues) of the CAD model in the camera frame, the last
three the translation in metres. (Apollo's dev kit treats the first three as
Euler angles; the local data here is axis-angle compatible with Rodrigues -
validated by the Task 8 visual overlay before any number is trusted.)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class ApolloCar:
    """One GT car in a frame: rotation, translation, CAD model id."""

    car_id: int
    r: np.ndarray  # (3, 3)
    t: np.ndarray  # (3,)


def load_intrinsics(cam_path: str | Path) -> np.ndarray:
    """Parse a `.cam` file into a 3x3 intrinsics matrix."""
    params: dict[str, float] = {}
    for line in Path(cam_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if "=" in line:
            key, _, val = line.partition("=")
            try:
                params[key.strip()] = float(val.strip())
            except ValueError:
                continue
    return np.array(
        [
            [params["fx"], 0.0, params["Cx"]],
            [0.0, params["fy"], params["Cy"]],
            [0.0, 0.0, 1.0],
        ]
    )


def pose6_to_rt(pose: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Convert a 6-vector [rvec(3), t(3)] into (R 3x3, t 3,)."""
    rvec = np.array(pose[:3], dtype=np.float64)
    r = cv2.Rodrigues(rvec)[0]
    t = np.array(pose[3:6], dtype=np.float64)
    return r, t


def load_frame_cars(pose_json: str | Path) -> list[ApolloCar]:
    """Load all GT cars from one Apollo car_poses JSON file."""
    data = json.loads(Path(pose_json).read_text(encoding="utf-8"))
    cars: list[ApolloCar] = []
    for entry in data:
        r, t = pose6_to_rt(entry["pose"])
        cars.append(ApolloCar(car_id=int(entry["car_id"]), r=r, t=t))
    return cars
