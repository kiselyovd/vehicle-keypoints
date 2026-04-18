"""Direct Ultralytics training — bypass Hydra + settings.

Usage: uv run python -u scripts/direct_train_yolo.py  >> logs/yolo.log 2>&1

Tuned for RTX 3080 10 GB on Windows:
- workers=0 avoids the Ultralytics multiprocessing hang on Windows.
- cache=True preloads decoded numpy arrays to disk .npy cache — first epoch slow, rest fast.
- imgsz=480 halves FLOPs vs 640 with minimal keypoint quality loss.
- batch=32 saturates GPU at imgsz=480.
"""

from __future__ import annotations

import sys
from pathlib import Path

import ultralytics
from ultralytics import YOLO


def main() -> None:
    artifacts = Path("artifacts").resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    ultralytics.settings.update({"runs_dir": str(artifacts).replace("\\", "/")})

    print("=== YOLO direct training ===", flush=True)
    print(f"artifacts = {artifacts}", flush=True)

    model = YOLO("yolo26n-pose.pt")
    model.train(
        data="data/processed/data.yaml",
        epochs=30,
        imgsz=480,
        batch=32,
        workers=0,
        patience=8,
        project=str(artifacts),
        name="sota",
        cache=True,  # disk npy cache; ~10 GB one-time write
        verbose=True,
        plots=True,
    )
    print("=== DONE ===", flush=True)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
