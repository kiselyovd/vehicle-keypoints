from __future__ import annotations

import json
from pathlib import Path

import pytest

from vehicle_keypoints.data.synth_loader import SynthPhase0Dataset


@pytest.fixture
def synth_dir(tmp_path: Path) -> Path:
    rgb = tmp_path / "rgb"
    rgb.mkdir()
    ann = tmp_path / "annotations"
    ann.mkdir()

    # Two placeholder PNGs (any valid PNG; pillow can write a 32x32 white square)
    from PIL import Image
    for i in (1, 2):
        Image.new("RGB", (32, 32), (255, 255, 255)).save(rgb / f"frame_{i:06d}_cam0.png")

    coco = {
        "info": {},
        "images": [
            {"id": 1, "file_name": "rgb/frame_000001_cam0.png", "width": 32, "height": 32, "metadata": {}},
            {"id": 2, "file_name": "rgb/frame_000002_cam0.png", "width": 32, "height": 32, "metadata": {}},
        ],
        "annotations": [
            {
                "id": 1, "image_id": 1, "category_id": 1,
                "bbox": [4, 4, 20, 20], "area": 400, "iscrowd": 0,
                "keypoints": [10, 10, 2] + [0, 0, 0] * 23,
                "num_keypoints": 1,
            },
            {
                "id": 2, "image_id": 2, "category_id": 1,
                "bbox": [4, 4, 20, 20], "area": 400, "iscrowd": 0,
                "keypoints": [12, 12, 2] + [0, 0, 0] * 23,
                "num_keypoints": 1,
            },
        ],
        "categories": [{"id": 1, "name": "vehicle", "keypoints": ["x"] * 24, "skeleton": []}],
    }
    (ann / "coco.json").write_text(json.dumps(coco))
    return tmp_path


def test_synth_loader_returns_samples(synth_dir: Path) -> None:
    ds = SynthPhase0Dataset(root=synth_dir, schema="carfusion14")
    assert len(ds) == 2

    sample = ds[0]
    assert "image" in sample
    assert "keypoints" in sample
    # carfusion14 mode returns 14 points (the first 14 from the synth's 24-pt schema)
    assert sample["keypoints"].shape == (14, 3)


def test_synth_loader_extended_schema(synth_dir: Path) -> None:
    ds = SynthPhase0Dataset(root=synth_dir, schema="extended24")
    sample = ds[0]
    assert sample["keypoints"].shape == (24, 3)
