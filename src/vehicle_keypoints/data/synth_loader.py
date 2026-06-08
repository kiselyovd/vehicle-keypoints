"""Loader for the UE5 synthetic dataset (Phase 0 vertical slice).

This module is a Phase 0 stop-gap so we can prove the sim-to-real signal
before investing in the full multi-source dataloader described in the v2 spec.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

_SchemaName = Literal["carfusion14", "extended24"]


class SynthPhase0Dataset(Dataset):
    def __init__(self, root: Path, schema: _SchemaName = "carfusion14") -> None:
        self.root = Path(root)
        self.schema = schema

        coco_path = self.root / "annotations" / "coco.json"
        data = json.loads(coco_path.read_text(encoding="utf-8"))
        self._images_by_id = {img["id"]: img for img in data["images"]}
        self._annotations = list(data["annotations"])

    def __len__(self) -> int:
        return len(self._annotations)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ann = self._annotations[idx]
        img_meta = self._images_by_id[ann["image_id"]]
        img_path = self.root / img_meta["file_name"]
        image = np.asarray(Image.open(img_path).convert("RGB"))

        flat = ann["keypoints"]
        kpts_24 = np.array(flat, dtype=np.float32).reshape(24, 3)

        if self.schema == "carfusion14":
            kpts = kpts_24[:14]
        else:
            kpts = kpts_24

        bbox = np.array(ann["bbox"], dtype=np.float32)

        return {
            "image": torch.from_numpy(image).permute(2, 0, 1).float() / 255.0,
            "keypoints": torch.from_numpy(kpts).float(),
            "bbox": torch.from_numpy(bbox).float(),
        }
