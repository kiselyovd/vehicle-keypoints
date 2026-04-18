"""Finalize v0.1.0: ViTPose baseline + YOLO eval + ViTPose eval + summary.

Assumes YOLO is already trained (artifacts/sota/weights/best.pt exists).
Runs everything in one process to amortize the 30-min Windows torch-DLL scan.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def train_vitpose() -> Path:
    log("=== ViTPose baseline training ===")
    import lightning as L
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    from vehicle_keypoints.data import KeypointsDataModule
    from vehicle_keypoints.models import KeypointsModule, build_model

    dm = KeypointsDataModule(
        train_images="data/raw/train",
        train_annotations="data/raw/annotations/car_keypoints_train.json",
        val_images="data/raw/train",
        val_annotations="data/raw/annotations/car_keypoints_train.json",
        batch_size=32,
        num_workers=0,
        image_size=256,
        seed=42,
    )
    backbone = build_model("vitpose_s", num_keypoints=14, pretrained=True)
    lit = KeypointsModule(backbone, num_keypoints=14, lr=5e-4, model_name="vitpose_s")

    out_dir = Path("artifacts/baseline").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=out_dir / "checkpoints",
        filename="best",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
    )
    early_cb = EarlyStopping(monitor="val/loss", mode="min", patience=5)
    trainer = L.Trainer(
        max_epochs=15,
        accelerator="gpu",
        devices=1,
        callbacks=[ckpt_cb, early_cb],
        logger=False,
        log_every_n_steps=50,
        deterministic="warn",
    )
    trainer.fit(lit, dm)
    best = out_dir / "checkpoints" / "best.ckpt"
    log(f"ViTPose best.ckpt = {best}")
    return best


def score_yolo() -> Path:
    log("=== YOLO scoring (OKS-mAP + PCK) ===")
    import subprocess

    ckpt = Path("artifacts/sota/weights/best.pt")
    assert ckpt.exists(), f"missing {ckpt}"
    out = Path("reports/metrics.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "vehicle_keypoints.evaluation.evaluate",
        "--checkpoint",
        str(ckpt),
        "--out",
        str(out),
    ]
    log(f"running {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return out


def vitpose_predict_and_score(vitpose_ckpt: Path) -> Path:
    log("=== ViTPose predictions + scoring ===")
    import numpy as np
    import torch

    from vehicle_keypoints.data.coco_dataset import CocoKeypointsDataset
    from vehicle_keypoints.models import KeypointsModule, build_model

    backbone = build_model("vitpose_s", num_keypoints=14, pretrained=False)
    lit = KeypointsModule.load_from_checkpoint(str(vitpose_ckpt), model=backbone)
    lit.train(False)
    lit = lit.cuda()

    ds = CocoKeypointsDataset("data/raw/test", "data/raw/annotations/car_keypoints_test.json")

    results = []
    for idx in range(len(ds)):
        if idx % 500 == 0:
            log(f"  ViTPose inference {idx}/{len(ds)}")
        crop, _, _ = ds[idx]
        ann = ds.annotations[idx]
        with torch.inference_mode():
            hm = lit._forward_heatmaps(crop.unsqueeze(0).cuda())
        hm_np = hm.squeeze(0).cpu().numpy()
        K, H, W = hm_np.shape  # noqa: N806 — ML convention for shape dims
        bx, by, bw, bh = ann["bbox"]
        kpts_out = []
        for k in range(K):
            flat = int(np.argmax(hm_np[k]))
            py, px = np.unravel_index(flat, (H, W))
            ix = bx + (px / W) * bw
            iy = by + (py / H) * bh
            kpts_out.extend([float(ix), float(iy), 2.0])
        results.append(
            {
                "image_id": ann["image_id"],
                "category_id": 1,
                "bbox": ann["bbox"],
                "keypoints": kpts_out,
                "score": 1.0,
            }
        )
    preds = Path("reports/baseline_predictions.json")
    preds.parent.mkdir(parents=True, exist_ok=True)
    preds.write_text(json.dumps(results), encoding="utf-8")
    log(f"wrote {len(results)} baseline predictions")

    import subprocess

    out = Path("reports/metrics_baseline.json")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "vehicle_keypoints.evaluation.evaluate",
            "--predictions",
            str(preds),
            "--out",
            str(out),
        ],
        check=True,
    )
    return out


def build_summary(main_m: Path, base_m: Path) -> Path:
    log("=== metrics summary ===")
    m = json.loads(main_m.read_text(encoding="utf-8"))
    mb = json.loads(base_m.read_text(encoding="utf-8"))
    summary = {
        "Main model": "YOLO26-pose",
        "Main OKS-mAP": f"{m.get('oks_map', 0.0) * 100:.1f}%",
        "Main OKS-mAP50": f"{m.get('oks_map_50', 0.0) * 100:.1f}%",
        "Main PCK@0.05": f"{m.get('pck_0.05', 0.0) * 100:.1f}%",
        "Baseline model": "ViTPose-S",
        "Baseline OKS-mAP": f"{mb.get('oks_map', 0.0) * 100:.1f}%",
        "Baseline OKS-mAP50": f"{mb.get('oks_map_50', 0.0) * 100:.1f}%",
        "Baseline PCK@0.05": f"{mb.get('pck_0.05', 0.0) * 100:.1f}%",
        "Test size (images)": m.get("test_size", 0),
    }
    out = Path("reports/metrics_summary.json")
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(json.dumps(summary, indent=2))
    return out


def main() -> None:
    t0 = time.time()
    vp_ckpt = train_vitpose()
    score_yolo()
    vitpose_predict_and_score(vp_ckpt)
    build_summary(Path("reports/metrics.json"), Path("reports/metrics_baseline.json"))
    log(f"TOTAL TIME: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
