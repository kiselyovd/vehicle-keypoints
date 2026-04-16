# vehicle-keypoints

![vehicle-keypoints · YOLO26-pose 14-kpt car detector](images/hero.png)

Production-grade 14-keypoint vehicle pose estimation on the CarFusion (CMU) dataset — wheels, headlights, taillights, bumpers, side-mirrors, and license plates per car.

![CarFusion 14-keypoint skeleton · YOLO26-pose kpt_shape=[14, 3]](images/keypoints_skeleton.svg)

## Task

Per-vehicle 2D landmark localisation on RGB images. 14 anatomical keypoints per car instance: 4 wheels (front/rear × left/right), 4 lights (front/rear × left/right), 2 bumpers (front/rear), 2 side-mirrors (left/right), 2 license plates (front/rear). Each keypoint carries an `(x, y, v)` triplet where `v ∈ {0, 1, 2}` (absent / occluded / visible) following the COCO convention.

## Models

- **Main — YOLO26-pose** (Ultralytics 8.x). Anchor-free single-shot detector + keypoint head; non-human-keypoint-friendly, high throughput, standard `.pt` export.
- **Baseline — ViTPose-S** (HF `transformers` 5.x, Lightning-wrapped). Top-down (crop → heatmap) transformer; pretrained on COCO human 17-kpt, re-headed to 14 car keypoints.

## Stack

Python 3.13 · uv (cu124 torch wheels) · Ultralytics 8.x · transformers 5.x · PyTorch Lightning · Hydra · MLflow (ViTPose branch) · FastAPI · Docker · MkDocs Material · GitHub Actions.

## Navigation

- [Architecture](architecture.md) — data flow, two-branch pipeline, metric choices
- [Training](training.md) — YOLO + ViTPose commands, Hydra overrides, GPU notes
- [Serving](serving.md) — FastAPI `/detect`, Docker, JSON + PNG overlay responses
- [Model card](model_card.md.j2) — HF Hub card template (Jinja-rendered at publish time)

## Links

- **Code:** <https://github.com/kiselyovd/vehicle-keypoints>
- **Model:** <https://huggingface.co/kiselyovd/vehicle-keypoints>

## Intended use

Research and educational artifact demonstrating modern keypoint-detection pipelines on a non-human class. Not intended for any safety-critical, autonomous-driving, or surveillance deployment — the model is trained on a single academic dataset and has not been validated for production use.
