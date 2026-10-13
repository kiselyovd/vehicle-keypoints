# Vehicle Keypoints

[Open the interactive showcase](showcase.md)

![Monocular 3D pose wireframe on an ApolloCar3D image](images/pose3d_overlay.png)

*An example from the geometric PnP baseline; see the [protocol and limitations](usage-guide.md#monocular-3d-pose-pnp-baseline).*

Research toolkit for 14-keypoint vehicle pose estimation on the CarFusion (CMU) dataset - four wheels, four head- and tail-lights, four roof corners, the exhaust, and a body-centre reference point per car.

![CarFusion 14-keypoint skeleton · YOLO26-pose kpt_shape=[14, 3]](images/keypoints_skeleton.svg)

## Results and research

The main detector's published CarFusion evaluation and baseline caveats are documented in the [usage and evaluation guide](usage-guide.md#results). The 3D reconstruction example uses ApolloCar3D and has a separate evaluation protocol.

The [synthetic-data experiment](phase0/synthetic-pretraining.md) is complete: the tested synthetic supplement did not meet the **+2 pp OKS-mAP** criterion. Its contribution was **-1.7 pp** from the weak checkpoint and **-9.5 pp** from the strong one. The report distinguishes this negative result from the gain caused by correcting horizontal-flip keypoint labels.

## Task

Per-vehicle 2D landmark localisation on RGB images. 14 anatomical keypoints per car instance follow the original CarFusion (Reddy et al., CVPR 2018) schema: 4 wheels (front/rear × left/right), 4 head- and tail-lights (front/rear × left/right), 4 roof corners (front/rear × left/right), the exhaust, and a body-centre reference point. Each keypoint carries an `(x, y, v)` triplet where `v ∈ {0, 1, 2}` (absent / occluded / visible) following the COCO convention.

## Models

- **Main - YOLO26-pose** (Ultralytics 8.x). Anchor-free single-shot detector + keypoint head; non-human-keypoint-friendly, high throughput, standard `.pt` export.
- **Baseline - ViTPose-S** (HF `transformers` 5.x, Lightning-wrapped). Top-down (crop → heatmap) transformer; pretrained on COCO human 17-kpt, re-headed to 14 car keypoints.

## Stack

Python 3.13 · uv (cu124 torch wheels) · Ultralytics 8.x · transformers 5.x · PyTorch Lightning · Hydra · MLflow (ViTPose branch) · FastAPI · Docker · MkDocs Material · GitHub Actions.

## Navigation

- [Usage and evaluation](usage-guide.md) - quick start, dataset, API examples, 2D and 3D protocols
- [Synthetic-data experiment](phase0/synthetic-pretraining.md) - matched ablations and error analysis
- [Architecture](architecture.md) - data flow, two-branch pipeline, metric choices
- [Training](training.md) - YOLO + ViTPose commands, Hydra overrides, GPU notes
- [Serving](serving.md) - FastAPI `/detect`, Docker, JSON + PNG overlay responses
- [Model card](model_card.md.j2) - HF Hub card template (Jinja-rendered at publish time)

## Links

- **Code:** <https://github.com/kiselyovd/vehicle-keypoints>
- **Model:** <https://huggingface.co/kiselyovd/vehicle-keypoints>

## Intended use

Research and educational artifact demonstrating modern keypoint-detection pipelines on a non-human class. Not intended for any safety-critical, autonomous-driving, or surveillance deployment - the model is trained on a single academic dataset and has not been validated for production use.
