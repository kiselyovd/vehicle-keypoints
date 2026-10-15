# Vehicle Keypoints

Detect 14 vehicle keypoints in road images, evaluate pose quality, and explore monocular 3D reconstruction and synthetic-data transfer.

[![CI](https://github.com/kiselyovd/vehicle-keypoints/actions/workflows/ci.yml/badge.svg)](https://github.com/kiselyovd/vehicle-keypoints/actions/workflows/ci.yml) [![Docs](https://img.shields.io/badge/docs-MkDocs-blue)](https://kiselyovd.github.io/vehicle-keypoints/) [![HF Model](https://img.shields.io/badge/Hugging_Face-model-yellow)](https://huggingface.co/kiselyovd/vehicle-keypoints) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

YOLO26-pose is the main detector; ViTPose-S is an under-trained reference baseline. The project includes scene-level data preparation, evaluation, FastAPI serving, and published weights.

[![Interactive viewer of source annotations and recorded predictions](docs/images/result-example.png)](https://kiselyovd.github.io/vehicle-keypoints/showcase/)

**[Explore the keypoints](https://kiselyovd.github.io/vehicle-keypoints/showcase/)** — explore six traffic scenes with 3–7 labelled vehicles each. See all vehicles together, switch between annotations and recorded checkpoint predictions, or zoom to one car. The curated examples have pinned source records and are separate from the test benchmark. [3D/PnP companion](docs/usage-guide.md#monocular-3d-pose-pnp-baseline).

[Model & weights](https://huggingface.co/kiselyovd/vehicle-keypoints) · [Documentation](https://kiselyovd.github.io/vehicle-keypoints/) · [Русский README](README.ru.md)

**[Run live inference](https://huggingface.co/spaces/kiselyovd/vehicle-keypoints-demo)** on your own image. The Space uses the published real-data checkpoint; the explorer above uses original annotations and recorded outputs.

<!-- ARTIFACT-OVERVIEW:START -->
OKS-mAP **50.38%** · OKS-mAP50 **70.36%**. [Evaluation and scope](https://github.com/kiselyovd/vehicle-keypoints/blob/main/reports/metrics.json).

**Evidence:** Stored CarFusion evaluation; not re-executed as part of this audit. 12761 source test frames, OKS sigmas 0.05, PCK threshold 0.05 of bbox diagonal. Historical report is tracked; original training membership and immutable dataset revision were not recorded. Curated demo outputs are separate illustrations.
<!-- ARTIFACT-OVERVIEW:END -->

## Try the detector

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/). From the repository root:

Download the weights, then run inference:

```bash
uv sync --all-groups
uv run python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('kiselyovd/vehicle-keypoints', 'weights.pt', revision='d277c2d8c9b5ee49d9fed85fe43bdd72b16a0f06', local_dir='artifacts/hf'))"
uv run python -m vehicle_keypoints.inference.predict \
  --checkpoint artifacts/hf/weights.pt --input data/sample/images/car_butler1__16_04881.jpg
```

This runs inference on a bundled CarFusion sample. Output is JSON with boxes, confidence scores, and 14 `(x, y, score)` keypoints per detection. Boxes use `(x, y, width, height)`.

## Results and research

<!-- ARTIFACT-MANIFEST:START -->
### Current artifact and evaluation

**Stored CarFusion evaluation; not re-executed as part of this audit**

| Metric | Value |
|---|---:|
| OKS-mAP | 50.38% |
| OKS-mAP50 | 70.36% |
| PCK@0.05 | 76.06% |

12761 source test frames, OKS sigmas 0.05, PCK threshold 0.05 of bbox diagonal. Historical report is tracked; original training membership and immutable dataset revision were not recorded. Curated demo outputs are separate illustrations.

[Evaluation record](https://github.com/kiselyovd/vehicle-keypoints/blob/main/reports/metrics.json) · [Pinned model](https://huggingface.co/kiselyovd/vehicle-keypoints/tree/d277c2d8c9b5ee49d9fed85fe43bdd72b16a0f06)
<!-- ARTIFACT-MANIFEST:END -->

See [stored metrics](reports/metrics_summary.json) and the [evaluation notes](docs/usage-guide.md#results). The ViTPose run used 15 epochs and `val=train`; these numbers do not establish a fair comparison of architecture capacity.

The [UE5 pipeline](https://github.com/kiselyovd/ue5-vehicle-synth) supplies synthetic frames and a compatible 24-point schema. In the published matched experiments, adding synthetic data reduced real-data OKS-mAP by **1.7 pp** from the weak checkpoint and **9.5 pp** from the strong checkpoint. The **+2 pp** acceptance criterion was not met. The [experiment report](docs/phase0/synthetic-pretraining.md) separates this result from the much larger gain caused by correcting horizontal-flip labels.

## Documentation

| Guide | Covers |
|---|---|
| [Usage and evaluation](docs/usage-guide.md) | Dataset, training commands, Python API, 3D pose, and evaluation caveats |
| [Architecture](docs/architecture.md) | Components and data flow |
| [Training](docs/training.md) | Detector and baseline recipes |
| [Serving](docs/serving.md) | FastAPI and Docker |
| [Synthetic-data experiment](docs/phase0/synthetic-pretraining.md) | Matched ablations, negative results, and follow-up hypotheses |

## Limitations and licenses

Research and education use. CarFusion is a limited traffic-scene dataset, and the 3D baseline has substantial domain shift and limited detector coverage.

Repository code is [MIT](LICENSE). CarFusion, ApolloCar3D, Ultralytics, and published model artifacts have separate terms; consult their licenses and model cards before reuse.
