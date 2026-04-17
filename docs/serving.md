# Serving

## Local

```bash
uv run uvicorn vehicle_keypoints.serving.main:app --reload
```

The app reads `MODEL_CHECKPOINT` from the environment (default `artifacts/best.pt`) at startup — set it to your trained weights before launching. Without a checkpoint on disk the app still boots but `/detect` returns 503.

## Docker

```bash
docker compose up api
```

Image: `ghcr.io/kiselyovd/vehicle-keypoints`. The container honours the same `MODEL_CHECKPOINT` env var; if unset, the entrypoint falls back to the bundled weights baked into the image at build time (or to the HF Hub download if neither is present).

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check — returns `{"status": "ok"}` |
| `POST` | `/detect` | Run keypoint detection on one image |
| `GET` | `/metrics` | Prometheus metrics (requests, latency, in-flight) |

## Input format

`POST /detect` takes a multipart upload of a single **RGB JPEG or PNG**, max ~10 MB. The backing YOLO26-pose model resizes to 640×640 internally (letterbox), so you do not need to pre-resize — the returned `image_width` / `image_height` always reflect the original upload dimensions and keypoint coordinates are rescaled back into that original frame.

## Headers

- `X-Request-ID` — echoed if the client sends one, otherwise minted server-side; propagate it from your upstream gateway for end-to-end traces.
- `X-Detections` — present on PNG overlay responses; integer count of detected vehicles in the image.

## JSON response schema

Default (or when `Accept: application/json`):

```json
{
  "detections": [
    {
      "bbox": [x, y, w, h],
      "keypoints": [
        {"x": 123.4, "y": 456.7, "v": 2}
      ],
      "score": 0.91
    }
  ],
  "image_width": 1920,
  "image_height": 1080,
  "request_id": "01H..."
}
```

- `bbox` is xywh in the original image frame.
- `keypoints` is always a list of **14 entries** in the canonical CarFusion order (wheels, head-/tail-lights, exhaust, roof corners, body centre — see `CARFUSION_KEYPOINT_NAMES` in `vehicle_keypoints.inference.overlay`). Missing keypoints carry `v=0` and meaningless `x`/`y`.
- `score` is the per-instance detection confidence.

## Example curl

JSON response:

```bash
curl -X POST -F "file=@car.jpg" http://localhost:8000/detect
```

PNG overlay response:

```bash
curl -X POST -F "file=@car.jpg" \
  -H "Accept: image/png" \
  -o out.png \
  http://localhost:8000/detect
```

The PNG return draws boxes, keypoint dots, and the car skeleton (edges defined in `vehicle_keypoints.inference.skeleton`) on a copy of the uploaded image.
