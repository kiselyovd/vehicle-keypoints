# Published dataset and model split lineage

The published dataset at `bf2e175ddfb0b2dcd8dee3264f38eec3c19e643b` contains **1,996 train / 257 validation / 257 test frames**. The synthetic-only checkpoint at `bb0971f76e0944a3dd90faa236cbff37547b5590` was trained with a different **2,259 train / 251 validation** split.

All **2,510** actual pinned Parquet IDs, filenames and dimensions were compared with the source COCO. All **14,286** annotations were compared at absolute tolerance 0.0005 for float32 bbox/keypoint values. Only annotation/identity columns were read, approximately 5 MiB; image-byte equality was not tested.

| Published dataset split | Existing checkpoint train | Existing checkpoint validation |
|---|---:|---:|
| train | 1,792 | 204 |
| validation | 233 | 24 |
| test | **234** | 23 |

**The published test split is not independent of this checkpoint's training data.** Do not evaluate the released model on Hub test and present the result as held-out generalisation. The reported synthetic validation results refer to the original 251-frame model validation fold. Frame splits also share related capture/orbit views, so they do not prove independence by scene or vehicle identity. This does not change the separate matched real-CarFusion transfer experiment.

[Exact ID/path crosswalk](https://github.com/kiselyovd/ue5-vehicle-synth/blob/main/reports/provenance/synthetic/crosswalk.jsonl) · [Parquet comparison receipt](https://github.com/kiselyovd/ue5-vehicle-synth/blob/main/reports/provenance/synthetic/parquet-membership.json) · [Source/training identity](https://github.com/kiselyovd/ue5-vehicle-synth/blob/main/reports/provenance/synthetic/source.json)

Crosswalk SHA256: `840a9f538f95eed38889e2cdd4ab5ac97928d183a469bd57175ec980b92fcd0c`. The dataset and model splits remain unchanged; this receipt documents their relationship.

## Validate or repeat

`python scripts/validate_lineage.py` verifies the tracked crosswalk and report, unique IDs, exact counts and actual training basename checksums. To repeat the remote column comparison, run `scripts/verify_hub_lineage.py --coco PATH --training-images PATH --out PATH` in the declared project environment. The original source COCO and training image lists are required; the script rejects a changed COCO and oversized remote reads. No full image-column download is needed.

## Separate component terms

Code and author-created annotations are declared MIT. Rendered images incorporate Epic City Sample content. The [Epic Content License Agreement](https://www.unrealengine.com/eula/content) separately addresses rendered linear media, UE-only content and other restrictions; the author's MIT declaration does not relicense Epic asset files. The dataset contains renders/annotations rather than source meshes or textures. Historical legal notes are the author's earlier interpretation, not a replacement for applicable upstream terms.
