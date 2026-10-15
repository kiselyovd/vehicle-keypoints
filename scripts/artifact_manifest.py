"""Validate and render an artifact contract using only the Python standard library."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import urllib.request
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)
STATUSES = {"reported", "replayed_reconstructed", "verified_held_out", "unavailable"}


def local_path(root: Path, name: str) -> Path:
    """Resolve only repository-owned paths, including on Windows."""
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Manifest path escapes repository")
    return path


def digest(path: Path) -> str:
    """Hash a bounded artifact stream."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pointer(value: Any, path: str) -> float:
    """Resolve an explicit JSON pointer without substituting missing metrics."""
    if not path.startswith("/"):
        raise ValueError("Metric source must be an absolute JSON pointer")
    for key in path[1:].split("/"):
        key = key.replace("~1", "/").replace("~0", "~")
        value = value[int(key)] if isinstance(value, list) else value[key]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("Metric source is not numeric")
    if not 0 <= value <= 1:
        raise ValueError("Displayed score must be a finite fraction in [0,1]")
    return float(value)


def validate(
    root: Path, name: str = "artifact-manifest.json"
) -> tuple[dict[str, Any], dict[str, float]]:
    """Check pinned files, report hashes, score sources and evaluation claims."""
    manifest = json.loads(local_path(root, name).read_text(encoding="utf-8"))
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported artifact manifest schema")
    if manifest["hub"]["repo_id"].split("/")[0] != "kiselyovd":
        raise ValueError("Expected personal Hub owner kiselyovd")
    if not re.fullmatch(r"[0-9a-f]{40}", manifest["hub"]["revision"]):
        raise ValueError("Hub revision must be immutable")
    files = manifest["hub"]["files"]
    for item in files:
        if not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
            raise ValueError("Invalid artifact SHA256")
    result = manifest["evaluation"]
    if result["status"] not in STATUSES:
        raise ValueError("Unknown evaluation status")
    if result["status"] == "unavailable":
        if result.get("metrics"):
            raise ValueError("Unavailable evaluation must not advertise numeric metrics")
        return manifest, {}
    report_path = local_path(root, result["report"])
    if digest(report_path) != result["report_sha256"]:
        raise ValueError("Evaluation report SHA256 mismatch")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if result["status"] in {"replayed_reconstructed", "verified_held_out"}:
        weights = {f["sha256"] for f in files if f["role"] == "weights"}
        if report["artifact"]["sha256"] not in weights:
            raise ValueError("Replay evaluated different weights")
        if result["training_membership_verified"] is not report["training_membership_verified"]:
            raise ValueError("Training membership claim differs from report")
    if result["status"] == "verified_held_out" and not result["training_membership_verified"]:
        raise ValueError("Held-out claim requires verified training membership")
    arrays = report.get("prediction_arrays", report.get("arrays"))
    if arrays:
        array_path = report_path.parent / arrays.get("file", arrays.get("path", ""))
        checked = local_path(root, str(array_path.relative_to(root)))
        if digest(checked) != arrays["sha256"]:
            raise ValueError("Prediction array SHA256 mismatch")
    values = {m["name"]: pointer(report, m["pointer"]) for m in result["metrics"]}
    log.info(
        "Manifest validated: repo=%s, status=%s, metrics=%d",
        manifest["hub"]["repo_id"],
        result["status"],
        len(values),
    )
    return manifest, values


def render(manifest: dict[str, Any], values: dict[str, float]) -> str:
    """Generate the human-readable contract from the same score sources as CI."""
    evaluation = manifest["evaluation"]
    if evaluation["status"] == "unavailable":
        return "\n".join(
            [
                "<!-- ARTIFACT-MANIFEST:START -->",
                "### Current artifact and evaluation",
                "",
                evaluation["label"],
                "",
                evaluation["scope"],
                "<!-- ARTIFACT-MANIFEST:END -->",
            ]
        )
    hub_prefix = "datasets/" if manifest["hub"]["repo_type"] == "dataset" else ""
    report_base = f"https://github.com/kiselyovd/{manifest['project']}/blob/main/"
    lines = [
        "<!-- ARTIFACT-MANIFEST:START -->",
        "### Current artifact and evaluation",
        "",
        f"**{evaluation['label']}**",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    lines.extend(f"| {key} | {value:.2%} |" for key, value in values.items())
    lines.extend(
        [
            "",
            evaluation["scope"],
            "",
            f"[Evaluation record]({report_base}{evaluation['report']}) · "
            f"[Pinned model](https://huggingface.co/{hub_prefix}{manifest['hub']['repo_id']}/tree/{manifest['hub']['revision']})",
            "<!-- ARTIFACT-MANIFEST:END -->",
        ]
    )
    return "\n".join(lines)


def render_overview(manifest: dict[str, Any], values: dict[str, float]) -> str:
    """Render the first-fold result from the same checked numerical source."""
    evaluation = manifest["evaluation"]
    scores = " · ".join(f"{key} **{value:.2%}**" for key, value in list(values.items())[:2])
    report = f"https://github.com/kiselyovd/{manifest['project']}/blob/main/{evaluation['report']}"
    return "\n".join(
        [
            "<!-- ARTIFACT-OVERVIEW:START -->",
            f"{scores}. [Evaluation and scope]({report}).",
            "",
            f"**Evidence:** {evaluation['label']}. {evaluation['scope']}",
            "<!-- ARTIFACT-OVERVIEW:END -->",
        ]
    )


def verify_hub(manifest: dict[str, Any]) -> None:
    """Compare immutable Hub file metadata checksums without downloading model stores."""
    hub = manifest["hub"]
    kind = {"model": "models", "dataset": "datasets"}[hub["repo_type"]]
    url = (
        f"https://huggingface.co/api/{kind}/{hub['repo_id']}/revision/{hub['revision']}?blobs=true"
    )
    with urllib.request.urlopen(url, timeout=30) as response:  # nosec B310
        data = json.load(response)
    files = {item["rfilename"]: item for item in data["siblings"]}
    prefix = "datasets/" if hub["repo_type"] == "dataset" else ""
    for expected in hub["files"]:
        entry = files[expected["path"]]
        checksum = entry.get("lfs", {}).get("sha256")
        if checksum is None:
            raw_url = (
                f"https://huggingface.co/{prefix}{hub['repo_id']}/resolve/"
                f"{hub['revision']}/{expected['path']}"
            )
            if entry.get("size", 0) > 2 * 1024**2:
                raise ValueError(
                    "Hub omitted large artifact checksum; refusing an unbounded download"
                )
            with urllib.request.urlopen(raw_url, timeout=30) as response:  # nosec B310
                content = response.read(2 * 1024**2 + 1)
            if len(content) > 2 * 1024**2:
                raise ValueError("Small-file checksum read exceeded budget")
            checksum = hashlib.sha256(content).hexdigest()
        if checksum != expected["sha256"]:
            raise ValueError(f"Pinned Hub checksum mismatch: {expected['path']}")
    log.info("Pinned Hub metadata verified: repo=%s files=%d", hub["repo_id"], len(hub["files"]))


def main() -> None:
    """Validate alone, or generate/check a marked README contract block."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", default="artifact-manifest.json")
    parser.add_argument("--check-hub", action="store_true")
    parser.add_argument("--document", default="README.md")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    manifest, values = validate(args.root, args.manifest)
    if args.check_hub:
        verify_hub(manifest)
    if not (args.write or args.check):
        print(render(manifest, values))
        return
    readme = local_path(args.root, args.document)
    text = readme.read_text(encoding="utf-8")
    pattern = r"<!-- ARTIFACT-MANIFEST:START -->.*?<!-- ARTIFACT-MANIFEST:END -->"
    if len(re.findall(pattern, text, flags=re.S)) != 1:
        raise ValueError("README must contain exactly one artifact manifest block")
    updated = re.sub(pattern, lambda _: render(manifest, values), text, flags=re.S)
    overview = r"<!-- ARTIFACT-OVERVIEW:START -->.*?<!-- ARTIFACT-OVERVIEW:END -->"
    if "<!-- ARTIFACT-OVERVIEW:" in text:
        if len(re.findall(overview, text, flags=re.S)) != 1:
            raise ValueError("README must contain exactly one overview block")
        updated = re.sub(overview, lambda _: render_overview(manifest, values), updated, flags=re.S)
    if args.write:
        readme.write_text(updated, encoding="utf-8", newline="\n")
    elif updated != text:
        raise ValueError("README artifact contract drift: regenerate with --write")


if __name__ == "__main__":
    main()
