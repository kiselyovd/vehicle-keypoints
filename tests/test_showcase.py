"""Production documentation and source-artifact regression checks.

Run with: python -m unittest discover -s tests -p test_showcase.py
Requires the existing docs dependency group, without importing the ML runtime.
"""

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/assets/showcase"
DATA = json.loads((ASSETS / "data.json").read_text(encoding="utf-8"))


class Elements(HTMLParser):
    """Collect the emitted HTML, rather than inspecting source templates."""

    def __init__(self, text):
        super().__init__()
        self.nodes = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        self.nodes.append((tag, dict(attrs)))


class ShowcaseTests(unittest.TestCase):
    """Exercise the real MkDocs root and the data displayed by its UI."""

    @unittest.skipUnless(
        importlib.util.find_spec("mkdocs"),
        "Requires the docs dependency group; exercised by the Docs workflow.",
    )
    def test_production_root_and_local_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            site = Path(directory)
            build = subprocess.run(
                [sys.executable, "-m", "mkdocs", "build", "--site-dir", str(site)],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            page = site / "showcase/index.html"
            text = page.read_text(encoding="utf-8")
            nodes = Elements(text).nodes
            roots = [a for _, a in nodes if a.get("data-showcase") == DATA["kind"]]
            self.assertEqual(len(roots), 1)
            self.assertEqual(roots[0]["data-owner"], "kiselyovd")
            self.assertTrue(roots[0].get("aria-label"))
            self.assertIn("interactive controls load when JavaScript is enabled", text)
            self.assertTrue(
                any(
                    tag == "script"
                    and a.get("type") == "module"
                    and a.get("src", "").endswith("showcase/explorer.js")
                    for tag, a in nodes
                )
            )
            self.assertIn(DATA["source_sha"], text)
            pages = [page]
            guide = site / "project-guide/index.html"
            if guide.exists():
                pages.append(guide)
            for html in pages:
                for tag, attrs in Elements(html.read_text(encoding="utf-8")).nodes:
                    key = "src" if tag in {"img", "script"} else "href"
                    value = attrs.get(key, "")
                    url = urlsplit(value)
                    if not url.path or url.scheme or url.netloc:
                        continue
                    target = (
                        (site / unquote(url.path).lstrip("/"))
                        if url.path.startswith("/")
                        else (html.parent / unquote(url.path))
                    )
                    self.assertTrue(target.exists(), f"{html.name}: {tag} {value}")

    def test_annotations_and_recorded_prediction_provenance(self):
        coco = json.loads((ROOT / "data/sample/annotations.json").read_text(encoding="utf-8"))
        schema = coco["categories"][0]
        self.assertEqual(DATA["keypoints"], schema["keypoints"])
        self.assertEqual(DATA["skeleton"], schema["skeleton"])
        self.assertEqual(len(DATA["keypoints"]), 14)
        self.assertTrue(all(0 <= i < 14 for edge in DATA["skeleton"] for i in edge))
        self.assertEqual(len(DATA["samples"]), len(coco["images"]))
        original = {a["id"]: a for a in coco["annotations"]}
        for sample in DATA["samples"]:
            image = ASSETS / sample["image"]
            self.assertEqual(hashlib.sha256(image.read_bytes()).hexdigest(), sample["image_sha256"])
            self.assertEqual(
                image.read_bytes(), (ROOT / "data/sample/images" / image.name).read_bytes()
            )
            for annotation in sample["annotations"]:
                self.assertEqual(annotation["bbox"], original[annotation["id"]]["bbox"])
                self.assertEqual(annotation["keypoints"], original[annotation["id"]]["keypoints"])
            for prediction in sample["predictions"]:
                self.assertEqual(len(prediction["keypoints"]), 42)
                self.assertTrue(0 <= prediction["score"] <= 1)
                self.assertTrue(all(0 <= p <= 1 for p in prediction["keypoints"][2::3]))
        provenance = DATA["prediction_provenance"]
        self.assertEqual(provenance["model"], "kiselyovd/vehicle-keypoints")
        self.assertEqual(len(provenance["revision"]), 40)
        self.assertEqual(len(provenance["weights_sha256"]), 64)
        self.assertIn("not the full evaluation split", provenance["dataset_partition"])
