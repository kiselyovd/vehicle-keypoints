from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from phase0_train_v4 import oversample_real


def test_oversample_replicates_pairs(tmp_path):
    src = tmp_path / "real_subset"
    (src / "images" / "train").mkdir(parents=True)
    (src / "labels" / "train").mkdir(parents=True)
    (src / "images" / "train" / "a.png").write_bytes(b"x")
    (src / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    out = oversample_real(src, tmp_path / "real_x8", factor=8)
    imgs = list((out / "images" / "train").glob("*.png"))
    lbls = list((out / "labels" / "train").glob("*.txt"))
    assert len(imgs) == 8
    assert len(lbls) == 8
