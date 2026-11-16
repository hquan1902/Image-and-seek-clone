"""Annotation-only adapters; no argparse, GPU or model loading at import time."""
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_split(data, dataset, split="val", category=None):
    root = Path(data)
    if dataset == "circo":
        if split not in ("val", "test"):
            raise ValueError("CIRCO has no training split")
        root /= "CIRCO"
        rows = read(root / "annotations" / f"{split}.json")
        image_info = read(root / "COCO2017_unlabeled/annotations/image_info_unlabeled2017.json")
        gallery = {str(x["id"]): str(root / "COCO2017_unlabeled/unlabeled2017" / x["file_name"]) for x in image_info["images"]}
        queries = [{"id": str(r["id"]), "reference": str(r["reference_img_id"]),
                    "target": str(r["target_img_id"]) if "target_img_id" in r else None,
                    "text": r["relative_caption"].strip()} for r in rows]
    elif dataset == "cirr":
        if split not in ("train", "val", "test", "test1"):
            raise ValueError("Unknown CIRR split")
        split = "test1" if split == "test" else split
        root /= "CIRR"
        rows = read(root / "cirr/captions" / f"cap.rc2.{split}.json")
        gallery = {k: str(root / v) for k, v in read(root / "cirr/image_splits" / f"split.rc2.{split}.json").items()}
        queries = [{"id": str(r["pairid"]), "reference": r["reference"], "target": r.get("target_hard"), "text": r["caption"].strip()} for r in rows]
    elif dataset == "fashioniq":
        if category not in ("dress", "shirt", "toptee") or split not in ("train", "val", "test"):
            raise ValueError("FashionIQ needs valid category and split")
        root /= "FashionIQ"
        rows = read(root / "captions" / f"cap.{category}.{split}.json")
        gallery = {i: str(root / "images" / f"{i}.png") for i in read(root / "image_splits" / f"split.{category}.{split}.json")}
        queries = [{"id": str(i), "reference": r["candidate"], "target": r.get("target"),
                    "text": " and ".join(c.strip() for c in r["captions"] if c.strip())} for i, r in enumerate(rows)]
    else:
        raise ValueError("Unknown dataset")
    return rows, queries, gallery
