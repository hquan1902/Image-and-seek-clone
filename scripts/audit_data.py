"""Audit annotations and image paths without importing the legacy ML stack."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit(root):
    report = {"datasets": {}, "errors": [], "warnings": [], "sha256": {}}
    errors = report["errors"]
    def load(path):
        report["sha256"][str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return read(path)
    circo = root / "CIRCO"
    gallery = load(circo / "COCO2017_unlabeled/annotations/image_info_unlabeled2017.json")
    paths = {str(x["id"]): circo / "COCO2017_unlabeled/unlabeled2017" / x["file_name"] for x in gallery["images"]}
    missing = [k for k, p in paths.items() if not p.is_file()]
    if missing:
        errors.append({"circo_missing_gallery": missing})
    for split, expected in (("val", 220), ("test", 800)):
        rows = load(circo / "annotations" / (split + ".json"))
        if len(rows) != expected or {x["id"] for x in rows} != set(range(expected)):
            errors.append(f"CIRCO {split}: unexpected query IDs/count")
        for row in rows:
            ids = [row["reference_img_id"]]
            if split == "val":
                gt = row["gt_img_ids"]
                if not gt or len(gt) != len(set(gt)) or row["target_img_id"] not in gt:
                    errors.append(f"Invalid CIRCO positives: {row['id']}")
                ids += gt + [row["target_img_id"]]
            if any(str(i) not in paths for i in ids) or not row["relative_caption"].strip():
                errors.append(f"Invalid CIRCO query: {split}/{row['id']}")
        report["datasets"]["circo_" + split] = {"queries": len(rows), "gallery": len(paths), "missing_gallery": len(missing)}
        by_reference = defaultdict(list)
        for row in rows:
            by_reference[row["reference_img_id"]].append(row)
        conflicts = [{"reference": ref, "query_ids": [r["id"] for r in group]} for ref, group in by_reference.items() if len({r["relative_caption"] for r in group}) > 1]
        report["datasets"]["circo_" + split]["proxy_key_conflicts"] = conflicts
        if conflicts:
            report["warnings"].append(f"CIRCO {split}: proxy filenames keyed only by reference ID are ambiguous for {len(conflicts)} references")
    cirr = root / "CIRR"
    split_ids = {}
    for split in ("train", "val", "test1"):
        mapping = load(cirr / "cirr/image_splits" / f"split.rc2.{split}.json")
        rows = load(cirr / "cirr/captions" / f"cap.rc2.{split}.json")
        split_ids[split] = set(mapping)
        missing = [k for k, p in mapping.items() if not (cirr / p).is_file()]
        if missing:
            errors.append({"cirr_missing": missing})
        for row in rows:
            needed = [row["reference"]] + row["img_set"]["members"]
            if split != "test1":
                needed.append(row["target_hard"])
                if row["target_hard"] == row["reference"] or row["target_hard"] not in row["img_set"]["members"]:
                    errors.append(f"Invalid CIRR target: {split}/{row['pairid']}")
            if any(i not in mapping for i in needed):
                errors.append(f"Invalid CIRR query: {split}/{row['pairid']}")
            if not row["caption"].strip():
                report["warnings"].append(f"Blank CIRR caption retained: {split}/{row['pairid']}")
        report["datasets"]["cirr_" + split] = {"queries": len(rows), "gallery": len(mapping), "missing": len(missing)}
    if split_ids["train"] & (split_ids["val"] | split_ids["test1"]):
        errors.append("CIRR image ID leakage across splits")
    fiq = root / "FashionIQ"
    all_fiq_ids = {split: set() for split in ("train", "val", "test")}
    for category in ("dress", "shirt", "toptee"):
        ids_by_split = {}
        for split in ("train", "val", "test"):
            ids = load(fiq / "image_splits" / f"split.{category}.{split}.json")
            rows = load(fiq / "captions" / f"cap.{category}.{split}.json")
            ids_by_split[split] = set(ids)
            all_fiq_ids[split].update(ids)
            missing = [i for i in ids if not (fiq / "images" / (i + ".png")).is_file()]
            if missing:
                errors.append({f"fiq_missing_{category}_{split}": missing})
            for query_index, row in enumerate(rows):
                if row["candidate"] not in ids_by_split[split] or ("target" in row and row["target"] not in ids_by_split[split]):
                    errors.append(f"FashionIQ query outside gallery: {category}/{split}")
                if not row["captions"] or not any(c.strip() for c in row["captions"]):
                    errors.append(f"Empty FashionIQ caption: {category}/{split}")
                elif not all(c.strip() for c in row["captions"]):
                    report["warnings"].append(f"Partial empty FashionIQ caption: {category}/{split}/{query_index}")
            report["datasets"][f"fiq_{category}_{split}"] = {"queries": len(rows), "gallery": len(ids), "missing": len(missing)}
        if ids_by_split["train"] & (ids_by_split["val"] | ids_by_split["test"]):
            errors.append(f"FashionIQ {category}: image ID leakage")
    report["fashioniq_cross_category_overlap"] = {f"{a}_{b}": len(all_fiq_ids[a] & all_fiq_ids[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    if report["fashioniq_cross_category_overlap"]["train_val"]:
        report["warnings"].append("FashionIQ cross-category train/val overlap: do not combine categories without an explicit leakage policy")
    report["limitations"] = ["Checks file existence, not image decoding or semantic label correctness.", "Split checks use image IDs, not perceptual duplicate detection."]
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("reports/data_audit.json"))
    args = parser.parse_args()
    result = audit(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result["datasets"], indent=2))
    print("Errors:", len(result["errors"]), "Report:", args.output)
    raise SystemExit(bool(result["errors"]))
