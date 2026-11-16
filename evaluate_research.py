"""Strict validation evaluator. FashionIQ query IDs are zero-based annotation row indices."""
import argparse
import hashlib
import json
from pathlib import Path
from src.research_metrics import evaluate


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["circo", "cirr", "fashioniq"], required=True)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--category", choices=["dress", "shirt", "toptee"])
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.dataset == "circo":
        annotation = args.data / "CIRCO/annotations/val.json"
        gallery = [x["id"] for x in read(args.data / "CIRCO/COCO2017_unlabeled/annotations/image_info_unlabeled2017.json")["images"]]
    elif args.dataset == "cirr":
        annotation = args.data / "CIRR/cirr/captions/cap.rc2.val.json"
        gallery = read(args.data / "CIRR/cirr/image_splits/split.rc2.val.json")
    else:
        if args.category is None:
            parser.error("FashionIQ needs --category; evaluate all three and macro-average separately")
        annotation = args.data / "FashionIQ/captions" / f"cap.{args.category}.val.json"
        gallery = read(args.data / "FashionIQ/image_splits" / f"split.{args.category}.val.json")
    metrics = evaluate(read(annotation), read(args.predictions), gallery, args.dataset)
    result = {"dataset": args.dataset, "split": "val", "category": args.category,
              "units": "fraction", "metrics": metrics,
              "annotation_sha256": hashlib.sha256(annotation.read_bytes()).hexdigest(),
              "prediction_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
