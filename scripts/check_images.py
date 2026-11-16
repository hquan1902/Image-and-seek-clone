"""Decode dataset images and report failures; never silently discard a query."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from PIL import Image


def check(path):
    try:
        with Image.open(path) as image:
            image.convert("RGB").load()
            if min(image.size) < 1:
                return str(path), "Empty image"
    except Exception as error:
        return str(path), str(error)
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("reports/image_decode.json"))
    args = parser.parse_args()
    paths = sorted(p for p in args.data.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, result in enumerate(pool.map(check, paths), 1):
            if result:
                failures.append(result)
            if i % 10000 == 0:
                print(f"Decoded {i}/{len(paths)}; failures={len(failures)}", flush=True)
    report = {"images_checked": len(paths), "failures": failures, "method": "Pillow RGB decode/load"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(bool(failures))
