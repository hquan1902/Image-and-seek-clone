"""Cache all FashionIQ train/val categories with one model load and one image pass."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
import open_clip
from cache_research import targetpad_transform
from src.research_data import load_split


CATEGORIES = ("dress", "shirt", "toptee")
SPLITS = ("train", "val")


def output_path(root, category, split):
    return root / f"fashioniq_{category}_{split}_g14_targetpad.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--output-dir", type=Path, default=Path("features"))
    parser.add_argument("--model", default="ViT-g-14")
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    domains = {}
    all_images = {}
    for category in CATEGORIES:
        for split in SPLITS:
            rows, queries, gallery = load_split(args.data, "fashioniq", split, category)
            domains[(category, split)] = (rows, queries, gallery)
            for image_id, path in gallery.items():
                previous = all_images.setdefault(image_id, path)
                if Path(previous) != Path(path):
                    raise ValueError(f"Conflicting path for FashionIQ image {image_id}")

    outputs = [output_path(args.output_dir, category, split)
               for category in CATEGORIES for split in SPLITS]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        parser.error("Refusing to overwrite existing caches: " + ", ".join(existing))
    summary = {
        f"{category}_{split}": {"queries": len(domains[(category, split)][1]),
                                "gallery": len(domains[(category, split)][2])}
        for category in CATEGORIES for split in SPLITS
    }
    print(json.dumps({"unique_images": len(all_images), "domains": summary}, indent=2), flush=True)
    if args.dry_run:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = "fp16" if device == "cuda" else "fp32"
    torch.manual_seed(42)
    model, _, _ = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, device=device, precision=precision)
    model.eval().requires_grad_(False)
    tokenizer = open_clip.get_tokenizer(args.model)
    preprocess = targetpad_transform(224)

    union_ids = list(all_images)
    union_features = []
    with torch.inference_mode():
        for start in range(0, len(union_ids), args.batch_size):
            batch = []
            for image_id in union_ids[start:start + args.batch_size]:
                with Image.open(all_images[image_id]) as image:
                    batch.append(preprocess(image.convert("RGB")))
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                encoded = model.encode_image(torch.stack(batch).to(device))
            union_features.append(F.normalize(encoded.float(), dim=-1).cpu())
            if start == 0 or (start // args.batch_size) % 50 == 0:
                print(f"FashionIQ images {min(start+len(batch), len(union_ids))}/{len(union_ids)}", flush=True)
    union_features = torch.cat(union_features)
    position = {image_id: i for i, image_id in enumerate(union_ids)}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for category in CATEGORIES:
        for split in SPLITS:
            rows, queries, gallery = domains[(category, split)]
            gallery_ids = list(gallery)
            image_features = union_features[[position[image_id] for image_id in gallery_ids]]
            text_parts = []
            with torch.inference_mode():
                for start in range(0, len(queries), args.batch_size):
                    tokens = tokenizer([q["text"] for q in queries[start:start + args.batch_size]]).to(device)
                    with torch.autocast(device_type=device, enabled=device == "cuda"):
                        encoded = model.encode_text(tokens)
                    text_parts.append(F.normalize(encoded.float(), dim=-1).cpu())
            text_features = torch.cat(text_parts)
            metadata = {
                "dataset": "fashioniq", "split": split, "category": category,
                "model": args.model, "pretrained": args.pretrained,
                "preprocess_type": "targetpad",
                "preprocess_signature": "targetpad-r1.25-resize224-center224-clip-meanstd-v1",
                "preprocess": str(preprocess), "torch": str(torch.__version__),
                "annotation_sha256": hashlib.sha256(
                    json.dumps(rows, sort_keys=True).encode()).hexdigest(),
                "image_dtype": "float32", "model_precision": precision,
                "normalization": "L2", "seed": 42,
            }
            destination = output_path(args.output_dir, category, split)
            torch.save({"metadata": metadata, "rows": rows, "queries": queries,
                        "gallery_ids": gallery_ids, "image_features": image_features,
                        "text_features": text_features}, destination)
            destination.with_suffix(".metadata.json").write_text(
                json.dumps(metadata, indent=2), encoding="utf-8")
            print(f"Saved {destination}: images={len(gallery_ids)}, queries={len(queries)}", flush=True)


if __name__ == "__main__":
    main()
