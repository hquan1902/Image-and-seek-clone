"""Cache frozen CLIP embeddings with exact annotation identity and model provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
import open_clip
from torchvision.transforms import Compose, CenterCrop, InterpolationMode, Normalize, Resize, ToTensor
from torchvision.transforms import functional as TF
from src.research_data import load_split


class TargetPad:
    def __init__(self, ratio=1.25):
        self.ratio = ratio

    def __call__(self, image):
        width, height = image.size
        actual = max(width, height)/min(width, height)
        if actual < self.ratio:
            return image
        scaled = max(width, height)/self.ratio
        horizontal = max(int((scaled-width)/2), 0)
        vertical = max(int((scaled-height)/2), 0)
        return TF.pad(image, [horizontal, vertical, horizontal, vertical], 0, "constant")


def targetpad_transform(size=224):
    return Compose([TargetPad(1.25), Resize(size, interpolation=InterpolationMode.BICUBIC), CenterCrop(size),
                    lambda image: image.convert("RGB"), ToTensor(),
                    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["cirr", "fashioniq", "circo"])
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--category", choices=["dress", "shirt", "toptee"])
    p.add_argument("--data", default="data")
    p.add_argument("--model", default="ViT-B-32")
    p.add_argument("--pretrained", default="laion2b_s34b_b79k")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--preprocess-type", choices=["targetpad", "clip"], default="targetpad")
    p.add_argument("--legacy-gallery", type=Path, help="Validated torch cache with keys feature/name; CIRCO only")
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    if args.output.exists():
        p.error("Output exists; choose a new path to preserve experiment provenance")
    rows, queries, gallery = load_split(args.data, args.dataset, args.split, args.category)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    precision = "fp16" if device == "cuda" else "fp32"
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained, device=device, precision=precision)
    if args.preprocess_type == "targetpad":
        preprocess = targetpad_transform(224)
        preprocess_signature = "targetpad-r1.25-resize224-center224-clip-meanstd-v1"
    else:
        preprocess_signature = f"open-clip-default:{args.model}:{args.pretrained}"
    model.eval().requires_grad_(False)
    tokenizer = open_clip.get_tokenizer(args.model)
    ids = list(gallery)
    images = []
    texts = []
    with torch.inference_mode():
        if args.legacy_gallery:
            if args.dataset != "circo" or args.preprocess_type != "targetpad":
                p.error("--legacy-gallery is restricted to CIRCO targetpad caches")
            legacy = torch.load(args.legacy_gallery, map_location="cpu", weights_only=True)
            legacy_names = list(map(str, legacy["name"]))
            if len(legacy_names) != len(set(legacy_names)) or set(legacy_names) != set(ids):
                p.error("Legacy gallery IDs do not exactly match the CIRCO gallery")
            positions = {name: i for i, name in enumerate(legacy_names)}
            images.append(F.normalize(legacy["feature"][[positions[name] for name in ids]].float(), dim=-1))
        else:
            for start in range(0, len(ids), args.batch_size):
                batch = []
                for name in ids[start:start + args.batch_size]:
                    with Image.open(gallery[name]) as image:
                        batch.append(preprocess(image.convert("RGB")))
                with torch.autocast(device_type=device, enabled=device == "cuda"):
                    encoded = model.encode_image(torch.stack(batch).to(device))
                images.append(F.normalize(encoded.float(), dim=-1).cpu())
                if start % (args.batch_size * 50) == 0:
                    print(f"Images {start}/{len(ids)}", flush=True)
        for start in range(0, len(queries), args.batch_size):
            tokens = tokenizer([q["text"] for q in queries[start:start + args.batch_size]]).to(device)
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                encoded = model.encode_text(tokens)
            texts.append(F.normalize(encoded.float(), dim=-1).cpu())
    metadata = {"dataset": args.dataset, "split": args.split, "category": args.category,
                "model": args.model, "pretrained": args.pretrained, "preprocess_type": args.preprocess_type,
                "preprocess_signature": preprocess_signature,
                "preprocess": str(preprocess),
                "torch": str(torch.__version__), "annotation_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
                "image_dtype": "float32", "model_precision": precision, "normalization": "L2", "seed": 42,
                "legacy_gallery": str(args.legacy_gallery) if args.legacy_gallery else None,
                "legacy_gallery_sha256": file_sha256(args.legacy_gallery) if args.legacy_gallery else None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image_features, text_features = torch.cat(images), torch.cat(texts)
    if image_features.shape[0] != len(ids) or text_features.shape[0] != len(queries):
        raise RuntimeError("Cache/query/gallery alignment mismatch")
    for features in (image_features, text_features):
        if not torch.isfinite(features).all() or not torch.allclose(features.norm(dim=-1), torch.ones(len(features)), atol=1e-4):
            raise RuntimeError("Embedding cache is not finite and L2 normalized")
    torch.save({"metadata": metadata, "rows": rows, "queries": queries, "gallery_ids": ids,
                "image_features": image_features, "text_features": text_features}, args.output)
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
