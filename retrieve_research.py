"""Reproducible reference/text diagnostic baselines, not Imagine-and-Seek reproduction."""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from src.research_fusion import dispersion_normalized_scores, reciprocal_rank_fusion_scores
from src.research_metrics import evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=["image", "text", "sum", "weighted", "dispersion", "rrf"], default="sum")
    p.add_argument("--image-weight", type=float, default=0.2)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--ensemble-checkpoints", type=Path, nargs="+")
    args = p.parse_args()
    if args.checkpoint and args.ensemble_checkpoints:
        p.error("Use either --checkpoint or --ensemble-checkpoints, not both")
    checkpoint_paths = args.ensemble_checkpoints or ([args.checkpoint] if args.checkpoint else [])
    # Only read caches generated locally by cache_research.py.
    data = torch.load(args.cache, map_location="cpu", weights_only=True)
    ids = data["gallery_ids"]
    index = {name: i for i, name in enumerate(ids)}
    image = F.normalize(data["image_features"].float(), dim=-1)
    text = F.normalize(data["text_features"].float(), dim=-1)
    queries = data["queries"]
    refs = torch.tensor([index[q["reference"]] for q in queries])
    if not 0 <= args.image_weight <= 1:
        p.error("--image-weight must be in [0,1]")
    query = (image[refs] if args.mode == "image" else text if args.mode == "text" else
             F.normalize(args.image_weight*image[refs] + (1-args.image_weight)*text, dim=-1)
             if args.mode == "weighted" else F.normalize(image[refs] + text, dim=-1))
    checkpoints = []
    if checkpoint_paths:
        from src.research_adapter import AdaptiveComposer, MultiDomainResidualComposer
        query_parts = []
        for checkpoint_path in checkpoint_paths:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            checkpoints.append(checkpoint)
            for key in ("model", "preprocess_type", "preprocess_signature"):
                if checkpoint["metadata"][key] != data["metadata"][key]:
                    raise ValueError(f"Checkpoint/cache mismatch: {key}")
            checkpoint_pretrained = Path(checkpoint["metadata"]["pretrained"])
            cache_pretrained = Path(data["metadata"]["pretrained"])
            checkpoint_pretrained = checkpoint_pretrained.resolve() if checkpoint_pretrained.exists() else checkpoint_pretrained
            cache_pretrained = cache_pretrained.resolve() if cache_pretrained.exists() else cache_pretrained
            if checkpoint_pretrained != cache_pretrained:
                raise ValueError("Checkpoint/cache mismatch: pretrained")
            if checkpoint.get("model_class") == "MultiDomainResidualComposer":
                model = MultiDomainResidualComposer(
                    checkpoint["dimension"], checkpoint["hidden"], checkpoint["dropout"],
                    checkpoint["residual_scale"])
                dataset = data["metadata"]["dataset"]
                if dataset == "circo":
                    weights = []
                    for start in range(0, len(text), 64):
                        image_scores = image[refs[start:start+64]] @ image.T
                        text_scores = text[start:start+64] @ image.T
                        image_std = image_scores.std(dim=1, unbiased=False)
                        text_std = text_scores.std(dim=1, unbiased=False)
                        weights.append(text_std/(image_std+text_std).clamp_min(1e-6))
                    base_weight = torch.cat(weights)
                    base_strategy = "per-query gallery dispersion"
                else:
                    domain = data["metadata"].get("category") if dataset == "fashioniq" else dataset
                    base_weight = checkpoint["base_weights"][domain]
                    base_strategy = f"static:{base_weight}"
                model.load_state_dict(checkpoint["state_dict"])
                model.eval()
                with torch.no_grad():
                    checkpoint_query, _ = model(image[refs], text, base_weight)
            else:
                model = AdaptiveComposer(checkpoint["dimension"], checkpoint["hidden"],
                                         checkpoint["initial_image_weight"],
                                         checkpoint["use_residual"], checkpoint["gate_input"])
                model.load_state_dict(checkpoint["state_dict"])
                model.eval()
                with torch.no_grad():
                    checkpoint_query, _ = model(image[refs], text)
            query_parts.append(checkpoint_query)
        query = F.normalize(torch.stack(query_parts).mean(dim=0), dim=-1)
    full_result = {}
    with torch.inference_mode():
        for i, q in enumerate(queries):
            if args.mode == "dispersion" and not checkpoint_paths:
                scores = dispersion_normalized_scores(image[refs[i]] @ image.T, text[i] @ image.T)
            elif args.mode == "rrf" and not checkpoint_paths:
                scores = reciprocal_rank_fusion_scores(image[refs[i]] @ image.T, text[i] @ image.T)
            else:
                scores = query[i] @ image.T
            scores[refs[i]] = -torch.inf
            ranked_indices = torch.argsort(scores, descending=True, stable=True)
            if data["metadata"]["dataset"] != "cirr":
                ranked_indices = ranked_indices[:50]
            ranking = ranked_indices.tolist()
            ranking = [ids[j] for j in ranking if j != refs[i].item()]
            full_result[q["id"]] = ranking
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # CIRR group recall needs the full ranking, but submission/inspection files only need Top-50.
    result = {key: value[:50] for key, value in full_result.items()}
    args.output.write_text(json.dumps(result), encoding="utf-8")
    if data["metadata"]["split"] == "val":
        metrics = evaluate(data["rows"], full_result if data["metadata"]["dataset"] == "cirr" else result,
                           ids, data["metadata"]["dataset"])
        report_metadata = dict(data["metadata"])
        if checkpoint_paths:
            # Cache seed describes feature extraction; checkpoint seed/epoch
            # describe the trained adapter and must be recorded separately.
            report_metadata["checkpoint_seeds"] = [item.get("seed") for item in checkpoints]
            report_metadata["checkpoint_epochs"] = [item.get("epoch") for item in checkpoints]
        report = {"method": ("checkpoint_ensemble" if len(checkpoint_paths) > 1 else "supervised_adapter")
                  if checkpoint_paths else args.mode,
                  "image_weight": args.image_weight if args.mode == "weighted" else None,
                  "checkpoint": ([str(path) for path in checkpoint_paths] if len(checkpoint_paths) > 1
                                 else str(checkpoint_paths[0])) if checkpoint_paths else None,
                  "metadata": report_metadata, "metrics_fraction": metrics}
        if checkpoint_paths and checkpoints[0].get("model_class") == "MultiDomainResidualComposer":
            report["base_strategy"] = base_strategy
        args.output.with_suffix(".metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
