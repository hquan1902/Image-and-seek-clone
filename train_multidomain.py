"""Train a leakage-controlled residual composer on CIRR and FashionIQ."""
import argparse
import json
import math
import random
import time
from pathlib import Path
import torch
from src.research_adapter import (MultiDomainResidualComposer, full_gallery_cross_entropy,
                                  hard_negative_margin_loss, group_cross_entropy)


DOMAIN_SPECS = {
    "cirr": ("cirr_train_g14_targetpad.pt", "cirr_val_g14_targetpad.pt", 0.2, 1),
    "dress": ("fashioniq_dress_train_g14_targetpad.pt", "fashioniq_dress_val_g14_targetpad.pt", 0.2, 10),
    "shirt": ("fashioniq_shirt_train_g14_targetpad.pt", "fashioniq_shirt_val_g14_targetpad.pt", 0.3, 10),
    "toptee": ("fashioniq_toptee_train_g14_targetpad.pt", "fashioniq_toptee_val_g14_targetpad.pt", 0.3, 10),
}


def load_cache(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def make_domain(cache, device, forbidden_ids=None):
    gallery_ids = list(cache["gallery_ids"])
    keep_gallery = [i for i, image_id in enumerate(gallery_ids)
                    if forbidden_ids is None or image_id not in forbidden_ids]
    kept_ids = [gallery_ids[i] for i in keep_gallery]
    position = {image_id: i for i, image_id in enumerate(kept_ids)}
    keep_queries = [i for i, query in enumerate(cache["queries"])
                    if query["target"] in position and query["reference"] in position]
    if not keep_queries:
        raise ValueError("Leakage filtering removed every query")
    references = torch.tensor([position[cache["queries"][i]["reference"]]
                               for i in keep_queries], device=device)
    targets = torch.tensor([position[cache["queries"][i]["target"]]
                            for i in keep_queries], device=device)
    group_indices = None
    rows = cache.get("rows")
    if rows and len(rows) == len(cache["queries"]) and "img_set" in rows[0]:
        groups = []
        for query_index in keep_queries:
            reference = cache["queries"][query_index]["reference"]
            members = [position[image_id] for image_id in rows[query_index]["img_set"]["members"]
                       if image_id != reference and image_id in position]
            groups.append(members)
        widths = {len(group) for group in groups}
        if len(widths) != 1:
            raise ValueError(f"CIRR groups have inconsistent usable sizes: {widths}")
        group_indices = torch.tensor(groups, device=device)
    return {
        "image": cache["image_features"][keep_gallery].float().to(device),
        "text": cache["text_features"][keep_queries].float().to(device),
        "references": references, "targets": targets,
        "queries_before": len(cache["queries"]), "gallery_before": len(gallery_ids),
        "queries_after": len(keep_queries), "gallery_after": len(kept_ids),
        "group_indices": group_indices,
    }


@torch.no_grad()
def validate(model, domain, image_weight, cutoff, batch_size=64):
    model.eval()
    hits = 0
    for start in range(0, len(domain["text"]), batch_size):
        refs = domain["references"][start:start+batch_size]
        target = domain["targets"][start:start+batch_size]
        query, _ = model(domain["image"][refs], domain["text"][start:start+batch_size], image_weight)
        scores = query @ domain["image"].T
        scores.scatter_(1, refs[:, None], -torch.inf)
        ranked = torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :cutoff]
        hits += (ranked == target[:, None]).any(-1).sum().item()
    return hits / len(domain["text"])


@torch.no_grad()
def validate_cirr_group(model, domain, image_weight, batch_size=64):
    """Target R@1 among the five non-reference members of each CIRR group."""
    if domain["group_indices"] is None:
        raise ValueError("CIRR group metadata is unavailable")
    model.eval()
    hits = 0
    for start in range(0, len(domain["text"]), batch_size):
        stop = start+batch_size
        refs = domain["references"][start:stop]
        targets = domain["targets"][start:stop]
        groups = domain["group_indices"][start:stop]
        query, _ = model(domain["image"][refs], domain["text"][start:stop], image_weight)
        candidates = domain["image"][groups]
        scores = torch.einsum("bd,bgd->bg", query, candidates)
        predicted = groups.gather(1, scores.argmax(dim=1, keepdim=True)).squeeze(1)
        hits += (predicted == targets).sum().item()
    return hits / len(domain["text"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, default=Path("features"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--samples-per-domain", type=int, default=6000)
    parser.add_argument("--cirr-samples", type=int, default=18000,
                        help="CIRR examples per epoch; larger than each FIQ domain to reduce domain conflict")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--residual-scale", type=float, default=0.1)
    parser.add_argument("--residual-penalty", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--hard-negative-k", type=int, default=32)
    parser.add_argument("--hard-negative-weight", type=float, default=0.5)
    parser.add_argument("--hard-negative-margin", type=float, default=0.05)
    parser.add_argument("--cirr-group-loss-weight", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory exists; choose a new run directory")
    if min(args.epochs, args.batch_size, args.samples_per_domain, args.cirr_samples,
           args.hidden, args.patience, args.hard_negative_k) < 1:
        parser.error("Epoch, batch, sample, hidden and patience values must be positive")
    if not 0 <= args.dropout < 1 or args.residual_scale <= 0 or args.residual_penalty < 0:
        parser.error("Invalid residual model configuration")
    if (args.hard_negative_weight < 0 or args.hard_negative_margin < 0 or
            args.cirr_group_loss_weight < 0):
        parser.error("Hard-negative weight and margin must be nonnegative")

    paths = {}
    raw_train, raw_val = {}, {}
    for name, (train_file, val_file, _, _) in DOMAIN_SPECS.items():
        paths[name] = (args.feature_dir/train_file, args.feature_dir/val_file)
        if not all(path.exists() for path in paths[name]):
            parser.error(f"Missing cache for {name}: {paths[name]}")
        raw_train[name], raw_val[name] = map(load_cache, paths[name])

    reference_metadata = raw_train["cirr"]["metadata"]
    dimension = raw_train["cirr"]["image_features"].shape[1]
    for name, cache in {**raw_train, **{f"{k}_val": v for k, v in raw_val.items()}}.items():
        for key in ("model", "preprocess_signature"):
            if cache["metadata"][key] != reference_metadata[key]:
                parser.error(f"Cache mismatch for {name}: {key}")
        reference_pretrained = Path(reference_metadata["pretrained"])
        cache_pretrained = Path(cache["metadata"]["pretrained"])
        if (reference_pretrained.resolve() if reference_pretrained.exists() else reference_pretrained) != \
                (cache_pretrained.resolve() if cache_pretrained.exists() else cache_pretrained):
            parser.error(f"Cache mismatch for {name}: pretrained")
        if cache["image_features"].shape[1] != dimension:
            parser.error(f"Embedding dimension mismatch for {name}")

    # Cross-category FashionIQ IDs are removed from every FashionIQ training
    # gallery, not merely from query endpoints, to prevent validation exposure.
    fiq_val_ids = set()
    for name in ("dress", "shirt", "toptee"):
        fiq_val_ids.update(raw_val[name]["gallery_ids"])

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_domains, val_domains = {}, {}
    for name in DOMAIN_SPECS:
        forbidden = fiq_val_ids if name != "cirr" else None
        train_domains[name] = make_domain(raw_train[name], device, forbidden)
        val_domains[name] = make_domain(raw_val[name], device)
        td = train_domains[name]
        print(f"{name}: train queries {td['queries_after']}/{td['queries_before']}, "
              f"gallery {td['gallery_after']}/{td['gallery_before']}", flush=True)
    del raw_train, raw_val

    model = MultiDomainResidualComposer(
        dimension, args.hidden, args.dropout, args.residual_scale).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    args.output.mkdir(parents=True)
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config["device"] = device
    config["domains"] = DOMAIN_SPECS
    (args.output/"config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    def validation_metrics():
        metrics = {name: validate(model, val_domains[name], DOMAIN_SPECS[name][2], DOMAIN_SPECS[name][3])
                   for name in DOMAIN_SPECS}
        metrics["cirr_group"] = validate_cirr_group(
            model, val_domains["cirr"], DOMAIN_SPECS["cirr"][2])
        return metrics

    baseline = validation_metrics()
    best_score = 1.0
    best_metrics = baseline.copy()
    best_composite_score = 1.0
    best_composite_metrics = baseline.copy()
    history = [{"epoch": 0, "validation": baseline, "selection_score": best_score,
                "label": "zero-residual strong static baselines"}]

    def save(path, epoch, score, metrics, criterion):
        torch.save({
            "state_dict": model.state_dict(), "model_class": "MultiDomainResidualComposer",
            "dimension": dimension, "hidden": args.hidden, "dropout": args.dropout,
            "residual_scale": args.residual_scale, "base_weights": {
                name: spec[2] for name, spec in DOMAIN_SPECS.items()},
            "epoch": epoch, "selection_score": score, "selection_criterion": criterion,
            "validation": metrics, "seed": args.seed,
            "metadata": {key: reference_metadata[key]
                         for key in ("model", "pretrained", "preprocess_type", "preprocess_signature")},
        }, args.output/path)

    save("best.pt", 0, best_score, best_metrics, "balanced")
    save("best_composite.pt", 0, best_composite_score, best_composite_metrics,
         "unrestricted_mean_relative_score")
    (args.output/"history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print("[epoch 000] " + " ".join(f"{k}={100*v:.2f}%" for k, v in baseline.items())
          + " selection=1.0000 checkpoint=best.pt", flush=True)

    stale = 0
    for epoch in range(1, args.epochs+1):
        started = time.perf_counter()
        model.train()
        total_loss = total_examples = 0
        domain_sums = {name: 0.0 for name in DOMAIN_SPECS}
        domain_examples = {name: 0 for name in DOMAIN_SPECS}
        epoch_batches = []
        for name in DOMAIN_SPECS:
            domain = train_domains[name]
            requested = args.cirr_samples if name == "cirr" else args.samples_per_domain
            count = min(requested, len(domain["text"]))
            order = torch.randperm(len(domain["text"]), device=device)[:count]
            epoch_batches.extend((name, indices) for indices in order.split(args.batch_size))
        # Interleave domains instead of training four long domain blocks. This
        # reduces within-epoch catastrophic forgetting while preserving counts.
        random.shuffle(epoch_batches)
        for name, indices in epoch_batches:
            domain = train_domains[name]
            refs = domain["references"][indices]
            targets = domain["targets"][indices]
            query, residual = model(
                domain["image"][refs], domain["text"][indices], DOMAIN_SPECS[name][2])
            loss = full_gallery_cross_entropy(
                query, domain["image"], targets, refs,
                args.temperature, args.label_smoothing)
            if args.hard_negative_weight:
                hard_loss = hard_negative_margin_loss(
                    query, domain["image"], targets, refs,
                    args.hard_negative_k, args.hard_negative_margin, args.temperature)
                loss = loss + args.hard_negative_weight*hard_loss
            if name == "cirr" and args.cirr_group_loss_weight:
                group_loss = group_cross_entropy(
                    query, domain["image"], domain["group_indices"][indices], targets,
                    args.temperature)
                loss = loss + args.cirr_group_loss_weight*group_loss
            if args.residual_penalty:
                loss = loss + args.residual_penalty*residual.square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss in domain {name}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            weighted_loss = loss.item()*len(indices)
            domain_sums[name] += weighted_loss
            domain_examples[name] += len(indices)
            total_loss += weighted_loss
            total_examples += len(indices)
        domain_losses = {name: domain_sums[name]/domain_examples[name] for name in DOMAIN_SPECS}

        metrics = validation_metrics()
        ratios = {name: metrics[name]/baseline[name] for name in DOMAIN_SPECS}
        selection = sum(ratios.values())/len(ratios)
        non_degrading = sum(metrics[name] >= baseline[name] for name in DOMAIN_SPECS)
        group_eligible = (not args.cirr_group_loss_weight or
                          metrics["cirr_group"] >= baseline["cirr_group"])
        eligible = metrics["cirr"] >= baseline["cirr"] and non_degrading >= 3 and group_eligible
        improved = eligible and selection > best_score
        composite_improved = selection > best_composite_score
        if improved:
            best_score, best_metrics = selection, metrics.copy()
            save("best.pt", epoch, best_score, best_metrics, "balanced")
        if composite_improved:
            best_composite_score, best_composite_metrics = selection, metrics.copy()
            save("best_composite.pt", epoch, best_composite_score, best_composite_metrics,
                 "unrestricted_mean_relative_score")
        stale = 0 if improved or composite_improved else stale+1
        entry = {
            "epoch": epoch, "loss": total_loss/total_examples,
            "domain_losses": domain_losses, "validation": metrics,
            "relative_to_baseline": ratios, "selection_score": selection,
            "eligible": eligible, "best_score": best_score,
            "best_composite_score": best_composite_score,
            "checkpoint_updated": improved,
            "composite_checkpoint_updated": composite_improved,
            "seconds": time.perf_counter()-started,
        }
        history.append(entry)
        (args.output/"history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        metric_text = " ".join(f"{name}={100*metrics[name]:.2f}%" for name in DOMAIN_SPECS)
        metric_text += f" cirr_group={100*metrics['cirr_group']:.2f}%"
        print(f"[epoch {epoch:03d}/{args.epochs:03d}] loss={entry['loss']:.4f} {metric_text} "
              f"selection={selection:.4f} best={best_score:.4f} "
              f"composite_best={best_composite_score:.4f} time={entry['seconds']:.1f}s "
              f"checkpoint={'balanced' if improved else '-'} "
              f"composite_checkpoint={'updated' if composite_improved else '-'}", flush=True)
        if stale >= args.patience:
            print(f"Early stopping: neither balanced nor unrestricted composite improved "
                  f"for {stale} epochs", flush=True)
            break


if __name__ == "__main__":
    main()
