"""Paired bootstrap comparison of two retrieval prediction files."""
import argparse
import json
from pathlib import Path
import numpy as np
from src.research_data import load_split
from src.research_metrics import average_precision


def interval(values):
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def paired_bootstrap(first, second, clusters, rng, samples):
    """Return paired query- and reference-cluster bootstrap intervals."""
    difference = np.asarray(second, dtype=float) - np.asarray(first, dtype=float)
    query_boot = difference[rng.integers(0, len(difference), size=(samples, len(difference)))].mean(1)
    cluster_boot = np.empty(samples)
    for sample in range(samples):
        selected = rng.integers(0, len(clusters), size=len(clusters))
        values = np.concatenate([difference[clusters[i]] for i in selected])
        cluster_boot[sample] = values.mean()
    return {
        "baseline": float(np.mean(first)), "candidate": float(np.mean(second)),
        "delta": float(difference.mean()), "query_bootstrap_95ci": interval(query_boot),
        "reference_cluster_bootstrap_95ci": interval(cluster_boot),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["cirr", "fashioniq", "circo"], required=True)
    p.add_argument("--data", default="data")
    p.add_argument("--split", default="val")
    p.add_argument("--category", choices=["dress", "shirt", "toptee"])
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--samples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    rows, queries, _ = load_split(args.data, args.dataset, args.split, args.category)
    baseline, candidate = json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text())
    ids = [q["id"] for q in queries]
    if set(baseline) != set(ids) or set(candidate) != set(ids):
        p.error("Both prediction files must exactly match annotation query IDs")
    rng = np.random.default_rng(args.seed)
    references = np.array([q["reference"] for q in queries])
    unique_references = np.unique(references)
    clusters = [np.flatnonzero(references == ref) for ref in unique_references]
    result = {"baseline": str(args.baseline), "candidate": str(args.candidate), "samples": args.samples,
              "seed": args.seed, "unit": "fraction", "metrics": {}}
    ks = (5, 10, 25, 50) if args.dataset == "circo" else ((10, 50) if args.dataset == "fashioniq" else (1, 5, 10, 50))
    for k in ks:
        first = np.array([q["target"] in baseline[q["id"]][:k] for q in queries], dtype=float)
        second = np.array([q["target"] in candidate[q["id"]][:k] for q in queries], dtype=float)
        result["metrics"][f"R@{k}"] = paired_bootstrap(first, second, clusters, rng, args.samples)
        if args.dataset == "circo":
            first_ap = [average_precision(baseline[q["id"]], row["gt_img_ids"], k)
                        for row, q in zip(rows, queries)]
            second_ap = [average_precision(candidate[q["id"]], row["gt_img_ids"], k)
                         for row, q in zip(rows, queries)]
            result["metrics"][f"mAP@{k}"] = paired_bootstrap(
                first_ap, second_ap, clusters, rng, args.samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
