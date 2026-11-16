"""Summarize the three final group-aware runs for the scientific report."""
import argparse
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SEEDS = (42, 7, 123)
CATEGORIES = ("dress", "shirt", "toptee")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def metric_file(run, dataset):
    name = f"fashioniq_{dataset}_val.metrics.json" if dataset in CATEGORIES else f"{dataset}_val.metrics.json"
    return run / name


def aggregate(values):
    return {
        "values_fraction": values,
        "mean_fraction": statistics.mean(values),
        "sample_std_fraction": statistics.stdev(values),
        "mean_percent": 100*statistics.mean(values),
        "sample_std_percent": 100*statistics.stdev(values),
        "min_percent": 100*min(values),
        "max_percent": 100*max(values),
    }


def add_panel(ax, labels, baseline, means, stds, title):
    x = np.arange(len(labels))
    width = 0.36
    ax.bar(x-width/2, baseline, width, label="Baseline", color="#9aa0a6")
    bars = ax.bar(x+width/2, means, width, yerr=stds, capsize=4,
                  label="Group-aware (mean ± SD)", color="#3268a8")
    ax.set_title(title)
    ax.set_ylabel("Percent (%)")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, means):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f"{value:.2f}", ha="center", va="bottom", fontsize=8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/core_results"))
    args = parser.parse_args()

    runs = {seed: args.runs_dir/f"student_multidomain_group2_seed{seed}" for seed in SEEDS}
    required = [metric_file(run, dataset) for run in runs.values()
                for dataset in ("cirr", "circo", *CATEGORIES)]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        parser.error("Missing metric files: " + ", ".join(missing))

    per_seed = {}
    histories = {}
    metric_values = {}
    for seed, run in runs.items():
        history = read_json(run/"history.json")
        histories[seed] = history
        updated = [entry for entry in history if entry.get("checkpoint_updated")]
        best = max(updated, key=lambda entry: entry["selection_score"])
        per_seed[str(seed)] = {
            "best_epoch": best["epoch"],
            "selection_score": best["selection_score"],
            "metrics_fraction": {},
        }
        for dataset in ("cirr", "circo", *CATEGORIES):
            metrics = read_json(metric_file(run, dataset))["metrics_fraction"]
            per_seed[str(seed)]["metrics_fraction"][dataset] = metrics
            for name, value in metrics.items():
                metric_values.setdefault(dataset, {}).setdefault(name, []).append(value)

    baselines = {
        "cirr": read_json(args.runs_dir/"student_multidomain_hardneg_seed42"/
                          "cirr_static.metrics.json")["metrics_fraction"],
        "circo": read_json(args.runs_dir/"core_results"/
                           "circo_val_g14_targetpad_dispersion.metrics.json")["metrics_fraction"],
    }
    for category in CATEGORIES:
        baselines[category] = read_json(args.runs_dir/"student_multidomain_hardneg_seed42"/
                                       f"fashioniq_{category}_static.metrics.json")["metrics_fraction"]

    summary = {
        "method": "MultiDomainResidualComposer + hard negatives + CIRR group loss (weight 2.0)",
        "seeds": list(SEEDS),
        "standard_deviation": "sample SD (n-1 denominator)",
        "per_seed": per_seed,
        "aggregate": {},
        "baselines_fraction": baselines,
    }
    for dataset, metrics in metric_values.items():
        summary["aggregate"][dataset] = {}
        for name, values in metrics.items():
            item = aggregate(values)
            if name in baselines[dataset]:
                item["baseline_percent"] = 100*baselines[dataset][name]
                item["mean_delta_points"] = item["mean_percent"]-item["baseline_percent"]
            summary["aggregate"][dataset][name] = item

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_dir/"group2_three_seed_summary.json"
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    panels = [
        ("cirr", ("R@1", "R@5", "R@10", "R@50", "group_R@1"), "CIRR validation"),
        ("fashioniq", ("dress:R@10", "shirt:R@10", "toptee:R@10",
                       "dress:R@50", "shirt:R@50", "toptee:R@50"), "FashionIQ validation"),
        ("circo", ("R@10", "R@25", "R@50", "mAP@10", "mAP@25", "mAP@50"),
         "CIRCO transfer validation"),
    ]
    for ax, (dataset, labels, title) in zip(axes, panels):
        if dataset == "fashioniq":
            keys = [label.split(":") for label in labels]
            baseline = [100*baselines[category][metric] for category, metric in keys]
            means = [summary["aggregate"][category][metric]["mean_percent"] for category, metric in keys]
            stds = [summary["aggregate"][category][metric]["sample_std_percent"] for category, metric in keys]
        else:
            baseline = [100*baselines[dataset][metric] for metric in labels]
            means = [summary["aggregate"][dataset][metric]["mean_percent"] for metric in labels]
            stds = [summary["aggregate"][dataset][metric]["sample_std_percent"] for metric in labels]
        add_panel(ax, labels, baseline, means, stds, title)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Three-seed robustness of the final group-aware residual composer", fontsize=14)
    fig.savefig(args.output_dir/"group2_three_seed_summary.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    for seed in SEEDS:
        trained = [entry for entry in histories[seed] if entry["epoch"] > 0]
        epochs = [entry["epoch"] for entry in trained]
        axes[0].plot(epochs, [entry["loss"] for entry in trained], label=f"seed {seed}")
        axes[1].plot(epochs, [entry["selection_score"] for entry in trained], label=f"seed {seed}")
        axes[2].plot(epochs, [100*entry["validation"]["cirr"] for entry in trained],
                     label=f"global, seed {seed}")
        axes[2].plot(epochs, [100*entry["validation"]["cirr_group"] for entry in trained],
                     linestyle="--", label=f"group, seed {seed}")
    axes[0].set_title("Training loss")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("Multi-domain selection score")
    axes[1].axhline(1.0, color="black", linewidth=1, linestyle=":", label="baseline")
    axes[1].set_ylabel("Relative score")
    axes[2].set_title("CIRR validation")
    axes[2].set_ylabel("Recall (%)")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle("Learning curves across three random seeds", fontsize=14)
    learning_curve = args.output_dir/"group2_three_seed_learning_curves.png"
    fig.savefig(learning_curve, dpi=180)
    plt.close(fig)
    print(json.dumps({"json": str(output_json),
                      "figure": str(args.output_dir/"group2_three_seed_summary.png"),
                      "learning_curves": str(learning_curve)}, indent=2))


if __name__ == "__main__":
    main()
