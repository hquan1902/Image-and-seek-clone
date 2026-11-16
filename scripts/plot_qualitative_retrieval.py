"""Create deterministic CIRR success/failure examples for the report."""
import argparse
import json
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.research_data import load_split


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rank(predictions, target):
    try:
        return predictions.index(target)+1
    except ValueError:
        return None


def select_spread(items, count):
    if len(items) < count:
        raise ValueError(f"Need {count} cases, found {len(items)}")
    if count == 1:
        return [items[len(items)//2]]
    return [items[round(i*(len(items)-1)/(count-1))] for i in range(count)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--successes", type=int, default=2)
    parser.add_argument("--failures", type=int, default=2)
    args = parser.parse_args()

    _, queries, gallery = load_split(args.data, "cirr", "val")
    baseline, candidate = read(args.baseline), read(args.candidate)
    successes, failures = [], []
    for query in queries:
        qid, target = query["id"], query["target"]
        base_rank = rank(baseline[qid], target)
        candidate_rank = rank(candidate[qid], target)
        case = {**query, "baseline_rank": base_rank, "candidate_rank": candidate_rank,
                "candidate_top3": candidate[qid][:3]}
        if candidate_rank is not None and candidate_rank <= 5 and (base_rank is None or base_rank > 10):
            successes.append(case)
        if candidate_rank is None and base_rank is None:
            failures.append(case)
    chosen = [("Improved", case) for case in select_spread(successes, args.successes)]
    chosen += [("Failure", case) for case in select_spread(failures, args.failures)]

    columns = 5
    fig, axes = plt.subplots(len(chosen), columns, figsize=(15, 3.4*len(chosen)),
                             constrained_layout=True)
    for row_index, (kind, case) in enumerate(chosen):
        image_ids = [case["reference"], case["target"], *case["candidate_top3"]]
        titles = ["Reference", "Ground truth", "Ensemble Top-1", "Ensemble Top-2", "Ensemble Top-3"]
        base = case["baseline_rank"] if case["baseline_rank"] is not None else ">50"
        cand = case["candidate_rank"] if case["candidate_rank"] is not None else ">50"
        titles[0] = (f"{kind} — query {case['id']}\nBaseline rank {base}; ensemble rank {cand}\n"
                     + textwrap.fill(case["text"], width=42))
        for column, (image_id, title) in enumerate(zip(image_ids, titles)):
            ax = axes[row_index, column]
            with Image.open(gallery[image_id]) as image:
                ax.imshow(image.convert("RGB"))
            suffix = ""
            if column >= 2 and image_id == case["target"]:
                suffix = " ✓ target"
            ax.set_title(title+suffix, fontsize=8 if column == 0 else 10)
            ax.axis("off")

    fig.suptitle("CIRR qualitative retrieval: three-seed ensemble", fontsize=15)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    metadata = {"selection_rule": {
        "improved": "ensemble target rank <=5 and static baseline rank >10",
        "failure": "target absent from Top-50 for both methods",
    }, "cases": [{"kind": kind, **case} for kind, case in chosen]}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"figure": str(args.output),
                      "metadata": str(args.output.with_suffix('.json')),
                      "eligible_successes": len(successes),
                      "eligible_failures": len(failures)}, indent=2))


if __name__ == "__main__":
    main()
