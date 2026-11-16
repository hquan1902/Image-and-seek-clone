"""Pure Python retrieval metrics; scores are fractions in [0, 1]."""


def average_precision(ranking, positives, k):
    positives = set(map(str, positives))
    ranking = list(map(str, ranking))
    if k <= 0 or not positives or len(set(ranking)) != len(ranking):
        raise ValueError("Need positive k, nonempty ground truth and unique predictions")
    hits = 0
    score = 0.0
    for rank, item in enumerate(ranking[:k], 1):
        if item in positives:
            hits += 1
            score += hits / rank
    return score / min(k, len(positives))


def evaluate(rows, predictions, gallery, dataset, ks=None):
    if dataset not in ("circo", "cirr", "fashioniq"):
        raise ValueError("Unknown dataset")
    ks = tuple(ks or ((5, 10, 25, 50) if dataset == "circo" else (1, 5, 10, 50) if dataset == "cirr" else (10, 50)))
    if not rows or min(ks) <= 0:
        raise ValueError("Need queries and positive cutoffs")
    gallery = set(map(str, gallery))
    predictions = {str(k): list(map(str, v)) for k, v in predictions.items()}
    ids = [str(r["id"] if dataset == "circo" else r["pairid"] if dataset == "cirr" else i) for i, r in enumerate(rows)]
    if len(set(ids)) != len(ids) or set(ids) != set(predictions):
        raise ValueError("Prediction IDs must exactly match annotation IDs")
    sums = {}
    for qid, row in zip(ids, rows):
        ranking = predictions[qid]
        ref = str(row["reference_img_id"] if dataset == "circo" else row["reference"] if dataset == "cirr" else row["candidate"])
        target = str(row["target_img_id"] if dataset == "circo" else row["target_hard"] if dataset == "cirr" else row["target"])
        if target not in gallery or ref not in gallery:
            raise ValueError("Ground truth/reference outside gallery")
        if target == ref:
            raise ValueError("Target equals reference")
        if dataset == "circo":
            positives = list(map(str, row["gt_img_ids"]))
            if not positives or len(positives) != len(set(positives)) or target not in positives or not set(positives) <= gallery or ref in positives:
                raise ValueError("Invalid CIRCO ground truth")
        if len(ranking) < min(max(ks), len(gallery - {ref})) or len(ranking) != len(set(ranking)):
            raise ValueError("Ranking too short or contains duplicates")
        if ref in ranking or not set(ranking) <= gallery:
            raise ValueError("Ranking contains reference or unknown gallery ID")
        for k in ks:
            key = f"R@{k}"
            sums[key] = sums.get(key, 0.0) + float(target in ranking[:k])
            if dataset == "circo":
                key = f"mAP@{k}"
                sums[key] = sums.get(key, 0.0) + average_precision(ranking, row["gt_img_ids"], k)
        if dataset == "cirr":
            members = set(map(str, row["img_set"]["members"])) - {ref}
            if not members <= set(ranking):
                raise ValueError("CIRR group metrics require full-gallery ranking (not Top-50)")
            group = [i for i in ranking if i in members]
            for k in (1, 2, 3):
                key = f"group_R@{k}"
                sums[key] = sums.get(key, 0.0) + float(target in group[:k])
    return {k: v / len(rows) for k, v in sums.items()}
