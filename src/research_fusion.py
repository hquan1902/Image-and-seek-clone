"""Score-level fusion methods used by the reproducible retrieval pipeline."""
import torch


def dispersion_normalized_scores(image_scores, text_scores, eps=1e-6):
    """Fuse modalities after per-query gallery z-normalization.

    Centering does not alter a modality's ranking, but is retained so the
    returned values are interpretable. Population standard deviation is used
    because the gallery is the complete comparison population for the query.
    """
    if image_scores.shape != text_scores.shape or image_scores.ndim != 1:
        raise ValueError("Expected two one-dimensional score vectors of equal shape")
    if not torch.isfinite(image_scores).all() or not torch.isfinite(text_scores).all():
        raise ValueError("Fusion scores must be finite before reference exclusion")
    if eps <= 0:
        raise ValueError("eps must be positive")
    image_std = image_scores.std(unbiased=False).clamp_min(eps)
    text_std = text_scores.std(unbiased=False).clamp_min(eps)
    return ((image_scores - image_scores.mean()) / image_std
            + (text_scores - text_scores.mean()) / text_std)


def reciprocal_rank_fusion_scores(image_scores, text_scores, rank_constant=60.0):
    """Fuse two rankings using parameter-fixed reciprocal rank fusion."""
    if image_scores.shape != text_scores.shape or image_scores.ndim != 1:
        raise ValueError("Expected two one-dimensional score vectors of equal shape")
    if rank_constant <= 0:
        raise ValueError("rank_constant must be positive")
    size = image_scores.numel()
    ranks = []
    for scores in (image_scores, text_scores):
        order = torch.argsort(scores, descending=True, stable=True)
        rank = torch.empty_like(order)
        rank[order] = torch.arange(size, device=order.device)
        ranks.append(rank)
    return 1.0 / (rank_constant + ranks[0].float() + 1.0) + 1.0 / (rank_constant + ranks[1].float() + 1.0)
