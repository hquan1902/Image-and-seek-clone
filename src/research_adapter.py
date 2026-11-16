"""Small supervised two-stream control before introducing proxy-specific modules."""
import torch
from torch import nn
import torch.nn.functional as F


class AdaptiveComposer(nn.Module):
    def __init__(self, dimension, hidden=16, initial_image_weight=0.2, use_residual=False,
                 gate_input="summary"):
        super().__init__()
        if not 0 < initial_image_weight < 1:
            raise ValueError("initial_image_weight must be strictly between 0 and 1")
        self.initial_image_weight = initial_image_weight
        self.use_residual = use_residual
        if gate_input not in ("summary", "full"):
            raise ValueError("gate_input must be summary or full")
        self.gate_input = gate_input
        gate_dimension = 4 if gate_input == "summary" else 3*dimension
        self.gate = nn.Sequential(nn.Linear(gate_dimension, hidden), nn.GELU(), nn.Linear(hidden, 2))
        self.residual = (nn.Sequential(nn.Linear(3*dimension, hidden), nn.GELU(), nn.Linear(hidden, dimension))
                         if use_residual else None)
        # Start exactly at the normalized image+text baseline.
        nn.init.zeros_(self.gate[-1].weight)
        with torch.no_grad():
            self.gate[-1].bias.copy_(torch.log(torch.tensor([initial_image_weight, 1-initial_image_weight])))
        if self.residual is not None:
            nn.init.zeros_(self.residual[-1].weight)
            nn.init.zeros_(self.residual[-1].bias)

    def forward(self, reference, text):
        reference, text = F.normalize(reference, dim=-1), F.normalize(text, dim=-1)
        full_features = torch.cat((reference, text, reference*text), dim=-1)
        difference = (reference-text).abs()
        summary = torch.stack(((reference*text).sum(-1), difference.mean(-1), difference.amax(-1),
                               (reference*text).std(-1)), dim=-1)
        weights = self.gate(summary if self.gate_input == "summary" else full_features).softmax(-1)
        query = weights[:, :1]*reference + weights[:, 1:]*text
        if self.residual is not None:
            query = query + self.residual(full_features)
        return F.normalize(query, dim=-1), weights


class MultiDomainResidualComposer(nn.Module):
    """Shared low-rank residual around a strong per-domain static fusion."""
    def __init__(self, dimension, hidden=64, dropout=0.2, residual_scale=0.1):
        super().__init__()
        if hidden < 1 or not 0 <= dropout < 1 or residual_scale <= 0:
            raise ValueError("Invalid residual composer hyperparameters")
        self.dimension = dimension
        self.hidden = hidden
        self.dropout = dropout
        self.residual_scale = residual_scale
        self.adapter = nn.Sequential(
            nn.LayerNorm(4*dimension),
            nn.Linear(4*dimension, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dimension),
        )
        # Epoch 0 is exactly the supplied static baseline.
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, reference, text, base_image_weight):
        reference, text = F.normalize(reference, dim=-1), F.normalize(text, dim=-1)
        if not torch.is_tensor(base_image_weight):
            base_image_weight = reference.new_tensor(base_image_weight)
        while base_image_weight.ndim < reference.ndim:
            base_image_weight = base_image_weight.unsqueeze(-1)
        if torch.any((base_image_weight < 0) | (base_image_weight > 1)):
            raise ValueError("base_image_weight must be in [0, 1]")
        base = base_image_weight*reference + (1-base_image_weight)*text
        features = torch.cat((reference, text, reference*text, reference-text), dim=-1)
        residual = self.adapter(features)
        query = F.normalize(base + self.residual_scale*residual, dim=-1)
        return query, residual


def multi_positive_nce(query, targets, target_ids, temperature=0.07):
    """Same target IDs in a batch are positives, never negatives of one another."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = F.normalize(query, dim=-1) @ F.normalize(targets, dim=-1).T / temperature
    positive = target_ids[:, None] == target_ids[None, :]
    return (torch.logsumexp(logits, dim=1) - torch.logsumexp(logits.masked_fill(~positive, -torch.inf), dim=1)).mean()


def full_gallery_cross_entropy(query, gallery, target_indices, reference_indices=None,
                               temperature=0.07, label_smoothing=0.0):
    """Retrieval-aligned loss: classify each target against the complete gallery."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 <= label_smoothing < 1:
        raise ValueError("label_smoothing must be in [0, 1)")
    logits = F.normalize(query, dim=-1) @ F.normalize(gallery, dim=-1).T / temperature
    valid = None
    if reference_indices is not None:
        if torch.any(reference_indices == target_indices):
            raise ValueError("Reference and target indices must differ")
        valid = torch.ones_like(logits, dtype=torch.bool)
        valid.scatter_(1, reference_indices[:, None], False)
        logits = logits.masked_fill(~valid, -torch.inf)
    if not label_smoothing:
        return F.cross_entropy(logits, target_indices)
    log_prob = F.log_softmax(logits, dim=-1)
    nll = -log_prob.gather(1, target_indices[:, None]).squeeze(1)
    if valid is None:
        smooth = -log_prob.mean(dim=1)
    else:
        smooth = -log_prob.masked_fill(~valid, 0).sum(dim=1) / valid.sum(dim=1)
    return ((1-label_smoothing)*nll + label_smoothing*smooth).mean()


def hard_negative_margin_loss(query, gallery, target_indices, reference_indices=None,
                              topk=32, margin=0.05, temperature=0.07):
    """Focus learning on the highest-scoring incorrect gallery items."""
    if topk < 1 or margin < 0 or temperature <= 0:
        raise ValueError("Invalid hard-negative loss configuration")
    scores = F.normalize(query, dim=-1) @ F.normalize(gallery, dim=-1).T
    positive = scores.gather(1, target_indices[:, None])
    excluded = torch.zeros_like(scores, dtype=torch.bool)
    excluded.scatter_(1, target_indices[:, None], True)
    if reference_indices is not None:
        if torch.any(reference_indices == target_indices):
            raise ValueError("Reference and target indices must differ")
        excluded.scatter_(1, reference_indices[:, None], True)
    available = scores.shape[1] - (2 if reference_indices is not None else 1)
    if available < 1:
        raise ValueError("Gallery must contain at least one valid negative")
    negatives = scores.masked_fill(excluded, -torch.inf).topk(
        min(topk, available), dim=1).values
    return F.softplus((negatives-positive+margin)/temperature).mean()


def group_cross_entropy(query, gallery, group_indices, target_indices, temperature=0.07):
    """CIRR auxiliary objective: rank the target inside its five-image group."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if group_indices.ndim != 2 or group_indices.shape[0] != query.shape[0]:
        raise ValueError("group_indices must have shape [batch, group_size]")
    positive = group_indices == target_indices[:, None]
    if not torch.all(positive.sum(dim=1) == 1):
        raise ValueError("Each group must contain its target exactly once")
    query = F.normalize(query, dim=-1)
    candidates = F.normalize(gallery[group_indices], dim=-1)
    logits = torch.einsum("bd,bgd->bg", query, candidates) / temperature
    return F.cross_entropy(logits, positive.float().argmax(dim=1))
