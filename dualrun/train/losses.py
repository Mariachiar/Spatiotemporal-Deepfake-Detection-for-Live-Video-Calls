import torch, torch.nn as nn
import torch.nn.functional as F

import torch


def alignment(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 2,
):
    """
    https://arxiv.org/pdf/2005.10242

    Label-aware Alignment loss.

    Calculates alignment for embeddings of samples with the SAME label
    within a batch, assuming embeddings are already unit-normalized.

    Args:
        embeddings: Tensor [N, D] - Batch of unit-normalized embeddings.
        labels: Tensor [N] - Corresponding labels.
        alpha: Power to raise squared distance (hyperparameter, default=2).

    Returns:
        Tensor: Label-aware Alignment loss (scalar). Returns 0 if no positive pairs.
    """
    assert embeddings.size(0) == labels.size(0), "Embeddings and labels must have the same size."

    n_samples = embeddings.size(0)
    if n_samples < 2:
        return torch.tensor(0.0, device=embeddings.device)

    # Create a pairwise label comparison matrix (N x N), exclude self-pairs
    labels_equal_mask = (labels[:, None] == labels[None, :]).triu(diagonal=1)

    positive_indices = torch.nonzero(labels_equal_mask, as_tuple=False)
    if positive_indices.numel() == 0:
        return torch.tensor(0.0, device=embeddings.device)

    # Get embeddings of positive pairs
    x = embeddings[positive_indices[:, 0]]
    y = embeddings[positive_indices[:, 1]]

    # Calculate alignment loss
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniformity(
    x: torch.Tensor,
    t: float = 2,
    clip_value: float = 1e-6,
):
    """
    https://arxiv.org/pdf/2005.10242

    Calculates the Uniformity loss.

    Args:
        x: [N, D] - Batch of feature embeddings.
        t: Temperature parameter (hyperparameter).

    Returns:
        Tensor: Uniformity loss value (scalar).
    """
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().clamp(min=clip_value).log()


if __name__ == "__main__":
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
    )
    embeddings /= embeddings.norm(p=2, dim=1, keepdim=True)

    labels = torch.tensor([0, 0, 0, 1, 1])

    print("Embeddings:")
    print(embeddings.numpy())

    print("\nLabels:")
    print(labels.numpy())

    alignment_loss = alignment(embeddings, labels, alpha=2)
    print("\nAlignment loss:", alignment_loss.item())

    uniformity_loss = uniformity(embeddings, t=2, clip_value=1e-6)
    print("Uniformity loss:", uniformity_loss.item())

class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma: float=2.0, alpha: float|None=None, reduction: str="mean"):
        super().__init__(); self.gamma=float(gamma); self.alpha=alpha; self.reduction=reduction
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits); pt = torch.where(targets==1, p, 1-p).clamp(1e-6, 1-1e-6)
        loss = (1-pt)**self.gamma * bce
        if self.alpha is not None:
            a = torch.as_tensor(self.alpha, dtype=loss.dtype, device=loss.device)
            loss = torch.where(targets==1, a*loss, (1-a)*loss)
        if self.reduction=="mean": return loss.mean()
        if self.reduction=="sum": return loss.sum()
        return loss
    

def mse_masked(pred, target, mask):
    # pred,target: [B,T,D], mask: [B,T] where True means keep (non-pad AND real)
    if mask is None:
        return F.mse_loss(pred, target)
    m = mask.unsqueeze(-1).float()
    diff2 = (pred - target)**2 * m
    denom = m.sum().clamp_min(1.0)
    return diff2.sum() / denom

def temporal_infonce(q, k, mask, temperature=0.1):
    # q,k: [B,T,P] l2-normalized
    B, T, P = q.shape
    q = F.normalize(q, dim=-1); k = F.normalize(k, dim=-1)

    q = q.reshape(B*T, P)                         # [BT,P]
    k = k.reshape(B*T, P)                         # [BT,P]
    logits = (q @ k.t()) / temperature            # [BT,BT]
    labels = torch.arange(B*T, device=q.device)   # [BT]

    if mask is not None:
        keep = (~mask).reshape(B*T)               # True = frame valido
        logits = logits[keep][:, keep]            # [K,K]
        labels = torch.arange(logits.size(0), device=q.device)

    return F.cross_entropy(logits, labels)
