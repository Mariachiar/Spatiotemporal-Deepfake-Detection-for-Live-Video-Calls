# train/metrics.py
from typing import Dict, Any
import numpy as np, torch
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    accuracy_score, f1_score
)
from tqdm.auto import tqdm

__all__ = ["collect_logits", "metrics_from_logits"]


def _ema1d(x, alpha):
    # x: (B,T,D)
    if alpha <= 0: return x
    y = x.clone()
    for t in range(1, x.size(1)):
        y[:, t] = alpha*y[:, t-1] + (1-alpha)*x[:, t]
    return y

@torch.no_grad()
def collect_logits(model, loader, device, use_tqdm=False, desc="Eval",
                   smooth_alpha=0.0, tta_quality_n=0, return_ids=False):
    model.eval()
    all_logits, all_y = [], []
    all_trk, all_vid = [], []
    it = loader if not use_tqdm else tqdm(loader, desc=desc)
    for X, y in it:
        trk_id = vid_id = None
        if isinstance(X, (tuple, list)):
            A, L, lengths = X[0], X[1], X[2]
            if len(X) >= 5:
                trk_id, vid_id = X[-2], X[-1]
        else:
            A, L, lengths = X["A"], X["L"], X["lengths"]
        A, L = A.to(device), L.to(device)
        if smooth_alpha > 0.0:
            A = _ema1d(A, smooth_alpha); L = _ema1d(L, smooth_alpha)
        logits = model(A, L, lengths=lengths.to(device), dat_lambda=0.0)["bin_logits"].view(-1)
        all_logits.append(logits.cpu()); all_y.append(y.cpu())
        if return_ids and (trk_id is not None) and (vid_id is not None):
            all_trk.append(trk_id.cpu()); all_vid.append(vid_id.cpu())
    logits_np = torch.cat(all_logits).numpy()
    y_np = torch.cat(all_y).numpy()
    if return_ids and all_trk:
        trk_np = torch.cat(all_trk).numpy(); vid_np = torch.cat(all_vid).numpy()
        return logits_np, y_np, trk_np, vid_np
    return logits_np, y_np



def metrics_from_logits(logits: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    probs = 1.0/(1.0+np.exp(-logits))
    probs = np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
    preds = (probs >= threshold).astype(np.int64)

    cm = confusion_matrix(y, preds, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()

    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    balacc = 0.5 * (tpr + (1 - fpr))
    youden = tpr - fpr
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, zero_division=0)
    try: auc = roc_auc_score(y, probs)
    except: auc = float("nan")
    try: prauc = average_precision_score(y, probs)
    except: prauc = float("nan")
    return {"tn":tn,"fp":fp,"fn":fn,"tp":tp,"TPR":tpr,"FPR":fpr,"balacc":balacc,
            "youden":youden,"acc":acc,"f1":f1,"roc_auc":auc,"pr_auc":prauc,"probs":probs}
