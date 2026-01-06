from typing import Dict, Any, Tuple, Optional
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score, f1_score

def _stats_at_threshold(probs: np.ndarray, y: np.ndarray, t: float) -> Dict[str,Any]:
    preds = (probs >= t).astype(np.int64)
    tn,fp,fn,tp = confusion_matrix(y,preds).ravel()
    TPR = tp/max(tp+fn,1); FPR = fp/max(fp+tn,1)
    balacc = 0.5*(TPR + (1.0-FPR)); youden = TPR - FPR
    acc = accuracy_score(y,preds); f1 = f1_score(y,preds,zero_division=0)
    return {"tn":tn,"fp":fp,"fn":fn,"tp":tp,"TPR":TPR,"FPR":FPR,"balacc":balacc,"youden":youden,"acc":acc,"f1":f1}

def threshold_from_roc(probs: np.ndarray, y: np.ndarray, metric: str="youden", target_fpr: Optional[float]=None):
    fpr, tpr, thr = roc_curve(y, probs)
    if target_fpr is not None:
        mask = fpr <= float(target_fpr)
        if not np.any(mask):
            idx = int(np.argmin(fpr))
        else:
            idx_local = int(np.argmax(tpr[mask]))
            idx = int(np.arange(len(fpr))[mask][idx_local])
        best_t = float(thr[idx]); return best_t, _stats_at_threshold(probs,y,best_t)

    if metric == "youden": idx = int(np.argmax(tpr - fpr))
    elif metric == "balacc": idx = int(np.argmax(0.5*(tpr + (1.0 - fpr))))
    elif metric == "auc":
        mask = np.isfinite(thr)
        fpr_c, tpr_c, thr_c = fpr[mask], tpr[mask], thr[mask]
        if len(thr_c)==0: idx = int(np.argmax(tpr - fpr))
        else:
            dist2 = (fpr_c**2) + ((1.0 - tpr_c)**2)
            idx_local = int(np.argmin(dist2))
            idx = int(np.where(mask)[0][idx_local])
    else:
        scores = []
        for t in thr:
            s = _stats_at_threshold(probs,y,float(t))
            if metric=="acc": scores.append(s["acc"])
            elif metric=="f1": scores.append(s["f1"])
            else: scores.append(s["youden"])
        idx = int(np.argmax(scores))
    best_t = float(thr[idx]); return best_t, _stats_at_threshold(probs,y,best_t)
