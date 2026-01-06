import os, math, logging, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from typing import Optional
from tqdm import tqdm  
import torch.optim as optim

from utils.setup import device_of
from utils.io import save_json
from train import metrics as M
from train.thresholds import threshold_from_roc
from train.losses import BinaryFocalLoss, alignment, uniformity, mse_masked, temporal_infonce
from train.samplers import BalancedPerTechBaseSampler, BalancedPerTechLOOSampler
from data.dataset_dual import DualFeaturesClipDataset, _infer_tech_from_path
from model.dual_encoder import grad_reverse
import torch.nn.functional as F
from train import altfreezing

LOG = logging.getLogger("dualrun.engine")

def _slerp(A: torch.Tensor, B: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    A = F.normalize(A, dim=-1); B = F.normalize(B, dim=-1)
    dot = (A * B).sum(dim=-1, keepdim=True).clamp(-1 + 1e-7, 1 - 1e-7)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    near = sin_theta < 1e-6
    t_theta = t * theta
    coeff_a = torch.sin(theta - t_theta) / sin_theta
    coeff_b = torch.sin(t_theta) / sin_theta
    slerped = coeff_a * A + coeff_b * B
    lerped  = (1 - t) * A + t * B
    return torch.where(near, lerped, slerped)

def _slerp_aug_per_class(features: torch.Tensor, labels: torch.Tensor,
                         t0: float, t1: float) -> torch.Tensor:
    out = features.clone()
    for c in torch.unique(labels):
        idx = (labels == c).nonzero(as_tuple=False).squeeze(1)
        if idx.numel() < 2: 
            continue
        A = features.index_select(0, idx)
        B = features.index_select(0, idx.roll(shifts=1))
        t = torch.rand((A.size(0), 1), device=features.device, dtype=features.dtype)
        t = t * (t1 - t0) + t0
        aug = _slerp(A, B, t)
        out.index_copy_(0, idx, aug.to(features.dtype))
    return out 

class EarlyStopper:
    def __init__(self, patience: int = 10, mode: str = "max", warmup_epochs: int = 0, atol: float = 1e-6):
        assert mode in {"min", "max"}
        self.patience = int(patience); self.mode = mode
        self.warmup_epochs = int(warmup_epochs); self.counter = 0
        self.best_score = math.inf if mode == "min" else -math.inf
        self.best_epoch = 0; self.atol = float(atol)

    def _is_better(self, score: float) -> bool:
        return score > (self.best_score + self.atol) if self.mode == "max" else score < (self.best_score - self.atol)

    def __call__(self, score: float, epoch: int) -> bool:
        if self._is_better(score):
            self.best_score = score; self.best_epoch = epoch; self.counter = 0
        else:
            if epoch >= self.warmup_epochs:
                self.counter += 1
        if epoch < self.warmup_epochs: return False
        if self.counter >= self.patience:
            LOG.warning(f"Early stopping: best {self.best_score:.6f} at epoch {self.best_epoch}.")
            return True
        return False

def _person_trimmed_mean_probs_from_logits(logits, ids, trim=0.5):
    p = torch.sigmoid(logits.float()).clamp(1e-6, 1-1e-6)
    order = torch.argsort(ids)
    ids_s, p_s = ids[order], p[order]
    uniq, counts = torch.unique_consecutive(ids_s, return_counts=True)
    chunks = torch.split(p_s, counts.tolist())
    tm = []
    for c in chunks:
        k = c.numel()
        if k <= 2: tm.append(c.mean()); continue
        c_sorted, _ = torch.sort(c)
        m = int(round(k*trim/2))
        tm.append(c_sorted[m:k-m].mean())
    return uniq, torch.stack(tm), counts, order



def build_optimizer_scheduler_and_loss(model, args, device):
    LOG.info(f"[TRAIN] lr={args.lr} wd={args.wd}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # lo scheduler viene creato in train(), quando sappiamo steps_per_epoch
    scheduler = None

    if args.focal:
        criterion = BinaryFocalLoss(gamma=args.focal_gamma, alpha=args.focal_alpha)
    else:
        if args.pos_weight is not None:
            pw = torch.tensor(float(args.pos_weight), device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
            LOG.info(f"BCEWithLogitsLoss(pos_weight={float(args.pos_weight):.3f})")
        else:
            criterion = nn.BCEWithLogitsLoss()
            LOG.info("BCEWithLogitsLoss()")

    scaler = GradScaler(enabled=(args.amp and device.type == "cuda"))
    return optimizer, scheduler, criterion, scaler


def _save_checkpoint(path, model, optimizer, scheduler, epoch, best_val, best_thresh: float):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optim": optimizer.state_dict() if optimizer else None,
        "sched": scheduler.state_dict() if scheduler else None,
        "best_val": best_val,
        "best_thresh": float(best_thresh)
    }, path)
    with open(os.path.join(os.path.dirname(path), "best_threshold.txt"), "w") as f:
        f.write(f"{best_thresh:.6f}\n")
    LOG.info(f"Checkpoint saved: {path} | best_thresh={best_thresh:.3f}")


@torch.no_grad()
def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def fit_temperature_on_val(val_logits: np.ndarray, val_y: np.ndarray, device: torch.device) -> float:
    """
    Trova T* minimizzando la NLL su validation: p = sigmoid(logits / T).
    Non cambia il ranking (AUC invariato), ma calibra le probabilità.
    """
    z = torch.tensor(val_logits, device=device, dtype=torch.float32)   # (N,)
    y = torch.tensor(val_y,      device=device, dtype=torch.float32)   # (N,)
    T = torch.tensor(1.0, device=device, requires_grad=True)

    bce = nn.BCEWithLogitsLoss()
    opt = optim.LBFGS([T], lr=0.1, max_iter=50, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = bce(z / T.clamp(1e-2, 1e3), y) + 1e-4*(T-1.0).pow(2) 
        loss.backward()
        return loss

    try:
        opt.step(closure)
        T_star = float(T.detach().clamp(0.25, 20.0).item())
    except Exception as e:
        LOG.warning(f"Temperature scaling fallback (T=1.0) causa errore: {e}")
        T_star = 1.0

    return T_star

def _log1mexp(x: torch.Tensor) -> torch.Tensor:
    # x <= 0
    return torch.where(x < -0.69314718056, torch.log1p(-torch.exp(x)), torch.log(-torch.expm1(x)))

def _group_median_probs_from_logits(logits: torch.Tensor, ids: torch.Tensor):
    # ritorna (uniq_ids, p_group_median, counts, order) per riusare counts/order
    p = torch.sigmoid(logits.float())  
    order = torch.argsort(ids)
    ids_s, p_s = ids[order], p[order]
    uniq, counts = torch.unique_consecutive(ids_s, return_counts=True)
    chunks = torch.split(p_s, counts.tolist())
    meds = torch.stack([c.median() for c in chunks])
    return uniq, meds, counts, order

def _logit_torch(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = p.clamp(eps, 1.0 - eps)
    return torch.log(p) - torch.log1p(-p)


def _agg_video_from_person_median(bin_logits, trk_id, vid_id, y):
    # 1) mediana per persona (track)
    uniq_trk, p_person, counts_trk, order = _group_median_probs_from_logits(bin_logits, trk_id)
    y_s = y[order]
    y_trk_chunks = torch.split(y_s, counts_trk.tolist())
    y_person = torch.stack([(c.float().mean() >= 0.5).float() for c in y_trk_chunks])

    # prendi il vid_id rappresentativo di ogni persona
    vid_s = vid_id[order]
    vid_trk_chunks = torch.split(vid_s, counts_trk.tolist())
    vid_per_person = torch.stack([c[0] for c in vid_trk_chunks])

    # 2) OR su video: p_video = 1 - ∏(1 - p_person)
    order2 = torch.argsort(vid_per_person)
    vid2 = vid_per_person[order2]; pp2 = p_person[order2]; yp2 = y_person[order2]
    uniq_vid, counts_vid = torch.unique_consecutive(vid2, return_counts=True)
    pp_chunks = torch.split(pp2, counts_vid.tolist())
    yp_chunks = torch.split(yp2, counts_vid.tolist())
    p_video = []
    for c in pp_chunks:
        p = c.float().clamp(1e-6, 1-1e-6)
        s = torch.log1p(-p).sum()                    # somma log(1-p)
        p_video.append(1.0 - torch.exp(s).clamp_min(1e-12))
    p_video = torch.stack(p_video)
    y_video = torch.stack([c.max() for c in yp_chunks]).float()
    return p_video, y_video

def _agg_noisyor_person_logits(logits: torch.Tensor, ids: torch.Tensor, y: torch.Tensor):
    # logits: [B] (logit clip), ids: [B] (trk_id o vid_id), y: [B] {0,1}
    order = torch.argsort(ids)
    ids_s   = ids[order]
    z_s     = logits[order]
    y_s     = y[order]

    uniq, counts = torch.unique_consecutive(ids_s, return_counts=True)
    z_chunks = torch.split(z_s, counts.tolist())
    y_chunks = torch.split(y_s, counts.tolist())

    person_logits, person_y = [], []
    for zc, yc in zip(z_chunks, y_chunks):
        s = torch.logsigmoid(-zc).sum()         # log ∏(1 - p_i)
        log_p = _log1mexp(s)                    # log p_person = log(1 - exp(s))
        lp = log_p - s                          # logit(p) = log p - log(1-p)
        person_logits.append(lp)
        person_y.append(yc[0])                  # label coerente nel track
    return torch.stack(person_logits), torch.stack(person_y)

def _np_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1-1e-6)
    return np.log(p) - np.log(1-p)

def _agg_person_from_logits_for_eval(logits_np: np.ndarray, y_np: np.ndarray, trk_np: np.ndarray):
    z = torch.tensor(logits_np)
    y = torch.tensor(y_np)
    ids = torch.tensor(trk_np)
    _, p_person, counts, order = _group_median_probs_from_logits(z, ids)
    y_s = y[order]
    y_chunks = torch.split(y_s, counts.tolist())
    y_person = torch.stack([(c.float().mean() >= 0.5).float() for c in y_chunks])
    return p_person.detach().cpu().numpy(), y_person.detach().cpu().numpy()

def _agg_video_from_logits_for_eval(logits_np: np.ndarray, y_np: np.ndarray, trk_np: np.ndarray, vid_np: np.ndarray):
    z = torch.tensor(logits_np)
    y = torch.tensor(y_np)
    trk = torch.tensor(trk_np)
    vid = torch.tensor(vid_np)
    p_video, y_video = _agg_video_from_person_median(z, trk, vid, y)
    return p_video.detach().cpu().numpy(), y_video.detach().cpu().numpy()

def _group_mean_probs_from_logits(logits: torch.Tensor, ids: torch.Tensor):
    p = torch.sigmoid(logits.float())
    order = torch.argsort(ids)
    ids_s, p_s = ids[order], p[order]
    uniq, counts = torch.unique_consecutive(ids_s, return_counts=True)
    chunks = torch.split(p_s, counts.tolist())
    means = torch.stack([c.mean() for c in chunks])
    return uniq, means, counts, order

def _parse_boosts(slist):
        # accetta: ["neuraltextures:3", "faceswap:1.5"]
        out = {}
        if not slist: return out
        for s in slist:
            try:
                k,v = s.split(":")
                out[k.strip().lower()] = float(v)
            except Exception:
                continue
        return out

def train(model, train_ds: DualFeaturesClipDataset, val_ds: DualFeaturesClipDataset, test_ds: DualFeaturesClipDataset, args, heldout: Optional[str] = None):
    import numpy as np
    from sklearn import metrics as skm

    device = device_of(args.device); model.to(device)

    # --- flag e pesi ---
    use_dat = bool(args.dat)
    qual_lambda = float(getattr(args, "qual_lambda", 0.0) or 0.0)
    qual_ce_w   = float(getattr(args, "qual_ce_weight", 1.0) or 1.0)
    attn_entropy_lam = float(getattr(args, "attn_entropy", 0.0) or 0.0)
    attn_agree_lam   = float(getattr(args, "attn_agree",   0.0) or 0.0)
    use_attn_regs    = (attn_entropy_lam > 0.0) or (attn_agree_lam > 0.0)
    aux_pred_w = float(getattr(args, "aux_pred_w", 0.0) or 0.0)
    aux_con_w  = float(getattr(args, "aux_con_w",  0.0) or 0.0)
    use_aux    = (aux_pred_w > 0.0) or (aux_con_w > 0.0)
    lam_align   = float(getattr(args, "lam_align",   0.0))
    lam_uniform = float(getattr(args, "lam_uniform", 0.0))
    uni_t       = float(getattr(args, "uniform_t",   2.0))
    cons_w      = float(getattr(args, "consistency_w", 0.0) or 0.0)

    LOG.info(f"[AUX] pred_w={aux_pred_w} con_w={aux_con_w}")
    LOG.info(f"[QUALITY] lambda={qual_lambda} ce_w={qual_ce_w}")
    LOG.info(f"[ATTN REG] entropy={attn_entropy_lam} agree={attn_agree_lam}")
    LOG.info(f"[DAT] enabled={use_dat}")

    # freeze logic
    use_split_freeze = (getattr(args, "freeze_lmk", 0) > 0) or (getattr(args, "freeze_au", 0) > 0)
    if use_split_freeze and getattr(args, "freeze_encoders", 0) > 0:
        LOG.warning("freeze_encoders ignorato: attivi freeze_lmk/freeze_au")
        args.freeze_encoders = 0

    # --- sampler ---
    if hasattr(train_ds, "tech_names") and train_ds.tech_names:
        train_tech_names = [str(t or "unknown").lower() for t in train_ds.tech_names]
    else:
        train_tech_names = [str(_infer_tech_from_path(d) or "unknown").lower() for d in train_ds.clip_dirs]

    labels_set = set(int(y) for y in train_ds.labels)
    epoch_samples = int(args.epoch_samples)
    if epoch_samples % 2 != 0:
        LOG.warning(f"epoch_samples={epoch_samples} non pari; incremento a {epoch_samples+1}")
        epoch_samples += 1

    boosts = _parse_boosts(getattr(args, "boost_tech", None))
    min_quota = int(getattr(args, "min_quota_fake", 0))

    if len(labels_set) < 2:
        from torch.utils.data import RandomSampler
        sampler = RandomSampler(range(len(train_ds)), replacement=True, num_samples=epoch_samples)
        save_json({"mode": "single_class_uniform", "epoch_samples": epoch_samples},
                  os.path.join(args.out, "sampler_config.json"))
    else:
        heldout_arg = (getattr(args, "heldout_tech", "") or "").lower()
        use_heldout = (heldout or heldout_arg)
        if use_heldout:
            sampler = BalancedPerTechLOOSampler(
                labels=train_ds.labels, tech_names=train_tech_names, heldout=use_heldout,
                epoch_samples=epoch_samples, seed_base=args.seed,
                reshuffle_each_epoch=bool(getattr(args, "shuffle_every_epoch", True)),
                boosts=boosts, min_quota=min_quota
            )
            save_json({"mode":"balanced_per_tech_loo","epoch_samples":epoch_samples,
                       "heldout":use_heldout,"boosts":boosts,"min_quota":min_quota},
                       os.path.join(args.out, "sampler_config.json"))
        else:
            sampler = BalancedPerTechBaseSampler(
                labels=train_ds.labels, tech_names=train_tech_names,
                epoch_samples=epoch_samples, seed_base=args.seed,
                reshuffle_each_epoch=bool(getattr(args, "shuffle_every_epoch", True)),
                boosts=boosts, min_quota=min_quota
            )
            save_json({"mode":"balanced_per_tech_base","epoch_samples":epoch_samples,
                       "boosts":boosts,"min_quota":min_quota},
                       os.path.join(args.out, "sampler_config.json"))

    train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=False, persistent_workers=bool(args.num_workers > 0))
    val_loader = DataLoader(val_ds, batch_size=args.batch_eval, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    optimizer, scheduler, criterion, scaler = build_optimizer_scheduler_and_loss(model, args, device)
    eval_agg = str(getattr(args, "eval_agg", "none")).lower()
    alt_cfg = altfreezing.AltFreezeCfg(
        enabled=bool(getattr(args,"altfreeze_enabled",1)),
        warmup_epochs=int(getattr(args,"altfreeze_warmup",2)),
        period=int(getattr(args,"altfreeze_period",2)),
        joint_tail=int(getattr(args,"altfreeze_joint_tail",2)),
        start_epoch=int(getattr(args,"altfreeze_start",1)),
    )
    altf = altfreezing.AltFreezer(alt_cfg)
    last_epoch = int(getattr(args,"epochs",80))

    # --- scheduler ---
    steps_per_epoch = max(1, len(train_loader))
    if getattr(args, "scheduler", "cosine") == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps_per_epoch,
            pct_start=getattr(args, "onecycle_pct_start", 0.10),
            div_factor=getattr(args, "onecycle_div_factor", 25.0),
            final_div_factor=getattr(args, "onecycle_final_div", 1e4),
            anneal_strategy="cos"
        )
        LOG.info(f"Scheduler: OneCycleLR (max_lr={args.lr}, steps/epoch={steps_per_epoch})")
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs // 3))

        LOG.info("Scheduler: CosineAnnealingLR")

    stopper = EarlyStopper(patience=args.patience, mode="max", warmup_epochs=args.es_warmup)
    best_ckpt_path = os.path.join(args.out, "best.pt") if args.out else None
    best_metric, best_thresh = -1e9, 0.5
    global_step = 0
    

    for epoch in range(1, args.epochs + 1):
        model.train()
        if alt_cfg.enabled:
            phase = altf.apply(model, epoch, last_epoch, logger=LOG)
            LOG.info(f"[EPOCH {epoch}] altfreeze phase={phase} (A:freeze_lmk, B:freeze_au, joint)")
        else:
            phase = "joint"

         
        if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            LOG.info(f"[OneCycle] pct_start={getattr(args,'onecycle_pct_start',0.10)} "
                    f"div={getattr(args,'onecycle_div_factor',25.0)} "
                    f"final_div={getattr(args,'onecycle_final_div',1e4)}")


        for p in model.head.parameters(): p.requires_grad = True
        if use_dat and (model.domain_head is not None):
            for p in model.domain_head.parameters(): p.requires_grad = True

        if not alt_cfg.enabled:
            freeze_lmk = int(epoch <= getattr(args,"freeze_lmk",0))
            freeze_au  = int(epoch <= getattr(args,"freeze_au",0))
            for p in model.lmk_enc.parameters(): p.requires_grad = not bool(freeze_lmk)
            for p in model.au_enc.parameters():  p.requires_grad = not bool(freeze_au)
            if getattr(args,"freeze_encoders",0) > 0 and not (freeze_lmk or freeze_au):
                freeze_now = epoch <= args.freeze_encoders
                for p in model.au_enc.parameters():  p.requires_grad = not freeze_now
                for p in model.lmk_enc.parameters(): p.requires_grad = not freeze_now
        else:
            # quando AltFreezing è attivo, ignora tutti i freeze legacy
            args.freeze_encoders = 0
            freeze_lmk = freeze_au = 0
        head_prev = [p.data.detach().clone() for p in model.head.parameters()]

        if hasattr(sampler, "set_epoch"): sampler.set_epoch(epoch)
        LOG.info(f"[EPOCH {epoch}] freeze_lmk={freeze_lmk} freeze_au={freeze_au} "f"freeze_encoders<={getattr(args,'freeze_encoders',0)} eval_agg={eval_agg}")
        # --- breakdown accumulators ---
        main_sum = auxp_sum = auxc_sum = dat_sum = qual_sum = attn_sum = align_sum = uniform_sum = cons_sum = 0.0
        n_seen = 0
        running = 0.0
        bad_pad = 0

        train_iter = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch") if getattr(args,"tqdm",False) else train_loader

        for batch in train_iter:
            A_noisy = None
            L_noisy = None
            dom_id = None
            q = None
            trk_id = None
            vid_id = None
            # supporta sia dataset_features (X,y) sia dataset_video_regen (A,L,len,trk,vid,y,extras)
            if isinstance(batch, (tuple, list)) and len(batch) == 7:
                A, L, t_valid, trk_id, vid_id, y, extras = batch
                dom_id = extras.get("dom_id", None) if isinstance(extras, dict) else None
                q      = extras.get("q", None)      if isinstance(extras, dict) else None
                A_noisy = extras.get("au_noisy", None)
                L_noisy = extras.get("lmk_noisy", None)
            else:
                X, y = batch
                A = L = t_valid = dom_id = q = None
                trk_id = vid_id = None
                if isinstance(X, (tuple, list)):
                    X = list(X)
                    if len(X) >= 5:
                        trk_id, vid_id = X[-2], X[-1]; X = X[:-2]
                    if len(X) == 5:   A, L, t_valid, dom_id, q = X
                    elif len(X) == 4:
                        if use_dat:  A, L, t_valid, dom_id = X
                        else:        A, L, t_valid, dom_id = X[0], X[1], X[2], None
                    elif len(X) == 3:
                        A, L, t_valid = X 
                else:
                    A, L, t_valid = X

            # to(device)
            if trk_id is not None: trk_id = trk_id.to(device, non_blocking=True).long()
            if vid_id is not None: vid_id = vid_id.to(device, non_blocking=True).long()
            A = A.to(device, non_blocking=True); L = L.to(device, non_blocking=True)
            t_valid = t_valid.to(device, non_blocking=True)
            if dom_id is not None: dom_id = dom_id.to(device, non_blocking=True).long()
            if q is not None:      q = q.to(device, non_blocking=True).long()
            y = y.float().to(device, non_blocking=True)
            if A_noisy is not None: A_noisy = A_noisy.to(device, non_blocking=True)
            if L_noisy is not None: L_noisy = L_noisy.to(device, non_blocking=True)


            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=(args.amp and device.type == "cuda")):
                mask = model.lengths_to_mask(t_valid, A.size(1), A.device)  # True=pad
                if mask.dim() == 2:
                    valid_frac = float((~mask).float().mean().item())
                else:
                    valid_frac = float((~mask).float().mean().item())

                if mask.dim() == 2:
                    bad_pad += int(mask.all(dim=1).sum().item())

                # forward
                if use_attn_regs or use_aux:
                    za, wa, za_seq = model.au_enc(A, key_padding_mask=mask, return_weights=True, return_seq=True)
                    zl, wl, zl_seq = model.lmk_enc(L, key_padding_mask=mask, return_weights=True, return_seq=True)
                else:
                    za = model.au_enc(A, key_padding_mask=mask)
                    zl = model.lmk_enc(L, key_padding_mask=mask)

                z = torch.cat([za, zl], dim=-1)

                # z clean già calcolato (concat di za, zl)
                z_clean = z

                # SLERP aug opzionale sui clean
                if bool(getattr(args, "slerp_feature_augmentation", False)):
                    r = getattr(args, "slerp_feature_augmentation_range", (0.0, 1.0))
                    if not (isinstance(r, (list, tuple)) and len(r) == 2): r = (0.0, 1.0)
                    t0, t1 = float(r[0]), float(r[1])
                    z_clean = _slerp_aug_per_class(F.normalize(z_clean, p=2, dim=-1), labels=y.long(), t0=t0, t1=t1)

                # vista degradata -> embedding
                cons_term = torch.tensor(0., device=device)
                if cons_w > 0.0 and (A_noisy is not None) and (L_noisy is not None):
                    if use_attn_regs or use_aux:
                        za_n, _, _ = model.au_enc(A_noisy, key_padding_mask=mask, return_weights=True, return_seq=True)
                        zl_n, _, _ = model.lmk_enc(L_noisy, key_padding_mask=mask, return_weights=True, return_seq=True)
                    else:
                        za_n = model.au_enc(A_noisy, key_padding_mask=mask)
                        zl_n = model.lmk_enc(L_noisy, key_padding_mask=mask)
                    z_noisy = torch.cat([za_n, zl_n], dim=-1)
                    # MSE tra embedding L2-normalizzati
                    cons_term = cons_w * F.mse_loss(F.normalize(z_clean, dim=-1), F.normalize(z_noisy, dim=-1))

                # testa binaria
                bin_logits = model.head(z_clean).squeeze(-1)

                agg = str(getattr(args, "train_agg", "none")).lower()

                # main loss
                if agg == "track_median":
                    assert trk_id is not None
                    with autocast(device_type="cuda", enabled=False):
                        _, p_person, counts, order = _group_median_probs_from_logits(bin_logits, trk_id)
                        y_s = y[order]
                        y_person = torch.stack([(c.float().mean() >= 0.5).float() for c in torch.split(y_s, counts.tolist())])
                        main_loss = F.binary_cross_entropy(p_person.clamp(1e-6,1-1e-6), y_person)
                elif agg == "track_mean":
                    assert trk_id is not None
                    with autocast(device_type="cuda", enabled=False):
                        _, p_person, counts, order = _group_mean_probs_from_logits(bin_logits, trk_id)
                        y_s = y[order]
                        y_person = torch.stack([(c.float().mean() >= 0.5).float() for c in torch.split(y_s, counts.tolist())])
                        main_loss = F.binary_cross_entropy(p_person.clamp(1e-6,1-1e-6), y_person)
                elif agg == "video_or_median":
                    assert (trk_id is not None) and (vid_id is not None)
                    with autocast(device_type="cuda", enabled=False):
                        _, p_person, counts, order = _group_median_probs_from_logits(bin_logits, trk_id)
                        y_s = y[order]
                        y_person = torch.stack([(c.float().mean() >= 0.5).float() for c in torch.split(y_s, counts.tolist())])
                        vid_s = vid_id[order]; vid_chunks = torch.split(vid_s, counts.tolist())
                        vid_per_person = torch.stack([c[0] for c in vid_chunks])
                        o2 = torch.argsort(vid_per_person)
                        vid2, pp2, yp2 = vid_per_person[o2], p_person[o2].clamp(1e-6,1-1e-6), y_person[o2]
                        uniq_vid, counts_vid = torch.unique_consecutive(vid2, return_counts=True)
                        pp_chunks = torch.split(pp2, counts_vid.tolist())
                        yp_chunks = torch.split(yp2, counts_vid.tolist())
                        p_video = torch.stack([1.0 - torch.exp(torch.log1p(-c).sum()).clamp_min(1e-12) for c in pp_chunks])
                        y_video = torch.stack([c.max() for c in yp_chunks]).float()
                        main_loss = F.binary_cross_entropy(p_video, y_video)
                elif agg == "video_or_mean":
                    assert (trk_id is not None) and (vid_id is not None)
                    with autocast(device_type="cuda", enabled=False):
                        _, p_person, counts, order = _group_mean_probs_from_logits(bin_logits, trk_id)
                        y_s = y[order]
                        y_person = torch.stack([(c.float().mean() >= 0.5).float() for c in torch.split(y_s, counts.tolist())])
                        vid_s = vid_id[order]; vid_chunks = torch.split(vid_s, counts.tolist())
                        vid_per_person = torch.stack([c[0] for c in vid_chunks])
                        o2 = torch.argsort(vid_per_person)
                        vid2, pp2, yp2 = vid_per_person[o2], p_person[o2].clamp(1e-6,1-1e-6), y_person[o2]
                        uniq_vid, counts_vid = torch.unique_consecutive(vid2, return_counts=True)
                        pp_chunks = torch.split(pp2, counts_vid.tolist())
                        yp_chunks = torch.split(yp2, counts_vid.tolist())
                        p_video = torch.stack([1.0 - torch.exp(torch.log1p(-c).sum()).clamp_min(1e-12) for c in pp_chunks])
                        y_video = torch.stack([c.max() for c in yp_chunks]).float()
                        main_loss = F.binary_cross_entropy(p_video, y_video)
                else:
                    main_loss = criterion(bin_logits.view(-1), y)

                loss = main_loss
                loss = loss + cons_term

                # aux terms
                aux_pred_term = torch.tensor(0., device=device)
                aux_con_term  = torch.tensor(0., device=device)


                if use_aux:
                    if aux_pred_w > 0.0:
                        B, T, Dl = zl_seq.shape
                        au_hat = model.au_from_lmk(zl_seq.reshape(B*T, Dl)).reshape(B, T, -1)
                        is_real = (y == 0).view(-1,1).expand(-1, A.size(1))
                        valid   = (~mask) & is_real
                        diff = F.smooth_l1_loss(au_hat, A, reduction="none")
                        loss_auxP = (diff * valid.unsqueeze(-1).float()).sum() / valid.sum().clamp_min(1.0)
                        aux_pred_term = aux_pred_term + aux_pred_w * loss_auxP
                    if aux_con_w > 0.0:
                        proj_au  = model.proj_au(za_seq)
                        proj_lmk = model.proj_lmk(zl_seq)
                        aux_con_term = aux_con_w * temporal_infonce(proj_lmk, proj_au, mask)

                    loss = loss + aux_pred_term + aux_con_term

                # DAT
                dat_term = torch.tensor(0., device=device)
                if use_dat and (dom_id is not None) and (model.domain_head is not None):
                    lam = (args.dat_lambda * (epoch / max(1, args.epochs))) if args.dat_schedule=="linear" else args.dat_lambda
                    z_rev = grad_reverse(z_clean, lam)
                    dom_logits = model.domain_head(z_rev)
                    did = dom_id.view(-1).long()

                    C = dom_logits.size(-1)
                    valid = (did >= 0) & (did < C)
                    if valid.any():
                        dat_term = F.cross_entropy(dom_logits[valid], did[valid])
                    # else: lascia dat_term=0 e prosegui
                    loss = loss + dat_term

                    #LOG.info(f"[DAT] domain_head C={model.domain_head.out_features} | did range in train batch: [{int(did.min())}, {int(did.max())}]")

                # quality
                qual_term = torch.tensor(0., device=device)
                if qual_lambda > 0.0 and q is not None and hasattr(model, "quality_head"):
                    qlogits = model.quality_head(grad_reverse(z_clean, qual_lambda))
                    qual_term = qual_ce_w * nn.functional.cross_entropy(qlogits, q)
                    loss = loss + qual_term

                # attn regs
                attn_term = torch.tensor(0., device=device)
                if use_attn_regs:
                    eps = 1e-8
                    def _entropy_neg(w):
                        Tt = w.size(1)
                        return -(w.clamp_min(eps) * w.clamp_min(eps).log()).sum(dim=1) / math.log(max(Tt, 2))
                    if attn_entropy_lam > 0.0:
                        ent_au  = _entropy_neg(wa).mean()
                        ent_lmk = _entropy_neg(wl).mean()
                        attn_term = attn_term + attn_entropy_lam * (ent_au + ent_lmk)
                    if attn_agree_lam > 0.0:
                        kl1 = nn.functional.kl_div(wa.clamp_min(eps).log(), wl.clamp_min(eps), reduction="batchmean")
                        kl2 = nn.functional.kl_div(wl.clamp_min(eps).log(), wa.clamp_min(eps), reduction="batchmean")
                        attn_term = attn_term + attn_agree_lam * (kl1 + kl2)
                    loss = loss + attn_term

                # align/uniform
                align_term = torch.tensor(0., device=device)
                uniform_term = torch.tensor(0., device=device)
                if (lam_align > 0.0) or (lam_uniform > 0.0):
                    z_norm = torch.nn.functional.normalize(z_clean, p=2, dim=-1)
                    if lam_align > 0.0:
                        align_term = lam_align * alignment(z_norm, y.long(), alpha=2)
                        loss = loss + align_term
                    if lam_uniform > 0.0:
                        uniform_term = lam_uniform * uniformity(z_norm, t=uni_t)
                        loss = loss + uniform_term

            # backward
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            total_grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    g = p.grad.detach()
                    total_grad_norm += float(g.norm(2).item() ** 2)
            total_grad_norm = math.sqrt(total_grad_norm)
            if args.clip_grad and args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer); scaler.update()
            if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR): scheduler.step()

            bs = y.size(0)
            running  += float(loss.detach().item()) * bs
            n_seen   += bs
            main_sum += float(main_loss.detach().item()) * bs
            auxp_sum += float(aux_pred_term.detach().item()) * bs
            auxc_sum += float(aux_con_term.detach().item()) * bs
            dat_sum  += float(dat_term.detach().item()) * bs
            qual_sum += float(qual_term.detach().item()) * bs
            attn_sum += float(attn_term.detach().item()) * bs
            align_sum+= float(align_term.detach().item()) * bs
            uniform_sum += float(uniform_term.detach().item()) * bs
            cons_sum += float(cons_term.detach().item()) * bs
            if (global_step % getattr(args,"log_every",100)) == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                LOG.info(f"[step {global_step}] lr={lr_now:.6g} "
                        f"loss={float(loss):.4f} main={float(main_loss):.4f} "
                        f"gradL2={total_grad_norm:.3f}")
            global_step += 1

        upd2 = w2 = 0.0
        for p, p_prev in zip(model.head.parameters(), head_prev):
            dp = (p.data.detach() - p_prev).float()
            upd2 += float(dp.norm(2).item() ** 2)
            w2   += float(p_prev.float().norm(2).item() ** 2)
        upd_ratio = math.sqrt(upd2) / max(1e-12, math.sqrt(w2))
        del head_prev

        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        # log ogni N step per non inondare


        train_loss = running / max(n_seen, 1)
        LOG.info(f"[{epoch:03d}|breakdown] main={main_sum/max(n_seen,1):.4f} "
                f"cons={cons_sum/max(n_seen,1):.4f} "
                f"auxP={auxp_sum/max(n_seen,1):.4f} auxC={auxc_sum/max(n_seen,1):.4f} "
                f"dat={dat_sum/max(n_seen,1):.4f} qual={qual_sum/max(n_seen,1):.4f} "
                f"attn={attn_sum/max(n_seen,1):.4f} align={align_sum/max(n_seen,1):.4f} "
                f"uniform={uniform_sum/max(n_seen,1):.4f} | bad_pad={bad_pad}")


        # --- Validation ---
        val_logits, val_y, val_trk, val_vid = M.collect_logits(
            model, val_loader, device, use_tqdm=getattr(args,"tqdm",False), desc="Val",
            smooth_alpha=getattr(args,"test_feature_smooth_alpha",0.0),
            tta_quality_n=getattr(args,"tta_quality_n",0), return_ids=True
        )

        if eval_agg == "track_median":
            val_probs_use, val_y_use = _agg_person_from_logits_for_eval(val_logits, val_y, val_trk)
        elif eval_agg == "track_mean":
            ztmp = torch.tensor(val_logits); ytmp = torch.tensor(val_y); trk = torch.tensor(val_trk)
            _, p_person, counts, order = _group_mean_probs_from_logits(ztmp, trk)
            ys = ytmp[order]
            y_person = torch.stack([(c.float().mean() >= 0.5).float() for c in torch.split(ys, counts.tolist())])
            val_probs_use, val_y_use = p_person.numpy(), y_person.numpy()
        elif eval_agg == "video_or_mean":
            ztmp = torch.tensor(val_logits); ytmp = torch.tensor(val_y)
            trk = torch.tensor(val_trk);  vid = torch.tensor(val_vid)
            _, p_person, counts, order = _group_mean_probs_from_logits(ztmp, trk)
            ys = ytmp[order]
            y_person = torch.stack([(c.float().mean() >= 0.5).float() for c in torch.split(ys, counts.tolist())])
            vid_s = vid[order]; vid_chunks = torch.split(vid_s, counts.tolist())
            vid_per_person = torch.stack([c[0] for c in vid_chunks])
            o2 = torch.argsort(vid_per_person)
            vid2, pp2, yp2 = vid_per_person[o2], p_person[o2].clamp(1e-6,1-1e-6), y_person[o2]
            uniq_v, cnt_v = torch.unique_consecutive(vid2, return_counts=True)
            pp_chunks = torch.split(pp2, cnt_v.tolist()); yp_chunks = torch.split(yp2, cnt_v.tolist())
            p_video = torch.stack([1 - torch.exp(torch.log1p(-c).sum()).clamp_min(1e-12) for c in pp_chunks])
            y_video = torch.stack([c.max() for c in yp_chunks]).float()
            val_probs_use, val_y_use = p_video.numpy(), y_video.numpy()
        else:
            val_probs_use, val_y_use = 1.0/(1.0+np.exp(-val_logits)), val_y

        val_probs_use = np.nan_to_num(val_probs_use, nan=0.5, posinf=1.0, neginf=0.0)

        try:
            val_auc   = float(skm.roc_auc_score(val_y_use, val_probs_use))
            val_prauc = float(skm.average_precision_score(val_y_use, val_probs_use))
        except Exception:
            val_auc, val_prauc = float("nan"), float("nan")

        # per-tech (clip-level, diagnostica)
        try:
            val_probs_full = 1.0/(1.0+np.exp(-val_logits))
            techs = np.array(getattr(val_ds, "tech_names", ["unknown"]*len(val_probs_full)))
            import collections
            bucket = collections.defaultdict(lambda: ([], []))
            for p,y_,t in zip(val_probs_full, val_y, techs):
                bucket[t][0].append(p); bucket[t][1].append(y_)
            for t,(pv,yv) in bucket.items():
                pv = np.asarray(pv); yv = np.asarray(yv)
                if len(np.unique(yv))<2: continue
                auc_t = skm.roc_auc_score(yv, pv); ap_t = skm.average_precision_score(yv, pv)
                LOG.info(f"[VAL per-tech] {t} N={len(yv)} AUC={auc_t:.3f} PR-AUC={ap_t:.3f}")
        except Exception as e:
            LOG.warning(f"per-tech logging skipped: {e}")

        val_logits_use = _np_logit(val_probs_use)
        m05 = M.metrics_from_logits(val_logits_use, val_y_use, threshold=0.5)

        best_t, best_stats = threshold_from_roc(val_probs_use, val_y_use, metric=args.es_metric,
                                                target_fpr=getattr(args, "target_fpr", None))
        metric_value = {"youden": best_stats["youden"], "balacc": best_stats["balacc"],
                        "acc": best_stats["acc"], "f1": best_stats["f1"], "auc": val_auc}.get(args.es_metric, best_stats["youden"])
        LOG.info(f"[ES] metric={args.es_metric} value={metric_value:.4f} best_so_far={best_metric:.4f}")

        # log riepilogo epoca
        cur_lr = optimizer.param_groups[0]["lr"]
        LOG.info(f"[{epoch:03d}] train={train_loss:.4f} | lr={cur_lr:.6f} | "
                 f"valAUC={val_auc:.4f} PR-AUC={val_prauc:.4f} | "
                 f"@0.5 acc={m05['acc']:.3f} TPR={m05['TPR']:.3f} FPR={m05['FPR']:.3f} | "
                 f"@best({args.es_metric}) t={best_t:.2f} acc={best_stats['acc']:.3f} "
                 f"TPR={best_stats['TPR']:.3f} FPR={best_stats['FPR']:.3f} balacc={best_stats['balacc']:.3f} "
                 f"youden={best_stats['youden']:.3f} f1={best_stats['f1']:.3f}")
        
        
        if metric_value > best_metric:
            best_metric = metric_value; best_thresh = float(best_t)
            if best_ckpt_path:
                _save_checkpoint(best_ckpt_path, model, optimizer, scheduler, epoch, best_metric, best_thresh)

        if stopper(metric_value, epoch): break

    # --- ricarica best ---
    if args.out and os.path.isfile(os.path.join(args.out, "best.pt")):
        ck = torch.load(os.path.join(args.out, "best.pt"), map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); best_thresh = float(ck.get("best_thresh", best_thresh))
        LOG.info(f"Loaded best checkpoint, threshold={best_thresh:.3f}")

    # --- temp scaling su VAL ---
    val_loader = DataLoader(val_ds, batch_size=args.batch_eval, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    val_logits, val_y, val_trk, val_vid = M.collect_logits(
        model, val_loader, device, use_tqdm=getattr(args,"tqdm",False), desc="Val",
        smooth_alpha=getattr(args,"test_feature_smooth_alpha",0.0),
        tta_quality_n=getattr(args,"tta_quality_n",0), return_ids=True
    )
    LOG.info(f"[VAL] N_clips={len(val_logits)} agg={eval_agg}")

    T_star = fit_temperature_on_val(val_logits, val_y, device)
    val_logits_T = val_logits / max(T_star, 1e-6)

    if eval_agg == "track_median":
        val_probs_cal, val_y_use = _agg_person_from_logits_for_eval(val_logits_T, val_y, val_trk)
    elif eval_agg == "track_mean":
        zt = torch.tensor(val_logits_T); yt = torch.tensor(val_y); trk = torch.tensor(val_trk)
        _, p_person, counts, order = _group_mean_probs_from_logits(zt, trk)
        ys = yt[order]
        y_person = torch.stack([(c.float().mean() >= 0.5).float() for c in torch.split(ys, counts.tolist())])
        val_probs_cal, val_y_use = p_person.numpy(), y_person.numpy()
    elif eval_agg == "video_or_median":
        val_probs_cal, val_y_use = _agg_video_from_logits_for_eval(val_logits_T, val_y, val_trk, val_vid)
    elif eval_agg == "video_or_mean":
        zt = torch.tensor(val_logits_T); yt = torch.tensor(val_y)
        trk = torch.tensor(val_trk); vid = torch.tensor(val_vid)
        _, p_person, counts, order = _group_mean_probs_from_logits(zt, trk)
        ys = yt[order]
        y_person = torch.stack([(c.float().mean() >= 0.5).float() for c in torch.split(ys, counts.tolist())])
        vid_s = vid[order]; vid_chunks = torch.split(vid_s, counts.tolist()); vid_per_person = torch.stack([c[0] for c in vid_chunks])
        o2 = torch.argsort(vid_per_person)
        vid2, pp2, yp2 = vid_per_person[o2], p_person[o2].clamp(1e-6,1-1e-6), y_person[o2]
        uniq_v, cnt_v = torch.unique_consecutive(vid2, return_counts=True)
        pp_chunks = torch.split(pp2, cnt_v.tolist()); yp_chunks = torch.split(yp2, cnt_v.tolist())
        p_video = torch.stack([1 - torch.exp(torch.log1p(-c).sum()).clamp_min(1e-12) for c in pp_chunks])
        y_video = torch.stack([c.max() for c in yp_chunks]).float()
        val_probs_cal, val_y_use = p_video.numpy(), y_video.numpy()
    else:
        val_probs_cal, val_y_use = 1.0/(1.0+np.exp(-val_logits_T)), val_y

    LOG.info(f"[TEMP SCALE] Calibrated T* = {T_star:.3f}")
    if args.out:
        with open(os.path.join(args.out, "temperature.txt"), "w") as f: f.write(f"{T_star:.6f}\n")

    val_probs_cal = np.nan_to_num(val_probs_cal, nan=0.5, posinf=1.0, neginf=0.0)
    best_t_cal, best_stats_cal = threshold_from_roc(val_probs_cal, val_y_use, metric=args.es_metric,
                                                    target_fpr=getattr(args, "target_fpr", None))
    if args.out:
        with open(os.path.join(args.out, "best_threshold_calibrated.txt"), "w") as f:
            f.write(f"{best_t_cal:.6f}\n")
    LOG.info(f"[TEMP SCALE] Best threshold (val) ricalibrato: t={best_t_cal:.3f} "
             f"| acc={best_stats_cal['acc']:.3f} balacc={best_stats_cal['balacc']:.3f} "
             f"f1={best_stats_cal['f1']:.3f} youden={best_stats_cal['youden']:.3f}")
    LOG.info(f"[TEMP SCALE] T*={T_star:.3f} | using val-calibrated threshold t={best_t_cal:.3f}")


    # --- Test finale ---
    test_loader = DataLoader(test_ds, batch_size=args.batch_eval, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    test_logits, test_y, test_trk, test_vid = M.collect_logits(
        model, test_loader, device, use_tqdm=getattr(args,"tqdm",False), desc="Test",
        smooth_alpha=getattr(args,"test_feature_smooth_alpha",0.0),
        tta_quality_n=getattr(args,"tta_quality_n",0), return_ids=True
    )
    test_logits = test_logits / max(T_star, 1e-6)

    if eval_agg == "track_median":
        test_probs_use, test_y_use = _agg_person_from_logits_for_eval(test_logits, test_y, test_trk)
    elif eval_agg == "track_mean":
        zt = torch.tensor(test_logits); yt = torch.tensor(test_y); trk = torch.tensor(test_trk)
        _, p_person, counts, order = _group_mean_probs_from_logits(zt, trk)
        ys = yt[order]
        test_y_use = torch.stack([(c.float().mean() >= 0.5).float() for c in torch.split(ys, counts.tolist())]).numpy()
        test_probs_use = p_person.numpy()
    elif eval_agg == "video_or_median":
        test_probs_use, test_y_use = _agg_video_from_logits_for_eval(test_logits, test_y, test_trk, test_vid)
    elif eval_agg == "video_or_mean":
        zt = torch.tensor(test_logits); yt = torch.tensor(test_y)
        trk = torch.tensor(test_trk); vid = torch.tensor(test_vid)
        _, p_person, counts, order = _group_mean_probs_from_logits(zt, trk)
        ys = yt[order]
        y_person = torch.stack([(c.float().mean() >= 0.5).float() for c in torch.split(ys, counts.tolist())])
        vid_s = vid[order]; vid_chunks = torch.split(vid_s, counts.tolist()); vid_per_person = torch.stack([c[0] for c in vid_chunks])
        o2 = torch.argsort(vid_per_person)
        vid2, pp2, yp2 = vid_per_person[o2], p_person[o2].clamp(1e-6,1-1e-6), y_person[o2]
        uniq_v, cnt_v = torch.unique_consecutive(vid2, return_counts=True)
        pp_chunks = torch.split(pp2, cnt_v.tolist()); yp_chunks = torch.split(yp2, cnt_v.tolist())
        p_video = torch.stack([1 - torch.exp(torch.log1p(-c).sum()).clamp_min(1e-12) for c in pp_chunks])
        y_video = torch.stack([c.max() for c in yp_chunks]).float()
        test_probs_use, test_y_use = p_video.numpy(), y_video.numpy()
    else:
        test_probs_use, test_y_use = 1.0/(1.0+np.exp(-test_logits)), test_y

    test_probs_use = np.nan_to_num(test_probs_use, nan=0.5, posinf=1.0, neginf=0.0)
    test_logits_use = _np_logit(test_probs_use)
    test_m = M.metrics_from_logits(test_logits_use, test_y_use, threshold=float(best_t_cal))

    LOG.info(f"[TEST|TEMP] @t={best_t_cal:.2f} acc={test_m['acc']:.4f} f1={test_m['f1']:.4f} "
             f"balacc={test_m['balacc']:.4f} youden={test_m['youden']:.4f} "
             f"TPR={test_m['TPR']:.4f} FPR={test_m['FPR']:.4f} "
             f"ROC-AUC={test_m['roc_auc']:.4f} PR-AUC={test_m['pr_auc']:.4f}")

    # opzionale: AUC val ricalcolata dopo calibrazione (per coerenza col tuo dict)
    try:
        val_auc_cal = float(skm.roc_auc_score(val_y_use, val_probs_cal)) if len(np.unique(val_y_use))==2 else float("nan")
    except Exception:
        val_auc_cal = float("nan")

    return {"best_thresh": float(best_t_cal), "T_star": float(T_star),
            "val_auc": float(val_auc_cal), "test_metrics": test_m}

