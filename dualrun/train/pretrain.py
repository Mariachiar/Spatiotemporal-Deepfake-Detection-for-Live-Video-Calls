#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pretraining LMK-only su Vox (self-supervised dinamico).
Obiettivo: real vs time-shuffled (stessa sequenza con frame permutati).
Stack:
- vox_index.build_index -> split per speaker
- vox_ds.VoxLmkDataset  -> [T,D] landmark float32
- BranchEncoder(in_dim=...) -> pooling
- Testina binaria (BCEWithLogits)

Comando tipico:
python -u pretrain_lmk.py \
  --data datasets/processed_dataset/partial_extract/dev \
  --out runs/pretrain_lmk --epochs 80 --batch 256 --tqdm
"""

import os, sys, math, argparse, logging, random
from typing import Tuple
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# import locali
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.vox_index import build_index
from data.vox_ds import VoxLmkDataset, collate_pad
from model.dual_encoder import BranchEncoder  # deve avere __init__(in_dim=...)

LOG = logging.getLogger("pretrain.lmk")

# ------------------------- util -------------------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def setup_logging(level="INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    LOG.info("Logging initialized: %s", level)

def lengths_to_mask(lengths: torch.Tensor, T: int, device) -> torch.Tensor:
    lengths = torch.clamp(lengths, min=1)
    ar = torch.arange(T, device=device).unsqueeze(0)
    return ar >= lengths.unsqueeze(1)  # True=PAD

# ------------------------- modello -------------------------
class LMKDisc(nn.Module):
    """BranchEncoder + testa binaria."""
    def __init__(self, in_dim: int, d_model=256, nhead=4, num_layers=4, dim_ff=512, dropout=0.2):
        super().__init__()
        mlp_ratio = dim_ff / d_model           # se usi argomenti simili a quelli del detector
        self.enc = BranchEncoder(
            input_dim=in_dim,                  # dim delle feature LMK dal dataset Vox
            d_model=d_model,
            depth=num_layers,
            heads=nhead,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            pool_tau=0.7,                      # o esponilo come argomento
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x, key_padding_mask=None):
        z = self.enc(x, key_padding_mask=key_padding_mask)   # (B,D) pooled
        return self.head(z).squeeze(-1)                      # (B,)

# ------------------------- cli -------------------------
def parse_args():
    ap = argparse.ArgumentParser("Pretraining LMK-only su Vox (real vs time-shuffled)")
    ap.add_argument("--data", required=True, help="Root Vox preprocessata con lmk_features.npy")
    ap.add_argument("--out", required=True, help="Cartella output per best.pt e args.json")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--ff_dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.20)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--clip_grad", type=float, default=1.0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--tqdm", action="store_true")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--log_level", default="INFO")
    ap.add_argument("--val_ratio", type=float, default=0.05, help="Quota speaker per validation in build_index")
    ap.add_argument("--tmin", type=int, default=8, help="Min frame per clip in build_index")
    return ap.parse_args()

# ------------------------- train utils -------------------------
def make_shuffled(X: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
    """
    Permuta l'asse temporale separatamente per ogni sequenza, rispettando il padding.
    X: [B,T,D], pad: [B,T] con True=PAD
    """
    B, T, D = X.shape
    Xs = X.clone()
    for b in range(B):
        valid = (~pad[b]).nonzero(as_tuple=False).flatten()
        if valid.numel() <= 1:  # niente da permutare
            continue
        order = valid[torch.randperm(valid.numel(), device=X.device)]
        Xs[b, valid] = X[b, order]
    return Xs

@torch.no_grad()
def eval_step(model, dl, device, use_tqdm=False) -> Tuple[float, float]:
    model.eval()
    crit = nn.BCEWithLogitsLoss()

    tot, n, acc_c = 0.0, 0, 0
    it = tqdm(dl, desc="val", leave=True, dynamic_ncols=True, disable=not use_tqdm, file=sys.stdout) if use_tqdm else dl
    for X, pad in it:
        X = X.to(device, non_blocking=True)
        pad = pad.to(device, non_blocking=True)
        mask = pad
        # real
        logit_r = model(X, key_padding_mask=mask)
        y_r = torch.ones(X.size(0), device=device)
        # shuffled
        Xs = make_shuffled(X, pad)
        logit_s = model(Xs, key_padding_mask=mask)
        y_s = torch.zeros(X.size(0), device=device)
        # concat
        logits = torch.cat([logit_r, logit_s], 0)
        y = torch.cat([y_r, y_s], 0)
        loss = crit(logits, y)
        tot += float(loss.item()) * y.numel()
        n += y.numel()
        acc_c += int(((logits > 0).long() == y.long()).sum().item())
    return tot / max(1, n), acc_c / max(1, n)

# ------------------------- main -------------------------
import sys
def main():
    args = parse_args()
    setup_logging(args.log_level)
    set_seed(args.seed)

    os.makedirs(args.out, exist_ok=True)
    # salva args
    with open(os.path.join(args.out, "args_pretrain_lmk.json"), "w") as f:
        import json; json.dump(vars(args), f, indent=2)

    # index + dataset
    tr_list, va_list = build_index(args.data, tmin=args.tmin, val_speakers_ratio=args.val_ratio)
    if not tr_list or not va_list:
        raise SystemExit("Indice vuoto: controlla che esistano file lmk_features.npy non vuoti.")
    ds_tr = VoxLmkDataset(tr_list, time_warp=(0.9, 1.1))
    ds_va = VoxLmkDataset(va_list, time_warp=(1.0, 1.0))
    # dimensione input
    import numpy as np
    D = np.load(tr_list[0], mmap_mode="r").shape[1]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LMKDisc(
        in_dim=D, d_model=args.d_model, nhead=args.heads,
        num_layers=args.layers, dim_ff=args.ff_dim, dropout=args.dropout
    ).to(device)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True,
                       num_workers=args.num_workers, pin_memory=(device=="cuda"),
                       collate_fn=collate_pad, persistent_workers=(args.num_workers>0 and device=="cuda"))
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False,
                       num_workers=args.num_workers, pin_memory=(device=="cuda"),
                       collate_fn=collate_pad, persistent_workers=(args.num_workers>0 and device=="cuda"))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device=="cuda"))
    crit = nn.BCEWithLogitsLoss()

    best_loss, best_acc, best_ep = float("inf"), 0.0, -1
    ckpt_path = os.path.join(args.out, "best.pt")

    for ep in range(1, args.epochs+1):
        model.train()
        it = tqdm(dl_tr, desc=f"train {ep}/{args.epochs}", unit="batch",
                  leave=True, dynamic_ncols=True, disable=not args.tqdm, file=sys.stdout) if args.tqdm else dl_tr
        for X, pad in it:
            X = X.to(device, non_blocking=True)
            pad = pad.to(device, non_blocking=True)
            mask = pad
            # real
            with torch.cuda.amp.autocast(enabled=(args.amp and device=="cuda")):
                logit_r = model(X, key_padding_mask=mask)
                y_r = torch.ones(X.size(0), device=device)
                loss_r = crit(logit_r, y_r)
            # shuffled
            Xs = make_shuffled(X, pad)
            with torch.cuda.amp.autocast(enabled=(args.amp and device=="cuda")):
                logit_s = model(Xs, key_padding_mask=mask)
                y_s = torch.zeros(X.size(0), device=device)
                loss_s = crit(logit_s, y_s)
                loss = (loss_r + loss_s) * 0.5

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if args.clip_grad and args.clip_grad > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(opt); scaler.update()

        # validation
        val_loss, val_acc = eval_step(model, dl_va, device, use_tqdm=args.tqdm)
        LOG.info(f"[{ep:03d}] val_loss={val_loss:.6f}  val_acc={val_acc:.4f}")

        if val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss):
            best_loss, best_acc, best_ep = val_loss, val_acc, ep
            torch.save({"epoch": ep, "model": model.state_dict()}, ckpt_path)
            LOG.info("Saved best -> %s", ckpt_path)

    LOG.info("Best epoch: %d  val_loss=%.6f  val_acc=%.4f", best_ep, best_loss, best_acc)

if __name__ == "__main__":
    main()
