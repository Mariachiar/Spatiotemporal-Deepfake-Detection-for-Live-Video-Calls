#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, sys, json, logging, random, torch
from typing import Dict, Any, List, Tuple

# path locali
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.setup import setup_logging, set_seed
from utils.io import ensure_index_json, save_json
from data.dataset_dual import DualFeaturesClipDataset
from data.dataset_regen import DualVideoRegenDataset
from model.dual_encoder import DualEncoderAU_LMK
from train.engine import train
from opts import parse_opts

LOG = logging.getLogger("dualrun")

# ------------------------------ util ------------------------------

def _read_paths_list(p: str | None) -> List[str]:
    if not p: return []
    if p.lower().endswith(".json"):
        xs = json.load(open(p, "r", encoding="utf-8"))
        return [x.strip() for x in xs if x and os.path.isfile(x.strip())]
    out = []
    with open(p, "r", encoding="utf-8-sig") as f:
        for l in f:
            s = l.strip().strip('"').lstrip("\ufeff")
            if s and os.path.isfile(s):
                out.append(s)
    # univoci e ordinati
    return sorted(set(out))

def _mk_feat_ds(clip_dirs: List[str] | None, T: int, is_train: bool, random_crop: bool, ds_kwargs: Dict[str, Any]):
    kw = dict(ds_kwargs)
    kw.update(dict(is_train=is_train, random_crop=random_crop, validate=False))
    if clip_dirs is not None:
        return DualFeaturesClipDataset(clip_dirs=clip_dirs, T=T, **kw)
    raise RuntimeError("clip_dirs mancante per dataset features.")

def _mk_video_ds(video_paths: List[str], T: int, is_train: bool, zscore: str, norm_stats_path: str | None,
                 jpeg_q: Tuple[int,int], scale: Tuple[float,float], offcenter: float, mblur_k: Tuple[int,int]):
    # niente degradazioni per val/test
    if not is_train:
        jpeg_q, scale, offcenter, mblur_k = (90,90), (1.0,1.0), 0.0, (0,0)
    return DualVideoRegenDataset(
        video_paths=video_paths, T=T, zscore=zscore, norm_stats_path=norm_stats_path,
        jpeg_q=jpeg_q, scale=scale, offcenter=offcenter, mblur_k=mblur_k, is_train=is_train
    )

def _split_80_10_10(full: DualFeaturesClipDataset) -> Tuple[List[str], List[str], List[str]]:
    N = len(full); idx = list(range(N))
    random.Random(42).shuffle(idx)
    n_train = int(0.8*N); n_val = int(0.1*N)
    def subset(idxs): return [full.clip_dirs[i] for i in idxs]
    return subset(idx[:n_train]), subset(idx[n_train:n_train+n_val]), subset(idx[n_train+n_val:])


# ------------------------------ main ------------------------------

def main():
    args = parse_opts()
    setup_logging(args.log_level)
    set_seed(getattr(args, "seed", 123))

    # comuni per dataset features
    ds_kwargs = dict(
        dtype=torch.float32,
        mmap=not getattr(args, "no_mmap", False),
        is_train=False,
        random_crop=False,
        zscore=getattr(args, "zscore", "none"),
        zscore_apply=getattr(args, "zscore_apply", "both"),
        return_tech=True,
        allow_missing_au=False,
        safe_mmap=True,
        aug_noise_au=0.0, aug_noise_lmk=0.0, aug_tdrop=0.0, stitch_k=getattr(args, "stitch_k", 1),
    )

    # normalizzazione globale
    norm_stats_path = getattr(args, "norm_stats", None) or os.getenv("NORM_STATS", None)
    zscore = getattr(args, "zscore", "clip")

    # degradazioni train video
    regen_jpeg = tuple(getattr(args, "regen_jpeg", (3, 25)))
    regen_scale = tuple(getattr(args, "regen_scale", (0.3, 0.8)))
    regen_offcenter = float(getattr(args, "regen_offcenter", 0.06))
    regen_mblur = tuple(getattr(args, "regen_mblur", (0, 9)))

    # liste video (se presenti, usiamo rigenerazione on-the-fly)
    train_vlist = _read_paths_list(getattr(args, "train_videos_list", None))
    val_vlist   = _read_paths_list(getattr(args, "val_videos_list", None))
    test_vlist  = _read_paths_list(getattr(args, "test_videos_list", None))

    # costruzione dataset
    if train_vlist:
        train_ds = _mk_video_ds(
            video_paths=train_vlist, T=args.T, is_train=True, zscore=zscore, norm_stats_path=norm_stats_path,
            jpeg_q=regen_jpeg, scale=regen_scale, offcenter=regen_offcenter, mblur_k=regen_mblur
        )
    else:
        # fallback: features via index/root
        index_json = ensure_index_json(args.index, args.data, args.T, **ds_kwargs) if args.index else None
        if index_json and all(k in index_json for k in ("train","val","test")):
            train_ds = _mk_feat_ds(index_json["train"], args.T, True, True, ds_kwargs)
        else:
            full = DualFeaturesClipDataset(root_dir=args.data, T=args.T, **ds_kwargs)
            tr, va, te = _split_80_10_10(full)
            train_ds = _mk_feat_ds(tr, args.T, True, True, ds_kwargs)

    if val_vlist:
        val_ds = _mk_video_ds(
            video_paths=val_vlist, T=args.T, is_train=False, zscore=zscore, norm_stats_path=norm_stats_path,
            jpeg_q=regen_jpeg, scale=regen_scale, offcenter=regen_offcenter, mblur_k=regen_mblur
        )
    else:
        if 'index_json' not in locals():
            index_json = ensure_index_json(args.index, args.data, args.T, **ds_kwargs) if args.index else None
        if index_json and "val" in index_json:
            val_ds = _mk_feat_ds(index_json["val"], args.T, False, False, ds_kwargs)
        else:
            if 'full' not in locals():
                full = DualFeaturesClipDataset(root_dir=args.data, T=args.T, **ds_kwargs)
                tr, va, te = _split_80_10_10(full)
            val_ds = _mk_feat_ds(va, args.T, False, False, ds_kwargs)

    if test_vlist:
        test_ds = _mk_video_ds(
            video_paths=test_vlist, T=args.T, is_train=False, zscore=zscore, norm_stats_path=norm_stats_path,
            jpeg_q=regen_jpeg, scale=regen_scale, offcenter=regen_offcenter, mblur_k=regen_mblur
        )
    else:
        if index_json and "test" in index_json:
            test_ds = _mk_feat_ds(index_json["test"], args.T, False, False, ds_kwargs)
        else:
            if 'full' not in locals():
                full = DualFeaturesClipDataset(root_dir=args.data, T=args.T, **ds_kwargs)
                tr, va, te = _split_80_10_10(full)
            test_ds = _mk_feat_ds(te, args.T, False, False, ds_kwargs)

    # intensità aug feature-space solo per train su FEATURE DS (non su video)
    if isinstance(train_ds, DualFeaturesClipDataset):
        train_ds.aug_noise_au  = float(getattr(args, "aug_noise_au", 0.0))
        train_ds.aug_noise_lmk = float(getattr(args, "aug_noise_lmk", 0.0))
        train_ds.aug_tdrop     = float(getattr(args, "aug_tdrop", 0.0))
        if hasattr(train_ds, "lmk_affine_deg"):
            train_ds.lmk_affine_deg    = float(getattr(args, "lmk_affine_deg", 0.0))
            train_ds.lmk_dropout_p     = float(getattr(args, "lmk_dropout_p", 0.0))
            train_ds.lmk_temporal_alpha= float(getattr(args, "lmk_temporal_alpha", 0.0))
            train_ds.au_dropout_p      = float(getattr(args, "au_dropout_p", 0.0))
            train_ds.au_temporal_alpha = float(getattr(args, "au_temporal_alpha", 0.0))
            train_ds.lmk_add_deltas    = bool(getattr(args, "lmk_add_deltas", False))

    # qualità opzionale (solo train se supportato)
    # --- simmetriche: stessa probabilità di "sporcare" real e fake ---
    dirty_p = float(getattr(args, "dirty_p", 0.0))

    train_ds.qual_factorized = True          # label-agnostic
    train_ds.dirty_p = dirty_p               # P(q=1) uguale per tutti
    train_ds.clean_fake_p = 1.0              # disattiva ramo label-dipendente
    train_ds.clean_real_p = 1.0
    train_ds.return_quality = bool(dirty_p > 0.0)

    # consenti temporal drop anche sui REAL
    train_ds.protect_real_for_consistency = False


    # DAT classi di dominio
    domain_classes = int(getattr(train_ds, "n_domains", 0))
    LOG.info(f"[DAT] domain_classes={domain_classes}")

    # modello
    mlp_ratio = float(getattr(args, "ff_dim")) / float(getattr(args, "d_model"))
    model = DualEncoderAU_LMK(
        au_dim=getattr(train_ds, "au_dim", 36),
        lmk_dim=getattr(train_ds, "lmk_dim", 132),
        d_model=getattr(args, "d_model"),
        depth=getattr(args, "layers"),
        heads=getattr(args, "heads"),
        mlp_ratio=mlp_ratio,
        dropout=getattr(args, "dropout"),
        use_dat=bool(getattr(args, "dat", False)),
        domain_classes=domain_classes,
        pool_tau=float(getattr(args, "pool_tau", 1.0)),
    )

    # init opzionale
    if getattr(args, "init", None):
        ck = torch.load(args.init, map_location="cpu")
        sd = ck.get("model", ck)
        msd = model.state_dict()
        keep = {k: v for k, v in sd.items() if k in msd and v.shape == msd[k].shape}
        missing, unexpected = model.load_state_dict(keep, strict=False)
        LOG.info(f"Init parziale: loaded={len(keep)} missing={len(missing)} unexpected={len(unexpected)}")

    # salva args e split
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        save_json(vars(args).copy(), os.path.join(args.out, "args.json"))
        payload = {
            "note": "Split usati per questa run",
            "train_videos": getattr(train_ds, "video_paths", []),
            "val_videos":   getattr(val_ds,   "video_paths", []),
            "test_videos":  getattr(test_ds,  "video_paths", []),
            "train": getattr(train_ds, "clip_dirs", []),
            "val":   getattr(val_ds,   "clip_dirs", []),
            "test":  getattr(test_ds,  "clip_dirs", []),
        }
        save_json(payload, os.path.join(args.out, "splits_used.json"))

    # riepilogo
    LOG.info("==== CONFIG ====")
    for k, v in sorted(vars(args).items()): LOG.info(f"{k:24s} = {v}")
    LOG.info("================")

    # train
    train(model, train_ds, val_ds, test_ds, args, heldout=getattr(args, "heldout_tech", None))

if __name__ == "__main__":
    main()
