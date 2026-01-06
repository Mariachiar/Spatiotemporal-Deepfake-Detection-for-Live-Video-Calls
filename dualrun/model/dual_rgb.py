# -*- coding: utf-8 -*-
import torch, torch.nn as nn, torch.nn.functional as F
from typing import Optional, Tuple

# riusa i tuoi encoder già presenti
from .dual_encoder import BranchEncoder, DualEncoderAU_LMK

# --- Wrapper AltFreezing (RGB) ---
class AltFreezingRGBEncoder(nn.Module):
    """
    Carica il backbone AltFreezing originale e produce embedding per clip.
    Congelato (no grad). Accetta:
      - frames RGB:  Tensor [B, T, 3, H, W]
      - oppure feature pre-estratte: Tensor [B, T, D] (pass-through)
    """
    def __init__(self, backbone: nn.Module, out_dim: int, from_features: bool = False):
        super().__init__()
        self.backbone = backbone if backbone is not None else nn.Identity()
        self.out_dim = int(out_dim)
        self.from_features = bool(from_features)
        # congela tutto
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Ritorna Z_v in shape [B, D] (pool su tempo con mask).
        - Se from_features=True: x è [B, T, D] e fa solo masked mean.
        - Altrimenti: x è [B, T, 3, H, W] -> backbone -> [B, T, D].
        """
        if self.from_features:
            zt = x  # [B, T, D]
        else:
            # backbone deve restituire feature per frame/tempo -> [B, T, D]
            zt = self.backbone(x)  # assicurati che il backbone faccia il reshape corretto

        if key_padding_mask is None:
            return zt.mean(dim=1)

        valid = (~key_padding_mask).float()  # [B, T]
        w = (valid / valid.clamp_min(1e-6).sum(dim=1, keepdim=True)).unsqueeze(-1)
        return (zt * w).sum(dim=1)


class DualEncoderRGB(nn.Module):
    """
    Tri-modale: AU + LMK allenabili, RGB (AltFreezing) congelato.
    Head unica. Allena solo au_enc, lmk_enc, head (+ eventuali teste extra).
    """
    def __init__(self,
                 au_dim: int, lmk_dim: int,
                 vis_dim: int,             # = embedding RGB AltFreezing
                 d_model: int = 256,
                 depth: int = 4, heads: int = 4, ff_dim: int = 768, dropout: float = 0.1,
                 rgb_backbone: Optional[nn.Module] = None,
                 rgb_from_features: bool = True):
        super().__init__()
        self.au_enc  = BranchEncoder(au_dim,  d_model, depth, heads, ff_dim, dropout)
        self.lmk_enc = BranchEncoder(lmk_dim, d_model, depth, heads, ff_dim, dropout)

        # ramo RGB congelato
        self.rgb = AltFreezingRGBEncoder(backbone=rgb_backbone, out_dim=vis_dim,
                                         from_features=rgb_from_features)

        # proiezione opzionale del vettore RGB a d_model per simmetria
        self.rgb_proj = nn.Linear(vis_dim, d_model, bias=False)
        for p in self.rgb_proj.parameters():  # si può anche congelare per massima stabilità
            p.requires_grad = False

        # head unica su 3*d_model
        self.head = nn.Sequential(
            nn.LayerNorm(3*d_model),
            nn.Linear(3*d_model, 2*d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2*d_model, 1)
        )

        # opzionali: domain/quality heads compatibili con il tuo engine
        self.use_dat = False
        self.domain_head = None
        self.quality_head = None

    # utility per compatibilità engine
    @staticmethod
    def lengths_to_mask(t_valid: torch.Tensor, T: int, device: torch.device):
        return DualEncoderAU_LMK.lengths_to_mask(t_valid, T, device)

    def forward(self,
                A: torch.Tensor, L: torch.Tensor,
                V: Optional[torch.Tensor] = None,            # [B,T,D] features o [B,T,3,H,W] frames
                key_padding_mask: Optional[torch.Tensor] = None,
                return_weights: bool = False,
                return_seq: bool = False):
        """
        Output: bin_logits [B]
        Se return_* sono true, ritorna anche pesi/seq dei soli rami AU/LMK come nel tuo DualEncoder.
        """
        # AU/LMK
        if return_weights or return_seq:
            za, wa, za_seq = self.au_enc(A, key_padding_mask=key_padding_mask,
                                         return_weights=True, return_seq=True)
            zl, wl, zl_seq = self.lmk_enc(L, key_padding_mask=key_padding_mask,
                                          return_weights=True, return_seq=True)
        else:
            za = self.au_enc(A, key_padding_mask=key_padding_mask)
            zl = self.lmk_enc(L, key_padding_mask=key_padding_mask)

        # RGB: pooled a vettore e poi proj -> d_model
        zv_clip = self.rgb(V, key_padding_mask=key_padding_mask)  # [B, vis_dim]
        zv = self.rgb_proj(zv_clip)                               # [B, d_model]

        z = torch.cat([za, zl, zv], dim=-1)                       # [B, 3*d_model]
        bin_logits = self.head(z).squeeze(-1)

        if return_weights and return_seq:
            return bin_logits, (wa, wl), (za_seq, zl_seq)
        if return_weights:
            return bin_logits, (wa, wl)
        if return_seq:
            return bin_logits, (za_seq, zl_seq)
        return bin_logits
