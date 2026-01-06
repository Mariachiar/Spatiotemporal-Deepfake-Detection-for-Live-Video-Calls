import math, logging, torch, torch.nn as nn
from typing import Optional, Union, Tuple
import torch.nn.functional as F


LOG = logging.getLogger("dual_model")

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lam: float): ctx.lam = float(lam); return x.view_as(x)
    @staticmethod
    def backward(ctx, g: torch.Tensor): return -ctx.lam * g, None
def grad_reverse(x: torch.Tensor, lam: float) -> torch.Tensor: return GradReverse.apply(x, lam)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float=0.0, max_len: int=2048, batch_first: bool=True):
        super().__init__(); self.dropout = nn.Dropout(dropout); self.batch_first = batch_first
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:,0::2] = torch.sin(pos*div); pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.batch_first: x = x + self.pe[:, :x.size(1), :]
        else: x = x + self.pe[:x.size(0), :].unsqueeze(1)
        return self.dropout(x)

# --- AttentionPooling: soft attention + pesi opzionali ---
class AttentionPooling(nn.Module):
    def __init__(self, d_model: int, k: Optional[int]=None, tau: float=1.0):
        super().__init__()
        self.v = nn.Parameter(torch.randn(d_model))
        self.k = k
        self.tau = max(float(tau), 1e-3)

    def forward(self, x, key_padding_mask=None, return_weights=False):
        # x: (B,T,D)
        scores = (x @ self.v) / self.tau                     # (B,T)
        if key_padding_mask is not None:
            neg_large = torch.finfo(scores.dtype).min        # OK in FP16 e FP32
            scores = scores.masked_fill(key_padding_mask, neg_large)

        w = torch.softmax(scores, dim=1)
        pooled = (w.unsqueeze(-1) * x).sum(dim=1)
        return (pooled, w) if return_weights else pooled

def _len2mask(lengths, T, device):
    lengths = lengths.clamp_min(1)
    idx = torch.arange(T, device=device).unsqueeze(0)
    return (idx >= lengths.unsqueeze(1))  # bool


class BranchEncoder(nn.Module):
    def __init__(self, input_dim, d_model=256, depth=4, heads=4, mlp_ratio=2.0, dropout=0.1, pool_tau=0.7):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.ln_in = nn.LayerNorm(d_model)
        self.temporal = nn.ModuleList([
            nn.Conv1d(d_model, d_model, 3, padding=1, groups=d_model, dilation=1),
            nn.Conv1d(d_model, d_model, 3, padding=2, groups=d_model, dilation=2),
            nn.Conv1d(d_model, d_model, 3, padding=4, groups=d_model, dilation=4),
        ])
        self.pointwise = nn.Conv1d(d_model, d_model, 1)
        self.pe = PositionalEncoding(d_model, dropout=0.0, batch_first=True)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=int(d_model*mlp_ratio),
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.pool = AttentionPooling(d_model, k=None, tau=pool_tau)  # tau esposto

        self.d_model = d_model

    def forward(self, x, key_padding_mask=None, return_weights=False, return_seq=False):
        # x: (B,T,D_in)
        h = self.ln_in(self.proj(x))                          # (B,T,D)

        # Δ primo ordine
        diff  = h[:, 1:, :] - h[:, :-1, :]
        delta = torch.cat([torch.zeros_like(h[:, :1, :]), diff], dim=1)  # (B,T,D)

        # passa-alto con MA-5
        h_c   = h.transpose(1, 2)                             # (B,D,T)
        ma    = F.avg_pool1d(h_c, kernel_size=5, stride=1, padding=2)    # (B,D,T)
        highp = (h_c - ma).transpose(1, 2)                    # (B,T,D)

        # mix conservativo
        h = h + 0.5*delta + 0.5*highp                         # (B,T,D)

        # piramide depthwise + skip
        h_c = h.transpose(1, 2)                               # (B,D,T)
        pyr = sum(conv(h_c) for conv in self.temporal)        # (B,D,T)
        h_c = pyr + h_c                                       # residuo
        h_c = F.gelu(self.pointwise(h_c))                     # (B,D,T)
        h   = h_c.transpose(1, 2)                             # (B,T,D)

        # Transformer + pooling
        h = self.pe(h)
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        if return_weights:
            clip, w = self.pool(h, key_padding_mask=key_padding_mask, return_weights=True)
        else:
            clip = self.pool(h, key_padding_mask=key_padding_mask, return_weights=False); w = None

        if return_seq and return_weights: return clip, w, h
        if return_seq:                    return clip, h
        if return_weights:                return clip, w
        return clip


class DualEncoderAU_LMK(nn.Module):
    def __init__(self, au_dim, lmk_dim, d_model=256, depth=4, heads=4, mlp_ratio=2.0,
                 dropout=0.1, proj_dim=128, use_dat=False, domain_classes=0, pool_tau: float = 1.0):
        super().__init__()
        self.au_enc  = BranchEncoder(au_dim,  d_model, depth, heads, mlp_ratio, dropout, pool_tau)
        self.lmk_enc = BranchEncoder(lmk_dim, d_model, depth, heads, mlp_ratio, dropout, pool_tau)
        # dual_encoder.py
        self.head = nn.Sequential(
            nn.LayerNorm(2*d_model),
            nn.Linear(2*d_model, 2*d_model),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(2*d_model, 1),
        )
        #nn.init.constant_(self.head[-1].bias, -2.0)
        self.use_dat = use_dat
        self.domain_head = nn.Linear(2*d_model, domain_classes) if use_dat and domain_classes>0 else None

        # --- teste ausiliarie ---
        self.au_from_lmk = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, au_dim))
        self.proj_au  = nn.Linear(d_model, proj_dim)
        self.proj_lmk = nn.Linear(d_model, proj_dim)

    @staticmethod
    # dentro model (o utils), sostituisci/aggiungi:
    def lengths_to_mask(lengths: torch.Tensor, S: int, device) -> torch.Tensor:
        """
        Input:
        lengths: (N,) con lunghezze int   oppure (N,S) con 1=valido, 0=pad
        S: sequence length
        Output:
        mask: (N,S) bool con True = PAD
        """
        lengths = lengths.to(device)
        if lengths.dim() == 2:
            mask = (lengths == 0)
        elif lengths.dim() == 1:
            N = lengths.size(0)
            ar = torch.arange(S, device=device).expand(N, S)
            mask = ar >= lengths.view(-1, 1)
        else:
            raise ValueError(f"lengths dim must be 1 or 2, got {lengths.dim()}")
        return mask.bool()


    def forward(self, A, L, lengths=None, need_aux=False, return_z=False, return_seq=False,
                dat_lambda: float=0.0):
        # A,L: [B,T,D], lengths: [B]
        B,T,_ = A.shape
        pad = self.lengths_to_mask(lengths, T, A.device) if lengths is not None else None
        if pad is not None and pad.dim()==2:
            all_pad = pad.all(dim=1)
            if all_pad.any():
                pad = pad.clone()
                pad[all_pad, 0] = False                 # sblocca il token 0 se tutto pad

        if need_aux or return_seq:
            za_clip, za_w, za_seq = self.au_enc(A, key_padding_mask=pad, return_weights=True, return_seq=True)
            zl_clip, zl_w, zl_seq = self.lmk_enc(L, key_padding_mask=pad, return_weights=True, return_seq=True)
        else:
            za_clip = self.au_enc(A, key_padding_mask=pad)
            zl_clip = self.lmk_enc(L, key_padding_mask=pad)

        z = torch.cat([za_clip, zl_clip], dim=-1)
        bin_logits = self.head(z).squeeze(-1)

        dom_logits = None
        if self.use_dat and dat_lambda > 0:
            z_rev = grad_reverse(z, dat_lambda)
            dom_logits = self.domain_head(z_rev)


        out = {"bin_logits": bin_logits, "dom_logits": dom_logits}

        if return_z:
            out["z"] = z
        if return_seq:
            out["za_seq"] = za_seq
            out["zl_seq"] = zl_seq
            out["weights"] = {"au": za_w, "lmk": zl_w}
        if need_aux:
            # LMK->AU per-frame
            B,T,D = zl_seq.shape
            out["au_pred"] = self.au_from_lmk(zl_seq.reshape(B*T, D)).reshape(B, T, -1)
            # proiezioni contrastive
            out["proj_au"]  = self.proj_au(za_seq)    # [B,T,P]
            out["proj_lmk"] = self.proj_lmk(zl_seq)   # [B,T,P]
            out["pad_mask"] = pad                     # [B,T] bool
        return out

