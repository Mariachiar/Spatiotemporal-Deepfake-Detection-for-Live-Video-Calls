# remap_pretrain_to_lmk.py
import sys, torch, re
from pathlib import Path

def infer_target_pool_name():
    """
    Legge dual_encoder.py, cerca come si chiama l'attributo di pooling
    dentro BranchEncoder: 'pool' oppure 'pooling'. Default: 'pool'.
    """
    try:
        txt = Path("dualrun/dual_encoder.py").read_text(encoding="utf-8")
    except Exception:
        try:
            txt = Path("dual_encoder.py").read_text(encoding="utf-8")
        except Exception:
            return "pool"  # fallback prudente
    # euristica: se trova 'self.pooling =' preferisci 'pooling'
    if re.search(r"\bself\.pooling\s*=", txt):
        return "pooling"
    if re.search(r"\bself\.pool\s*=", txt):
        return "pool"
    return "pool"  # fallback

def main(src, dst):
    ck = torch.load(src, map_location="cpu")
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    target_pool = infer_target_pool_name()  # 'pool' oppure 'pooling'

    new = {}
    for k, v in sd.items():
        # 1) togli eventuale "module."
        if k.startswith("module."):
            k = k[len("module."):]
        # 2) prendiamo solo il ramo encoder salvato come "enc.*"
        if not k.startswith("enc."):
            continue
        k2 = k.split("enc.", 1)[1]

        # 3) normalizza nome del pooling verso il target
        #    se nel sorgente è 'pooling.' e target è 'pool.' => converti
        #    se nel sorgente è 'pool.'   e target è 'pooling.' => converti
        if target_pool == "pool":
            k2 = k2.replace("pooling.", "pool.")
        else:
            k2 = k2.replace("pool.", "pooling.")

        # 4) prefisso finale del modello target
        new["lmk_enc." + k2] = v

    torch.save({"model": new}, dst)
    print(f"saved {dst} | keys: {len(new)} | target_pool={target_pool}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python remap_pretrain_to_lmk.py SRC_CKPT PT_DST")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
