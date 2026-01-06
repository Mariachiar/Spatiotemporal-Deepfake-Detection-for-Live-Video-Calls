import torch, re

src = "altfreezing/checkpoints/ft_ffpp_c2.pt"   # tuo ckpt FT
dst = "altfreezing/checkpoints/ft_ffpp_c22.pt"

ckpt = torch.load(src, map_location="cpu", weights_only=False)

# prendi lo state_dict (se già è un dict di pesi va bene)
sd = ckpt.get("state_dict", ckpt)

# rimuovi prefissi non attesi
new_sd = {}
for k,v in sd.items():
    k2 = k
    if k2.startswith("_warped_network."):
        k2 = k2[len("_warped_network."):]
    if k2.startswith("module."):
        k2 = k2[len("module."):]
    new_sd[k2] = v

torch.save(new_sd, dst)
print("salvato:", dst)
