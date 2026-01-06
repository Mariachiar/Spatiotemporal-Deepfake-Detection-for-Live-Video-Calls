# altfreezing.py
from dataclasses import dataclass

@dataclass
class AltFreezeCfg:
    enabled: bool = True
    warmup_epochs: int = 2      # epoche iniziali joint
    period: int = 2             # durata di ciascuna fase A o B
    joint_tail: int = 2         # epoche finali joint
    start_epoch: int = 1        # 1-indexed (coerente con i log)

class AltFreezer:
    def __init__(self, cfg: AltFreezeCfg):
        self.cfg = cfg

    def phase(self, epoch: int, last_epoch: int) -> str:
        if not self.cfg.enabled:
            return "joint"
        if epoch < self.cfg.start_epoch:
            return "joint"
        # warmup iniziale
        if epoch < (self.cfg.start_epoch + self.cfg.warmup_epochs):
            return "joint"
        # joint finale
        if epoch > max(self.cfg.start_epoch, last_epoch - self.cfg.joint_tail):
            return "joint"
        # alternanza A/B
        k = (epoch - self.cfg.start_epoch - self.cfg.warmup_epochs) // max(1, self.cfg.period)
        return "A" if (k % 2 == 0) else "B"

    def apply(self, model, epoch: int, last_epoch: int, logger=None) -> str:
        ph = self.phase(epoch, last_epoch)

        # head e teste sempre allenabili
        if hasattr(model, "head"):
            for p in model.head.parameters(): p.requires_grad = True
        if getattr(model, "domain_head", None) is not None:
            for p in model.domain_head.parameters(): p.requires_grad = True

        # AU / LMK
        if ph == "A":  # allena AU, congela LMK
            for p in model.au_enc.parameters():  p.requires_grad = True
            for p in model.lmk_enc.parameters(): p.requires_grad = False
        elif ph == "B":  # allena LMK, congela AU
            for p in model.au_enc.parameters():  p.requires_grad = False
            for p in model.lmk_enc.parameters(): p.requires_grad = True
        else:  # joint
            for p in model.au_enc.parameters():  p.requires_grad = True
            for p in model.lmk_enc.parameters(): p.requires_grad = True

        if logger is not None:
            logger.info(f"[AltFreezing] epoch={epoch} / last={last_epoch} -> phase={ph}")
        return ph
