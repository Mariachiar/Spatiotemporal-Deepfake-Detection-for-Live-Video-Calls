import random
from typing import List, Dict, Iterator
from torch.utils.data import Sampler

def _cycle_pick(pool: List[int], k: int, rng: random.Random) -> List[int]:
    """Se k>|pool| cicla più volte il pool con permutazioni diverse, massimizzando la copertura."""
    n = len(pool)
    if n == 0: return []
    out = []
    need = k
    start = rng.randrange(n)
    cur = pool[:]
    while need > 0:
        # nuova permutazione con rotazione + shuffle leggero
        rng.shuffle(cur)
        cur = cur[start:] + cur[:start]
        take = min(need, n)
        out.extend(cur[:take])
        need -= take
        start = rng.randrange(n)
    return out

class BalancedPerTechBaseSampler(Sampler[int]):
    """
    Ogni epoca: N campioni = metà real, metà fake; i fake bilanciati per tecnica,
    con possibilità di boostare alcune tecniche difficili.
    """
    def __init__(
        self,
        labels: List[int],
        tech_names: List[str],
        epoch_samples: int,
        seed_base: int = 0,
        reshuffle_each_epoch: bool = True,
        boosts: Dict[str, float] = None,   # es: {"neuraltextures": 3.0}
        min_quota: int = 0                 # minimo assoluto per-tech tra i fake
    ):
        super().__init__(None)
        assert epoch_samples > 0 and epoch_samples % 2 == 0
        self.labels = labels
        self.tech_names = [(t or "unknown").lower() for t in tech_names]
        self.N = int(epoch_samples)
        self.seed = int(seed_base)
        self.reshuffle = bool(reshuffle_each_epoch)
        self.boosts = { (k or "unknown").lower(): float(v) for k,v in (boosts or {}).items() }
        self.min_quota = int(min_quota)

        self.real_idx = [i for i,y in enumerate(labels) if y == 0]
        self.fake_idx = [i for i,y in enumerate(labels) if y == 1]
        if not self.real_idx or not self.fake_idx:
            raise ValueError("TRAIN deve contenere sia real che fake.")

        tech2idx: Dict[str, List[int]] = {}
        for i in self.fake_idx:
            tech2idx.setdefault(self.tech_names[i], []).append(i)
        # rimuovi tecniche senza campioni
        self.tech2idx = {t: idxs for t,idxs in tech2idx.items() if len(idxs) > 0}
        self.techs = sorted(self.tech2idx.keys())
        if not self.techs:
            raise ValueError("Nessuna tecnica fake identificata nel TRAIN.")

        self._rng = random.Random(self.seed)
        self._epoch = 0

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)

    def __len__(self):
        return self.N

    def __iter__(self) -> Iterator[int]:
        rng = self._rng
        rng.seed(self.seed + self._epoch * 10007 + 17)

        half = self.N // 2  # metà real, metà fake

        # quote base uguali per-tech
        ntech = len(self.techs)
        base = max(half // ntech, 0)

        # applica boost moltiplicativo poi rinormalizza alla somma=half rispettando min_quota
        weights = []
        for t in self.techs:
            w = self.boosts.get(t, 1.0)
            # evita di sprecare quota su tecniche con pool piccolo
            w = max(w, 1e-6)
            weights.append(w)
        wsum = sum(weights)
        raw_quota = {t: max(self.min_quota, int(round(half * (w / wsum)))) for t,w in zip(self.techs, weights)}

        # correzione della somma a mezzo “water-filling” per arrivare esattamente a half
        total = sum(raw_quota.values())
        if total != half:
            # aggiusta incrementando o decrementando una unità per volta sulle tecniche più “capienti”
            # capacità = dimensione del pool
            tech_order = sorted(self.techs, key=lambda t: len(self.tech2idx[t]), reverse=True)
            diff = half - total
            step = 1 if diff > 0 else -1
            diff = abs(diff)
            j = 0
            while diff > 0 and tech_order:
                t = tech_order[j % len(tech_order)]
                if step < 0 and raw_quota[t] <= self.min_quota:
                    j += 1
                    if j >= 10 * len(tech_order): break
                    continue
                raw_quota[t] += step
                diff -= 1
                j += 1

        # pick fake
        fakes: List[int] = []
        for t in self.techs:
            pool = self.tech2idx[t][:]
            if self.reshuffle: rng.shuffle(pool)
            k = raw_quota[t]
            if k <= len(pool):
                pick = rng.sample(pool, k)
            else:
                pick = _cycle_pick(pool, k, rng)
            fakes.extend(pick)

        # pick real
        real_pool = self.real_idx[:]
        if self.reshuffle: rng.shuffle(real_pool)
        if half <= len(real_pool):
            reals = rng.sample(real_pool, half)
        else:
            reals = _cycle_pick(real_pool, half, rng)

        merged = reals + fakes
        rng.shuffle(merged)
        for i in merged:
            yield i

class BalancedPerTechLOOSampler(BalancedPerTechBaseSampler):
    """Come sopra, ma esclude la tecnica held-out dai fake del TRAIN."""
    def __init__(
        self,
        labels: List[int],
        tech_names: List[str],
        heldout: str,
        epoch_samples: int,
        seed_base: int = 0,
        reshuffle_each_epoch: bool = True,
        boosts: Dict[str, float] = None,
        min_quota: int = 0
    ):
        held = (heldout or "").lower()
        tech_names_norm = [(t or "unknown").lower() for t in tech_names]
        fake_mask = [y == 1 and tech_names_norm[i] != held for i,y in enumerate(labels)]
        # filtra etichette/tecniche per costruzione dei pool mantenendo API compatibile
        labels_loo = [int(y) if fake_mask[i] or y == 0 else 0 for i,y in enumerate(labels)]
        super().__init__(
            labels=labels_loo,
            tech_names=tech_names,
            epoch_samples=epoch_samples,
            seed_base=seed_base,
            reshuffle_each_epoch=reshuffle_each_epoch,
            boosts=boosts,
            min_quota=min_quota
        )
