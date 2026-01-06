import logging, random, numpy as np, torch

LOG = logging.getLogger("dualrun")

def setup_logging(level: str = "INFO"):
    level = level.upper()
    logging.basicConfig(level=getattr(logging, level, logging.INFO),
                        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    LOG.info(f"Logging initialized: {level}")

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    LOG.info(f"Seed set: {seed}")

def device_of(choice: str) -> torch.device:
    if choice == "cuda" and torch.cuda.is_available():
        LOG.info("Using CUDA"); return torch.device("cuda")
    LOG.warning("Using CPU"); return torch.device("cpu")
