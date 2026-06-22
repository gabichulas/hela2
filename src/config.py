from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / "data"
CACHE_DIR = BASE_DIR.parent / "cache"
TEMPLATES_DIR = BASE_DIR / "templates"

DB_PATH = DATA_DIR / "centros.db"

DEFAULT_ACO_PARAMS = {
    "n_ants": 30,
    "n_iters": 80,
    "alpha": 1.0,
    "beta": 2.0,
    "rho": 0.5,
    "q": 100.0,
}
