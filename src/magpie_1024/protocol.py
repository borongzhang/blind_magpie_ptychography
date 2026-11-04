"""Fixed settings for the 1024/128 overlap study with selected parameters."""
from itertools import product

PTYCHI_REPOSITORY = "https://github.com/AdvancedPhotonSource/pty-chi"
PTYCHI_COMMIT = "9bff0739cc91b9ec58f49341429bd955a767389d"
METHODS = ("rpie", "lsqml", "gm_rpie", "gm_magpie")
METHOD_NAMES = {"rpie": "rPIE", "lsqml": "LSQML", "gm_rpie": "GM-rPIE", "gm_magpie": "GM-MAGPIE"}
ALPHAS = (0.004, 0.02, 0.1, 0.5)
DAMPING = (0.04, 0.1, 0.2, 1.0)
SIGMAS = (0.3, 0.4, 0.5, 0.6)
ETAS = (0.1,)
BATCHES = (27,)
BASE_ETA = 0.1
BASE_BATCH = 27
EPOCHS = {0.625: 10500, 0.75: 5000, 0.875: 1300}


def float_token(value):
    """Stable directory token for a finite numerical setting."""
    import math
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("A directory setting must be finite.")
    return format(value, ".9g").replace("-", "m").replace(".", "p").replace("+", "")


def case_token(overlap):
    if float(overlap) not in EPOCHS:
        raise ValueError(f"Unsupported overlap {overlap}; choose {tuple(EPOCHS)}.")
    return {0.625: "overlap_0625", 0.75: "overlap_075", 0.875: "overlap_0875"}[float(overlap)]


def parameter_grid(method):
    if method not in METHODS:
        raise ValueError(f"Unknown method {method}.")
    if method == "lsqml":
        return [dict(damping=d, sigma=s) for d, s in product(DAMPING, SIGMAS)]
    return [dict(object_alpha=a, probe_alpha=b) for a, b in product(ALPHAS, ALPHAS)]


def run_id(method, parameters):
    """Name a parameter pair; each method has its own parent directory."""
    if method not in METHODS:
        raise ValueError(f"Unknown method {method}.")
    if method == "lsqml":
        expected = {"damping", "sigma"}
        names = (("damp", "damping"), ("sigma", "sigma"))
    else:
        expected = {"object_alpha", "probe_alpha"}
        names = (("oa", "object_alpha"), ("pa", "probe_alpha"))
    if set(parameters) != expected:
        raise ValueError(f"{method} requires exactly {sorted(expected)}.")
    return "_".join(f"{label}_{float_token(parameters[key])}" for label, key in names)
