"""The synthetic experiments reported in the paper."""
from itertools import product
import math

PTYCHI_COMMIT = "9bff0739cc91b9ec58f49341429bd955a767389d"
METHODS = ("rpie", "lsqml", "gm_rpie", "gm_magpie")
METHOD_NAMES = dict(zip(METHODS, ("rPIE", "LSQML", "GM-rPIE", "GM-MAGPIE")))
OVERLAPS = (.75, .875, .9375)
EPOCHS = {.75: 3000, .875: 800, .9375: 200}
PATTERNS = {.75: 841, .875: 3249, .9375: 12769}
ALPHAS = (.004, .02, .1)
DAMPING = (.04, .1, .2, 1.)
SIGMAS = (.3, .4, .5, .6)
ETAS = (.1, .2, .4)
BATCHES = (27, 108, 432)


def float_token(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Settings must be finite.")
    return format(value, ".9g").replace("-", "m").replace(".", "p").replace("+", "")


def case_token(overlap):
    try:
        return {.75: "overlap_075", .875: "overlap_0875", .9375: "overlap_09375"}[float(overlap)]
    except (KeyError, ValueError) as error:
        raise ValueError(f"Choose an overlap in {OVERLAPS}.") from error


def parameter_grid(method):
    """Nine PIE pairs or sixteen LSQML pairs, selected by final object error."""
    if method not in METHODS:
        raise ValueError(f"Choose a method in {METHODS}.")
    keys, values = (("damping", "sigma"), (DAMPING, SIGMAS)) if method == "lsqml" else (
        ("object_alpha", "probe_alpha"), (ALPHAS, ALPHAS))
    return [dict(zip(keys, pair)) for pair in product(*values)]


def run_id(method, parameters):
    if method not in METHODS:
        raise ValueError(f"Choose a method in {METHODS}.")
    names = (("damp", "damping"), ("sigma", "sigma")) if method == "lsqml" else (
        ("oa", "object_alpha"), ("pa", "probe_alpha"))
    if set(parameters) != {key for _, key in names}:
        raise ValueError(f"{method} requires {[key for _, key in names]}.")
    if any(not math.isfinite(value) or value <= 0 for value in parameters.values()):
        raise ValueError("Parameters must be positive and finite.")
    return "_".join(f"{label}_{float_token(parameters[key])}" for label, key in names)
