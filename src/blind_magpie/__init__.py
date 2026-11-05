"""Synthetic reproduction code and saved-result viewers for blind MAGPIE."""

__version__ = "0.4.0"


def run_case(*args, **kwargs):
    """Run one synthetic reconstruction; imports GPU dependencies on demand."""
    from .reconstruct import run_case as run
    return run(*args, **kwargs)


def run_experiments(*args, **kwargs):
    """Repeat selected synthetic cases or parameter sweeps."""
    from .experiments import run_experiments as run
    return run(*args, **kwargs)
