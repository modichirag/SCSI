"""Centralized default paths and run-folder naming for SCSI.

Environment variables:
    SCSI_DATA     — dataset cache root (default: ./data)
    SCSI_RESULTS  — training output root (default: ./results)

Drivers expose these as `--data_root` / `--results_root` argparse flags,
using `default_data_root()` / `default_results_root()` as the default value.

Run-folder naming
-----------------
Historically each driver built its run-folder slug inline as
``{dataset}-{corruption}-{cname}[-tok1][-tok2]...[-suffix]`` under either
``{results_root}/singleview/`` or ``{results_root}/multiview/``. This module
exposes the shared helpers `view_root`, `cname_from_levels`, `build_run_slug`,
and `build_run_dir` so all drivers and eval scripts produce identical paths
from the same args.
"""
import os


def default_data_root() -> str:
    return os.environ.get("SCSI_DATA", "./data")


def default_results_root() -> str:
    return os.environ.get("SCSI_RESULTS", "./results")


def view_root(args) -> str:
    """Path under results_root corresponding to args.multiview.

    Returns ``{results_root}/multiview`` or ``{results_root}/singleview``.
    Use this when composing paths that share the singleview/multiview split
    (training checkpoint dirs, ``--load_model_path`` resolution, etc.).
    """
    view = "multiview" if getattr(args, "multiview", False) else "singleview"
    return os.path.join(args.results_root, view)


def cname_from_levels(corruption_levels) -> str:
    """Format corruption levels as a hyphen-joined slug fragment (``0.2f`` per level)."""
    return "-".join(f"{lvl:0.2f}" for lvl in corruption_levels)


_TOKEN_SCHEMA = {
    # name -> (predicate(args), formatter(args))
    "cds":      (lambda a: getattr(a, "cleansteps", -1) != -1,
                 lambda a: f"cds{a.cleansteps}"),
    "tr":       (lambda a: getattr(a, "transport_steps", 1) != 1,
                 lambda a: f"tr{a.transport_steps}"),
    "sde":      (lambda a: getattr(a, "smodel", False),
                 lambda a: "sde"),
    "g_2f":     (lambda a: getattr(a, "gamma_scale", 0) != 0,
                 lambda a: f"g{a.gamma_scale:0.2f}"),
    "g_gated":  (lambda a: getattr(a, "gamma_scale", 0) != 0,
                 lambda a: (f"g{a.gamma_scale:0.3f}" if a.gamma_scale < 0.01
                            else f"g{a.gamma_scale:0.2f}")),
    # `dc` is gated on `smodel` in every legacy driver, matching the inline blocks.
    "dc":       (lambda a: getattr(a, "smodel", False),
                 lambda a: f"dc{a.diffusion_coeff:0.3f}"),
    "sampler":  (lambda a: getattr(a, "sampler", "euler") != "euler",
                 lambda a: a.sampler),
    "randt":    (lambda a: getattr(a, "randomize_t", False),
                 lambda a: "randt"),
    "combined": (lambda a: getattr(a, "combinedsde", False),
                 lambda a: "combined"),
    "condy":    (lambda a: getattr(a, "cond_y", False),
                 lambda a: "condy"),
    "embed":    (lambda a: getattr(a, "embed", False),
                 lambda a: "embed"),
}


def build_run_slug(args, *, base=None, tokens=(), awgn=False, subfolder=False) -> str:
    """Build the run-folder slug from an argparse namespace.

    Slug shape::

        [prefix-]{base}[-tok1][-tok2]...[-suffix][/subfolder/][-awgn]

    Parameters
    ----------
    args : argparse.Namespace
        Must expose attributes referenced by the chosen tokens; missing
        attributes are treated as default (token skipped). ``args.prefix`` /
        ``args.suffix`` / ``args.subfolder`` are looked up via ``getattr`` and
        skipped when absent or empty.
    base : str, optional
        Slug root. Defaults to
        ``f"{args.dataset}-{args.corruption}-{cname_from_levels(args.corruption_levels)}"``.
        Pass explicitly for non-standard roots (e.g. qsos.py).
    tokens : iterable of str
        Names from ``_TOKEN_SCHEMA`` to attempt, in order. Each is appended as
        ``-<formatted>`` when its predicate is true. Different drivers historically
        emitted different subsets in different orders; pass exactly the set the
        driver used.
    awgn : bool
        Append a trailing ``-awgn`` after suffix/subfolder (awgn.py /
        fid_eval_awgn.py convention).
    subfolder : bool
        Honor ``args.subfolder`` by nesting under it inside the run dir
        (used by clean_interpolants.py, fid_eval_*.py, lpips_eval_*.py).
    """
    if base is None:
        cname = cname_from_levels(args.corruption_levels)
        base = f"{args.dataset}-{args.corruption}-{cname}"
    slug = base
    for tok in tokens:
        pred, fmt = _TOKEN_SCHEMA[tok]
        if pred(args):
            slug = f"{slug}-{fmt(args)}"
    prefix = getattr(args, "prefix", "")
    if prefix:
        slug = f"{prefix}-{slug}"
    suffix = getattr(args, "suffix", "")
    if suffix:
        slug = f"{slug}-{suffix}"
    if subfolder:
        sub = getattr(args, "subfolder", "")
        if sub:
            slug = f"{slug}/{sub}/"
    if awgn:
        slug = f"{slug}-awgn"
    return slug


def build_run_dir(args, *, slug, view=True) -> str:
    """Compose the full run folder path.

    With ``view=True`` (default), prepends ``multiview/`` or ``singleview/``
    based on ``args.multiview``. With ``view=False``, skips that level
    (used by ``scsi_synthetic.py``). Always returns a trailing ``/``.
    """
    root = view_root(args) if view else args.results_root
    return f"{root}/{slug}/"
