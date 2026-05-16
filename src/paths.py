"""Centralized default paths and run-folder naming for SCSI.

Environment variables:
    SCSI_DATA     — dataset cache root (default: ./data)
    SCSI_RESULTS  — training output root (default: ./results)

Drivers expose these as `--data_root` / `--results_root` argparse flags,
using `default_data_root()` / `default_results_root()` as the default value.

Run-folder layout
-----------------
Each run is identified by a path of the form::

    {results_root}/{dataset}/{corruption}/{slug}/

with ``slug = {cname}[-tok1][-tok2]...[-suffix][-mv][/subfolder/]`` where:

  * cname     = hyphen-joined corruption levels (``0.50-0.00-1.00``)
  * tok*      = architectural variant tokens (cds, tr, sde, g, dc, sampler,
                randt, combined, condy, embed) — emitted only when the
                corresponding ``args.*`` flag deviates from its default
  * suffix    = ``args.suffix`` (user label)
  * mv        = appended when ``args.multiview`` is True; the
                ``singleview/``/``multiview/`` directory level used in the
                pre-2026 layout no longer exists
  * subfolder = nested directory level for related runs

Datasets without a meaningful per-run corruption (e.g. ``qsos.py``) skip the
``{corruption}/`` segment by passing ``corruption_key=None`` to
``build_run_dir``.
"""
import os


def default_data_root() -> str:
    return os.environ.get("SCSI_DATA", "./data")


def default_results_root() -> str:
    return os.environ.get("SCSI_RESULTS", "./results")


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


def build_run_slug(args, *, base=None, tokens=(), subfolder=False) -> str:
    """Build the run-folder slug from an argparse namespace.

    Slug shape::

        [prefix-]{base}[-tok1][-tok2]...[-suffix][-mv][/subfolder/]

    The ``-mv`` token is placed *after* suffix (not as a regular schema token)
    so that pre-2026 ``multiview/<old-slug>/`` paths migrate by simply
    appending ``-mv`` to the slug — no parsing of suffix vs. token segments
    required.

    Parameters
    ----------
    args : argparse.Namespace
        Must expose attributes referenced by the chosen tokens; missing
        attributes are treated as default (token skipped). ``args.prefix`` /
        ``args.suffix`` / ``args.subfolder`` / ``args.multiview`` are looked
        up via ``getattr``.
    base : str, optional
        Slug root. Defaults to ``cname_from_levels(args.corruption_levels)``
        (just the levels — dataset and corruption are now path segments via
        ``build_run_dir``). Pass explicitly for non-standard roots (e.g.
        ``qsos.py``).
    tokens : iterable of str
        Names from ``_TOKEN_SCHEMA`` to attempt, in order. Each is appended
        as ``-<formatted>`` when its predicate is true. Different drivers
        emit different subsets; pass exactly the set the driver supports.
    subfolder : bool
        Honor ``args.subfolder`` by nesting under it inside the run dir
        (used by ``clean_interpolants.py`` and the eval drivers).
    """
    if base is None:
        base = cname_from_levels(args.corruption_levels)
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
    if getattr(args, "multiview", False):
        slug = f"{slug}-mv"
    if subfolder:
        sub = getattr(args, "subfolder", "")
        if sub:
            slug = f"{slug}/{sub}/"
    return slug


def build_run_dir(args, *, slug, dataset_key=None, corruption_key="__from_args__") -> str:
    """Compose the full run folder path::

        {results_root}/{dataset}/{corruption}/{slug}/

    Parameters
    ----------
    args : argparse.Namespace
        Must expose ``results_root``. ``dataset`` and ``corruption`` are
        looked up unless overridden.
    slug : str
        The slug portion (typically from ``build_run_slug``).
    dataset_key : str, optional
        Overrides ``args.dataset`` (e.g. pass ``"qso"`` when ``args.dataset``
        is absent).
    corruption_key : str or None
        Defaults to ``args.corruption``. Pass ``None`` to drop the
        ``{corruption}/`` segment entirely (for datasets without a
        per-run corruption name, e.g. ``qsos.py``).
    """
    parts = [args.results_root, dataset_key or args.dataset]
    if corruption_key == "__from_args__":
        corruption_key = getattr(args, "corruption", None)
    if corruption_key is not None:
        parts.append(corruption_key)
    parts.append(slug)
    return "/".join(parts) + "/"
