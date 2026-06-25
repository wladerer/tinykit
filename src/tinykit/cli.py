# PYTHON_ARGCOMPLETE_OK
"""Shared CLI building blocks.

Holds the argument groups/resolvers reused by the VASP-input tools, plus the
unified ``tinykit`` dispatcher. The dispatcher imports the individual tool
modules lazily (inside ``main``) so this module stays import-cheap and free of
circular imports.
"""

import logging
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from tinykit.presets import load_incar_preset, available_presets

try:
    __version__ = version("tinykit")
except PackageNotFoundError:  # not installed (e.g. running from a source tree)
    __version__ = "0.0.0"

# Subcommand name -> (module providing build_parser/main, one-line help).
# The help text is kept here, not read from the module docstring, so that
# `tk --help` can list the commands without importing any of the heavy tool
# modules (pymatgen/ase/matplotlib). Only the invoked command is imported.
SUBCOMMANDS = {
    "adsorb": ("tinykit.adsorb", "Generate adsorbate-on-surface structures and VASP inputs."),
    "slabgen": ("tinykit.slabgen", "Generate surface slabs and write VASP inputs."),
    "viz": ("tinykit.viz", "Render structures and charge-density isosurfaces with POV-Ray."),
    "stmplot": ("tinykit.stmplot", "Simulate constant-current STM images from PARCHG/CHGCAR."),
    "surfind": ("tinykit.surfind", "Find surface-localized states from PROCAR/OUTCAR."),
}


def get_logger(name: str, verbose: bool = False) -> logging.Logger:
    """Return a console logger with a single handler and a consistent format.

    Idempotent: repeated calls for the same name don't stack handlers. `verbose`
    selects DEBUG over INFO. Shared so the tools don't each reinvent logging.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    return logger


# ---------------------------------------------------------------------------
# Shared argument groups for tools that emit VASP inputs
# ---------------------------------------------------------------------------

def add_incar_args(parser, default_preset: str):
    """Add --preset / --incar for choosing the INCAR source."""
    group = parser.add_argument_group("INCAR")
    group.add_argument(
        "--preset", default=default_preset,
        help=f"Named INCAR preset ({', '.join(available_presets())}); default: {default_preset}",
    )
    group.add_argument(
        "--incar", default=None,
        help="Custom INCAR file; overrides --preset",
    )
    return parser


def resolve_incar(args):
    """Return an Incar (from --incar file) or a preset dict (from --preset)."""
    if getattr(args, "incar", None):
        from pymatgen.io.vasp.inputs import Incar
        return Incar.from_file(Path(args.incar).resolve(strict=True))
    return load_incar_preset(args.preset)


def add_potcar_args(parser):
    """Add --functional for the POTCAR family."""
    parser.add_argument(
        "--functional", default="PBE",
        help="POTCAR functional family (default: PBE)",
    )
    return parser


def add_kpoints_args(parser, default):
    """Add --kpoints as a gamma-centered mesh with a per-tool default."""
    parser.add_argument(
        "--kpoints", type=int, nargs=3, default=list(default), metavar=("KX", "KY", "KZ"),
        help=f"Gamma-centered k-point mesh (default: {default[0]} {default[1]} {default[2]})",
    )
    return parser


def gamma_kpoints(args):
    """Build a gamma-centered Kpoints object from --kpoints."""
    from pymatgen.io.vasp.inputs import Kpoints
    return Kpoints.gamma_automatic(tuple(args.kpoints))


def add_overwrite_args(parser):
    """Add --no-overwrite (default: overwrite existing directories)."""
    parser.add_argument(
        "--no-overwrite", dest="overwrite", action="store_false",
        help="Skip directories that already exist instead of overwriting",
    )
    parser.set_defaults(overwrite=True)
    return parser


# ---------------------------------------------------------------------------
# Unified `tinykit` dispatcher
# ---------------------------------------------------------------------------

def _top_parser(eager: bool = False):
    """Build the top-level `tk` parser.

    By default the subparsers are stubs carrying only the static help text, so
    listing commands (`tk --help`) imports nothing heavy. With `eager` (used
    only for shell completion) each tool module is imported and its full parser
    built, so argcomplete can complete per-command arguments.
    """
    import argparse
    import importlib

    parser = argparse.ArgumentParser(
        prog="tk",
        description="Toolkit for preparing and analyzing VASP calculations.",
    )
    parser.add_argument("--version", action="version", version=f"tinykit {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="<command>")
    for name, (module_path, help_text) in SUBCOMMANDS.items():
        sub = subparsers.add_parser(name, help=help_text)
        if eager:
            module = importlib.import_module(module_path)
            module.build_parser(sub)
            sub.set_defaults(_run=module.main)
    return parser


def _run(args, func):
    """Run a tool, reporting failures cleanly (TINYKIT_DEBUG=1 for a traceback)."""
    import os
    import sys
    try:
        return func(args)
    except KeyboardInterrupt:
        sys.stderr.write("\nInterrupted.\n")
        raise SystemExit(130)
    except Exception as exc:
        if os.environ.get("TINYKIT_DEBUG"):
            raise
        sys.stderr.write(f"Error: {exc}\n")
        raise SystemExit(1)


def main(argv=None):
    # PYTHON_ARGCOMPLETE_OK
    import importlib
    import os
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    completing = "_ARGCOMPLETE" in os.environ
    command = next((a for a in argv if not a.startswith("-")), None)

    # Fast path: a known command is on the line, so import only that module and
    # dispatch. This skips the other tools' heavy imports (the whole point).
    if command in SUBCOMMANDS and not completing:
        module = importlib.import_module(SUBCOMMANDS[command][0])
        sub = module.build_parser()
        sub.prog = f"tk {command}"
        rest = argv[argv.index(command) + 1:]
        return _run(sub.parse_args(rest), module.main)

    # Listing / help / version / unknown-command / completion path. No heavy
    # imports unless completion needs the full parser.
    parser = _top_parser(eager=completing)
    if completing:
        import argcomplete
        argcomplete.autocomplete(parser)
    args = parser.parse_args(argv)
    # Only reachable here for a real command during completion fallback.
    return _run(args, args._run)


if __name__ == "__main__":
    main()
