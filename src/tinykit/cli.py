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

from pymatgen.io.vasp.inputs import Incar, Kpoints

from tinykit.presets import load_incar_preset, available_presets

try:
    __version__ = version("tinykit")
except PackageNotFoundError:  # not installed (e.g. running from a source tree)
    __version__ = "0.0.0"

# Subcommand name -> "module:function" providing build_parser/main.
SUBCOMMANDS = {
    "adsorb": "tinykit.adsorb",
    "slabgen": "tinykit.slabgen",
    "charge": "tinykit.charge",
    "deploy": "tinykit.deploy",
    "viz": "tinykit.viz",
    "stmplot": "tinykit.stmplot",
    "surfind": "tinykit.surfind",
    "magviz": "tinykit.magviz",
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

def main(argv=None):
    # PYTHON_ARGCOMPLETE_OK
    import argparse
    import importlib
    import os

    parser = argparse.ArgumentParser(
        prog="tk",
        description="Toolkit for preparing and analyzing VASP calculations.",
    )
    parser.add_argument("--version", action="version", version=f"tinykit {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    for name, module_path in SUBCOMMANDS.items():
        module = importlib.import_module(module_path)
        sub = subparsers.add_parser(name, help=(module.__doc__ or name).strip().splitlines()[0])
        module.build_parser(sub)
        sub.set_defaults(_run=module.main)

    try:
        import argcomplete
        argcomplete.autocomplete(parser)
    except ImportError:
        pass

    args = parser.parse_args(argv)
    # Uniform error reporting: tools raise on failure; report cleanly here.
    # Set TINYKIT_DEBUG=1 to get the full traceback instead.
    try:
        return args._run(args)
    except KeyboardInterrupt:
        parser.exit(130, "\nInterrupted.\n")
    except Exception as exc:
        if os.environ.get("TINYKIT_DEBUG"):
            raise
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
