# PYTHON_ARGCOMPLETE_OK
"""Shared CLI building blocks.

Holds the argument groups/resolvers reused by the VASP-input tools, plus the
unified ``tinykit`` dispatcher. The dispatcher imports the individual tool
modules lazily (inside ``main``) so this module stays import-cheap and free of
circular imports.
"""

from pathlib import Path

from pymatgen.io.vasp.inputs import Incar, Kpoints

from tinykit.presets import load_incar_preset, available_presets

__version__ = "0.0.1"

# Subcommand name -> "module:function" providing build_parser/main.
SUBCOMMANDS = {
    "adsorb": "tinykit.adsorb",
    "slabgen": "tinykit.slabgen",
    "charge": "tinykit.charge",
    "deploy": "tinykit.deploy",
    "slabviz": "tinykit.slabviz",
    "bulkviz": "tinykit.bulkviz",
    "stmplot": "tinykit.stmplot",
    "surfind": "tinykit.surfind",
    "magviz": "tinykit.magviz",
}


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

    parser = argparse.ArgumentParser(
        prog="tinykit",
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
    return args._run(args)


if __name__ == "__main__":
    main()
