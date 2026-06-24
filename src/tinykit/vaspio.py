"""Centralized assembly and writing of VASP input sets.

Every tinykit CLI that emits VASP inputs (adsorb, slabgen, charge, deploy)
funnels through here so POSCAR/POTCAR/INCAR/KPOINTS are assembled consistently.
"""

from pathlib import Path

from pymatgen.io.vasp.inputs import Incar, Poscar, Potcar
from pymatgen.io.vasp.sets import VaspInput


def build_vasp_input(
    structure,
    incar,
    kpoints,
    *,
    sort_structure: bool = False,
    potcar_functional: str = "PBE",
    potcar_symbols: list[str] = None,
) -> VaspInput:
    """Assemble a VaspInput from a structure plus INCAR and KPOINTS.

    Args:
        structure:         pymatgen Structure (or Slab).
        incar:             An Incar, or a plain dict of INCAR tags.
        kpoints:           A pymatgen Kpoints object.
        sort_structure:    Sort sites by species when building the POSCAR.
        potcar_functional: POTCAR functional family (default "PBE").
        potcar_symbols:    Explicit POTCAR symbols; defaults to the POSCAR's
                           site symbols (use this to request variants like K_pv).
    """
    poscar = Poscar(structure, sort_structure=sort_structure)
    symbols = potcar_symbols if potcar_symbols is not None else poscar.site_symbols
    potcar = Potcar(symbols=symbols, functional=potcar_functional)

    if not isinstance(incar, Incar):
        incar = Incar.from_dict(dict(incar))

    return VaspInput(incar=incar, kpoints=kpoints, poscar=poscar, potcar=potcar)


def write_vasp_input(structure, directory, incar, kpoints, *, overwrite: bool = True, **kwargs):
    """Build a VaspInput and write it to `directory` (created if absent).

    When `overwrite` is False and the directory already exists with contents,
    nothing is written and None is returned. Extra keyword arguments are
    forwarded to :func:`build_vasp_input`. Returns the directory path on write.
    """
    path = Path(directory)
    if not overwrite and path.exists() and any(path.iterdir()):
        return None
    path.mkdir(parents=True, exist_ok=True)
    build_vasp_input(structure, incar, kpoints, **kwargs).write_input(path)
    return path


def write_many(jobs, directory, kpoints, *, overwrite: bool = True, **kwargs) -> int:
    """Write a batch of VASP input directories and return the number written.

    `jobs` is an iterable of ``(subdir_name, structure, incar)`` triples; each is
    written under ``directory/subdir_name``. Shared keyword arguments (e.g.
    ``sort_structure``, ``potcar_functional``, ``potcar_symbols``) apply to every
    job and are forwarded to :func:`write_vasp_input`. This is the common path
    for adsorb/charge/deploy, which differ only in how they build `jobs` (varying
    the structure, the INCAR, or both).
    """
    root = Path(directory)
    written = 0
    for name, structure, incar in jobs:
        path = write_vasp_input(structure, root / name, incar, kpoints,
                                overwrite=overwrite, **kwargs)
        if path is not None:
            written += 1
    return written
