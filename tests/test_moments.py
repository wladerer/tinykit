"""Magnetic-moment extraction and the viz --moments render dispatch."""
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import write

from tinykit import magviz, viz


def test_sibling_outcar_resolution():
    assert magviz._sibling_outcar("OUTCAR").name == "OUTCAR"
    assert magviz._sibling_outcar("/a/b/vasprun.xml") == Path("/a/b/OUTCAR")


def test_collinear_moments_from_outcar(monkeypatch):
    class FakeOutcar:
        def __init__(self, path):
            self.magnetization = [{"tot": 1.0}, {"tot": -2.0}]

    monkeypatch.setattr(magviz, "Outcar", FakeOutcar)
    moments = magviz.get_moment_vectors("OUTCAR", collinear=True)
    # collinear scalars become +z vectors by sign
    assert np.allclose(moments, [[0, 0, 1.0], [0, 0, -2.0]])


def _noncollinear_outcar_text():
    def block(component, tots):
        rows = "\n".join(f"    {i + 1}   0.0  0.0  0.0  0.0  {t}"
                         for i, t in enumerate(tots))
        return (f" magnetization ({component})\n\n"
                "# of ion       s       p       d       f       tot\n"
                "----------------------------------------\n"
                f"{rows}\n"
                "----------------------------------------\n"
                f"    tot   0.0 0.0 0.0 0.0 {sum(tots)}\n\n")
    # an earlier ionic step (stale) followed by the final one we want
    stale = block("x", [9.0, 9.0]) + block("y", [9.0, 9.0]) + block("z", [9.0, 9.0])
    final = block("x", [0.1, 0.2]) + block("y", [0.0, 0.0]) + block("z", [1.0, -2.0])
    return stale + final


def test_read_outcar_moment_vectors(tmp_path):
    oc = tmp_path / "OUTCAR"
    oc.write_text(_noncollinear_outcar_text())
    vecs = magviz.read_outcar_moment_vectors(oc)
    assert vecs.shape == (2, 3)
    # picks the LAST block of each component, not the stale 9.0 step
    assert np.allclose(vecs, [[0.1, 0.0, 1.0], [0.2, 0.0, -2.0]])


def test_collinear_outcar_has_no_yz_blocks(tmp_path):
    # only an 'x' block -> not a non-collinear vector source
    (tmp_path / "OUTCAR").write_text(
        " magnetization (x)\n\n# of ion  s p d f tot\n----\n    1  0 0 0 0 5.0\n----\n")
    assert magviz.read_outcar_moment_vectors(tmp_path / "OUTCAR") is None


def test_get_moment_vectors_prefers_outcar(tmp_path):
    # OUTCAR present -> uses it; vasprun.xml need not exist (no fallback hit)
    (tmp_path / "OUTCAR").write_text(_noncollinear_outcar_text())
    vecs = magviz.get_moment_vectors(str(tmp_path / "vasprun.xml"), collinear=False)
    assert np.allclose(vecs, [[0.1, 0.0, 1.0], [0.2, 0.0, -2.0]])


def _poscar(tmp_path):
    atoms = Atoms("Fe2", positions=[(0, 0, 0), (0, 0, 2.5)], cell=[5, 5, 5], pbc=True)
    p = tmp_path / "POSCAR"
    write(str(p), atoms, format="vasp")
    return p


def test_viz_moments_dispatch_and_supercell_tiling(tmp_path, monkeypatch):
    p = _poscar(tmp_path)
    monkeypatch.setattr(viz, "get_moment_vectors",
                        lambda path, collinear=False: np.array([[0, 0, 3.0], [0, 0, -3.0]]))
    captured = {}

    def fake_render(slab, moments, output, **kw):
        captured["n_atoms"] = len(slab)
        captured["n_moments"] = len(moments)
        captured["kw"] = kw
        return output

    monkeypatch.setattr(viz, "render_structure_with_moments", fake_render)

    # 2x1x1 supercell -> 4 atoms; the 2 moments must tile to 4 to match.
    args = viz.build_parser().parse_args(
        [str(p), "--moments", "vasprun.xml", "--supercell", "2", "1", "1",
         "--moment-threshold", "0.25", "--moment-by-magnitude",
         "-o", str(tmp_path / "o.png")])
    viz.main(args)
    assert captured["n_atoms"] == 4
    assert captured["n_moments"] == 4
    # the new knobs are forwarded to the renderer
    assert captured["kw"]["min_moment"] == 0.25
    assert captured["kw"]["scale_by_magnitude"] is True


def test_viz_moments_count_mismatch_raises(tmp_path, monkeypatch):
    p = _poscar(tmp_path)
    # return the wrong number of moments for a 2-atom cell
    monkeypatch.setattr(viz, "get_moment_vectors",
                        lambda path, collinear=False: np.zeros((3, 3)))
    monkeypatch.setattr(viz, "render_structure_with_moments",
                        lambda *a, **k: pytest.fail("should not render on mismatch"))
    args = viz.build_parser().parse_args([str(p), "--moments", "vasprun.xml",
                                          "-o", str(tmp_path / "o.png")])
    with pytest.raises(ValueError, match="does not match atom count"):
        viz.main(args)
