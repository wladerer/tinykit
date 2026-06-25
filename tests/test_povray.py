"""Pure POV-Ray helpers: path munging, the atom-center parse-back, the .ini
thread injection, and the dashed-line / arrow geometry. No actual render."""
import os

import numpy as np

from tinykit.povray import (
    update_image_extension, _prepare_pov_path, _parse_pov_atom_locs,
    _set_ini_threads, _dashed_line_pov, _arrow_pov, available_cpus, hex_to_rgb,
)


def test_update_image_extension():
    assert update_image_extension("fig") == "fig.pov"
    assert update_image_extension("fig.png") == "fig.pov"
    assert update_image_extension("fig.pov") == "fig.pov"
    assert update_image_extension("out/fig.png") == "out/fig.pov"


def test_prepare_pov_path_is_absolute_and_makes_dir(tmp_path, monkeypatch):
    # Regression: a relative subdir output must become absolute and have its
    # parent created, otherwise the render fails from a foreign cwd.
    monkeypatch.chdir(tmp_path)
    p = _prepare_pov_path("sub/deeper/x.png")
    assert os.path.isabs(p)
    assert p.endswith(os.path.join("sub", "deeper", "x.pov"))
    assert (tmp_path / "sub" / "deeper").is_dir()


def test_parse_pov_atom_locs(tmp_path):
    # Guards the regex against ASE .pov format drift: this is the foundation of
    # both the dashed bonds and the moment arrows.
    pov = tmp_path / "s.pov"
    pov.write_text(
        "// header line, ignored\n"
        "atom(< 1.00,  2.00,  3.00>, 0.50, rgb <0.5,0.5,0.5>, 0.0, ase3) // #0\n"
        "atom(<-1.50,  0.25, -4.00>, 0.70, rgb <0.1,0.2,0.3>, 0.0, ase3) // #1\n"
        "cylinder { <0,0,0>, <1,0,0>, 0.1 }  // not an atom\n"
    )
    locs, rads = _parse_pov_atom_locs(str(pov))
    assert set(locs) == {0, 1}
    assert np.allclose(locs[0], [1.0, 2.0, 3.0])
    assert np.allclose(locs[1], [-1.5, 0.25, -4.0])
    assert rads[0] == 0.5 and rads[1] == 0.7


def test_set_ini_threads(tmp_path):
    ini = tmp_path / "s.ini"
    ini.write_text("Width=100\n")
    _set_ini_threads(str(ini), 8)
    assert "Work_Threads=8" in ini.read_text()

    ini.write_text("Width=100\n")
    _set_ini_threads(str(ini), 0)  # zero is a no-op
    assert "Work_Threads" not in ini.read_text()

    _set_ini_threads(str(tmp_path / "missing.ini"), 8)  # missing file does not raise


def test_dashed_line_segment_count():
    base, tip = np.array([0.0, 0, 0]), np.array([1.0, 0, 0])
    out = _dashed_line_pov(base, tip, 0.05, 0.3, 0.2, (0.5, 0.5, 0.5), "phong 0.6")
    assert out.count("cylinder {") == 2  # period 0.5 over length 1.0 -> 2 dashes
    # a zero-length line draws nothing
    assert _dashed_line_pov(base, base, 0.05, 0.3, 0.2, (0.5, 0.5, 0.5), "x") == ""


def test_arrow_has_shaft_and_head():
    base, tip = np.array([0.0, 0, 0]), np.array([0.0, 0, 2.0])
    out = _arrow_pov(base, tip, 0.16, 0.40, 0.34, (0.8, 0.2, 0.2), "phong 0.9")
    assert out.count("cylinder {") == 1 and out.count("cone {") == 1


def test_available_cpus_at_least_one():
    assert available_cpus() >= 1


def test_hex_to_rgb():
    assert hex_to_rgb("#ff0000") == (255, 0, 0)
    assert hex_to_rgb("00ff00") == (0, 255, 0)
