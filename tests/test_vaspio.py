"""write_many batching: counting, naming, kwarg forwarding, overwrite skip."""
from tinykit import vaspio


def test_write_many_counts_names_and_forwards(monkeypatch):
    calls = []

    def fake_write(structure, directory, incar, kpoints, *, overwrite=True, **kw):
        calls.append((str(directory), structure, dict(incar), kw, overwrite))
        return directory

    monkeypatch.setattr(vaspio, "write_vasp_input", fake_write)

    jobs = [("NELECT_0.10", "S", {"NELECT": 1}), ("NELECT_0.20", "S", {"NELECT": 2})]
    n = vaspio.write_many(jobs, "/tmp/out", "KPTS",
                          potcar_symbols=["K_pv"], potcar_functional="PBE")

    assert n == 2
    assert [c[0] for c in calls] == ["/tmp/out/NELECT_0.10", "/tmp/out/NELECT_0.20"]
    assert [c[2]["NELECT"] for c in calls] == [1, 2]
    assert calls[0][3] == {"potcar_symbols": ["K_pv"], "potcar_functional": "PBE"}


def test_write_many_does_not_count_skipped(monkeypatch):
    # write_vasp_input returns None when a directory is skipped (no overwrite).
    def fake_write(structure, directory, incar, kpoints, *, overwrite=True, **kw):
        return None if "skip" in str(directory) else directory

    monkeypatch.setattr(vaspio, "write_vasp_input", fake_write)

    jobs = [("keep", "S", {}), ("skip", "S", {}), ("keep2", "S", {})]
    assert vaspio.write_many(jobs, "/tmp/out", "KPTS") == 2
