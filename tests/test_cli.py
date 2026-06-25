"""Dispatch (assembly, routing, lazy listing), the error contract, and the
shared logger."""
import importlib

import pytest

from tinykit import cli


def test_every_subcommand_builds_a_parser():
    for name, (module_path, help_text) in cli.SUBCOMMANDS.items():
        module = importlib.import_module(module_path)
        assert hasattr(module, "build_parser"), f"{name} missing build_parser"
        assert hasattr(module, "main"), f"{name} missing main"
        parser = module.build_parser()  # parser=None -> standalone ArgumentParser
        assert parser is not None
        assert help_text  # a static one-liner is registered for `tk --help`


def test_version_exits_clean():
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0


def test_help_lists_commands_lazily(capsys):
    # `tk --help` must list every command without importing the tool modules.
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for name in cli.SUBCOMMANDS:
        assert name in out


def test_get_logger_is_idempotent():
    a = cli.get_logger("tinykit.test")
    b = cli.get_logger("tinykit.test")
    assert a is b
    assert len(a.handlers) == 1  # no stacked handlers on repeat calls


def test_get_logger_verbose_sets_debug():
    import logging
    logger = cli.get_logger("tinykit.test.verbose", verbose=True)
    assert logger.level == logging.DEBUG


# --- dispatch routing --------------------------------------------------------

def test_unknown_command_exits_nonzero():
    with pytest.raises(SystemExit) as exc:
        cli.main(["bogus"])
    assert exc.value.code == 2  # argparse 'invalid choice'


def test_main_routes_to_only_the_named_command(monkeypatch):
    import tinykit.viz as viz
    called = {}
    monkeypatch.setattr(viz, "main", lambda args: called.setdefault("input", args.input))
    cli.main(["viz", "MYPOSCAR"])
    assert called["input"] == "MYPOSCAR"


# --- error contract ----------------------------------------------------------

def _boom(_):
    raise ValueError("nope")


def test_run_returns_value():
    assert cli._run(None, lambda a: 42) == 42


def test_run_wraps_exception_as_clean_exit(capsys):
    with pytest.raises(SystemExit) as exc:
        cli._run(None, _boom)
    assert exc.value.code == 1
    assert "Error: nope" in capsys.readouterr().err


def test_run_debug_env_reraises_original(monkeypatch):
    monkeypatch.setenv("TINYKIT_DEBUG", "1")
    with pytest.raises(ValueError):  # the real traceback, not a SystemExit
        cli._run(None, _boom)


def test_run_keyboard_interrupt_is_130(capsys):
    def interrupt(_):
        raise KeyboardInterrupt
    with pytest.raises(SystemExit) as exc:
        cli._run(None, interrupt)
    assert exc.value.code == 130
    assert "Interrupted" in capsys.readouterr().err
