"""Dispatcher assembly, shared logger, and version single-sourcing."""
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
