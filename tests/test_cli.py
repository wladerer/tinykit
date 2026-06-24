"""Dispatcher assembly, shared logger, and version single-sourcing."""
import importlib

import pytest

from tinykit import cli


def test_every_subcommand_builds_a_parser():
    for name, module_path in cli.SUBCOMMANDS.items():
        module = importlib.import_module(module_path)
        assert hasattr(module, "build_parser"), f"{name} missing build_parser"
        assert hasattr(module, "main"), f"{name} missing main"
        parser = module.build_parser()  # parser=None -> standalone ArgumentParser
        assert parser is not None


def test_dispatcher_assembles_all_subparsers():
    # --version forces a clean SystemExit only after every subparser is built,
    # so this exercises importing + build_parser for the whole tool set.
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0


def test_version_is_single_sourced():
    assert isinstance(cli.__version__, str)
    assert cli.__version__  # non-empty


def test_get_logger_is_idempotent():
    a = cli.get_logger("tinykit.test")
    b = cli.get_logger("tinykit.test")
    assert a is b
    assert len(a.handlers) == 1  # no stacked handlers on repeat calls


def test_get_logger_verbose_sets_debug():
    import logging
    logger = cli.get_logger("tinykit.test.verbose", verbose=True)
    assert logger.level == logging.DEBUG
