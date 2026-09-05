"""Tests for the public LAWS command-line entry point."""

from pathlib import Path

import run_laws
from laws import run_laws as package_runner
from laws.laws_dynamic import LAWSModel_dyn
from laws.laws_initial import LAWSModel_ini
from laws.laws_model import LAWSModel


def test_root_runner_exports_the_package_runner_api():
    """The root script and installed command must use one implementation."""
    for name in ("GNU", "main", "mainwarm", "parse_args", "usage"):
        assert getattr(run_laws, name) is getattr(package_runner, name)


def test_run_cwatm_entry_points_have_been_removed():
    """Prevent the retired launcher from being reintroduced."""
    project_root = Path(__file__).parents[1]

    assert not (project_root / "run_cwatm.py").exists()
    assert not (project_root / "laws" / "run_cwatm.py").exists()

    setup_source = (project_root / "setup.py").read_text(encoding="utf-8")
    assert "laws=laws.run_laws:run_from_command_line" in setup_source
    assert "laws.run_cwatm" not in setup_source


def test_laws_uses_renamed_initial_and_dynamic_modules():
    """The LAWS lifecycle modules must not fall back to their old filenames."""
    project_root = Path(__file__).parents[1]

    assert LAWSModel_ini.__module__ == "laws.laws_initial"
    assert LAWSModel_dyn.__module__ == "laws.laws_dynamic"
    assert not (project_root / "laws" / "cwatm_initial.py").exists()
    assert not (project_root / "laws" / "cwatm_dynamic.py").exists()


def test_laws_uses_renamed_top_level_model():
    """The executable model must be provided by laws.laws_model."""
    project_root = Path(__file__).parents[1]

    assert LAWSModel.__module__ == "laws.laws_model"
    assert issubclass(LAWSModel, LAWSModel_ini)
    assert issubclass(LAWSModel, LAWSModel_dyn)
    assert not (project_root / "laws" / "cwatm_model.py").exists()
