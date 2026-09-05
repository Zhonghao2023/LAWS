"""Release-boundary tests for the original LAWS crop modules."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from laws.agricultural_modules.BioMassCalc import BioMassCalc
from laws.agricultural_modules.crop_WU import crop_WU
from laws.agricultural_modules.new_PM_ET import new_PM_ET


def test_lai_calculation_does_not_store_shared_ajwa_state():
    """Irrigated and non-irrigated biomass factors must remain independent."""
    module = BioMassCalc.__new__(BioMassCalc)
    module.var = SimpleNamespace(
        HUIS=np.array([0.5]),
        RDR=np.array([1.0]),
        LAR=np.array([1.0]),
        LAIM=np.array([5.0]),
    )
    module.HRLT = np.array([12.0, 12.0])
    module.WDRM = np.array([10.0, 10.0])

    module.calculatedLAI(
        HUI=np.array([0.6, 0.7]),
        dHUF=np.array([0.1, 0.1]),
        LAI0=np.array([1.0, 1.0]),
        SLA0=np.array([1.0, 1.0]),
        WS=np.array([1.0, 1.0]),
        Frac=np.array([1.0, 1.0]),
        c=0,
    )
    irrigated_factor = module.calculateBiomassDeclineFactor(
        np.array([0.6, 0.7]), 0
    )
    non_irrigated_factor = module.calculateBiomassDeclineFactor(
        np.array([0.8, 0.9]), 0
    )

    assert not hasattr(module, "AJWA")
    assert not np.array_equal(irrigated_factor, non_irrigated_factor)


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("BioMassCalc.py", "BioMassCalc"),
        ("crop_WU.py", "crop_WU"),
        ("new_PM_ET.py", "new_PM_ET"),
    ],
)
def test_crop_module_global_variables_are_documented(module_name, class_name):
    """Every active or dynamically initialized ``self.var`` needs metadata."""
    module_path = (
        Path(__file__).parents[1] / "laws" / "agricultural_modules" / module_name
    )
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    module_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    class_docstring = ast.get_docstring(module_class)
    documented_rows = set()
    for line in class_docstring.splitlines():
        columns = line.split()
        if len(columns) < 3:
            continue
        variable_name = columns[0]
        if "{Irr,nonIrr}" in variable_name:
            documented_rows.add(variable_name.replace("{Irr,nonIrr}", "Irr"))
            documented_rows.add(variable_name.replace("{Irr,nonIrr}", "nonIrr"))
        elif variable_name.isidentifier():
            documented_rows.add(variable_name)

    global_variables = set()
    for node in ast.walk(module_class):
        if not isinstance(node, ast.Attribute):
            continue
        owner = node.value
        if not isinstance(owner, ast.Attribute) or owner.attr != "var":
            continue
        if isinstance(owner.value, ast.Name) and owner.value.id == "self":
            global_variables.add(node.attr)

    # Both LAWS coupling modules initialize groups of shared model variables
    # through ``for name in [...]: vars(self.var)[name] = ...``.
    for loop in (node for node in ast.walk(module_class) if isinstance(node, ast.For)):
        if not isinstance(loop.target, ast.Name):
            continue
        if not isinstance(loop.iter, (ast.List, ast.Tuple)):
            continue
        names = [
            item.value
            for item in loop.iter.elts
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        ]
        if len(names) != len(loop.iter.elts):
            continue
        target_name = loop.target.id
        dynamically_assigns_model_var = any(
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "vars"
            and isinstance(node.slice, ast.Name)
            and node.slice.id == target_name
            for node in ast.walk(loop)
        )
        if dynamically_assigns_model_var:
            global_variables.update(names)

    assert global_variables <= documented_rows


def test_irrigated_monthly_et_uses_irrigated_et_source():
    """Protect the irrigated/non-irrigated monthly accounting boundary."""
    soil_source = (
        Path(__file__).parents[1]
        / "laws"
        / "hydrological_modules"
        / "soil_crops.py"
    ).read_text(encoding="utf-8")

    assert (
        "actTransTotal_month_Irr[c] += self.var.ET_crop_Irr[c]" in soil_source
    )
    assert (
        "actTransTotal_month_Irr[c] += self.var.ET_crop_nonIrr[c]"
        not in soil_source
    )


def test_laws_soil_processes_are_isolated_from_original_soil():
    """Keep LAWS soil changes out of the inherited CWatM implementation."""
    project_root = Path(__file__).parents[1]
    original_soil_source = (
        project_root / "laws" / "hydrological_modules" / "soil.py"
    ).read_text(encoding="utf-8")
    crop_soil_source = (
        project_root / "laws" / "hydrological_modules" / "soil_crops.py"
    ).read_text(encoding="utf-8")
    landcover_source = (
        project_root / "laws" / "hydrological_modules" / "landcoverType.py"
    ).read_text(encoding="utf-8")

    crop_guard = "if self.var.crop_coupling and No in (1, 3):"
    assert crop_soil_source.count(crop_guard) == 3
    assert crop_guard not in original_soil_source
    assert "if self.var.crop_coupling:" in landcover_source
    assert 'if "MinDepth_SPLIT_SOIL_LAYER" in binding:' in landcover_source


EXPECTED_CROP_MODULES = {"BioMassCalc.py", "crop_WU.py", "new_PM_ET.py"}


def test_crop_package_contains_only_three_implementation_modules():
    crop_package = Path(__file__).parents[1] / "laws" / "agricultural_modules"
    implementation_modules = {
        path.name for path in crop_package.glob("*.py") if path.name != "__init__.py"
    }

    assert implementation_modules == EXPECTED_CROP_MODULES


def test_evaporation_implementation_follows_crop_coupling():
    """Select one evaporation implementation for initialization and dynamics."""
    project_root = Path(__file__).parents[1]
    initial_source = (project_root / "laws" / "laws_initial.py").read_text(
        encoding="utf-8"
    )
    landcover_source = (
        project_root / "laws" / "hydrological_modules" / "landcoverType.py"
    ).read_text(encoding="utf-8")

    assert (
        "from laws.hydrological_modules.evaporation_crops import "
        "evaporation_crops"
        in initial_source
    )
    assert "if self.var.crop_coupling:" in initial_source
    assert "self.evaporation_module = evaporation_crops(self)" in initial_source
    assert "self.evaporation_module = evaporation(self)" in initial_source
    assert "self.model.evaporation_module.dynamic(coverType, coverNo)" in landcover_source


def test_soil_implementation_follows_crop_coupling():
    """Select one soil implementation for initialization and dynamics."""
    project_root = Path(__file__).parents[1]
    initial_source = (project_root / "laws" / "laws_initial.py").read_text(
        encoding="utf-8"
    )
    landcover_source = (
        project_root / "laws" / "hydrological_modules" / "landcoverType.py"
    ).read_text(encoding="utf-8")

    assert (
        "from laws.hydrological_modules.soil_crops import soil_crops"
        in initial_source
    )
    assert "self.soil_module = soil_crops(self)" in initial_source
    assert "self.soil_module = soil(self)" in initial_source
    assert "self.soil_module.initial()" in initial_source
    assert "self.model.soil_module.dynamic(coverType, coverNo)" in landcover_source


def test_irrigation_implementation_follows_crop_coupling():
    """Select one irrigation implementation after crop coupling is configured."""
    project_root = Path(__file__).parents[1]
    water_demand_source = (
        project_root
        / "laws"
        / "hydrological_modules"
        / "water_demand"
        / "water_demand.py"
    ).read_text(encoding="utf-8")

    assert (
        "from laws.hydrological_modules.water_demand.irrigation_crops import "
        "waterdemand_irrigation_crops"
        in water_demand_source
    )
    assert "if self.var.crop_coupling:" in water_demand_source
    assert (
        "self.irrigation = waterdemand_irrigation_crops(self.model)"
        in water_demand_source
    )
    assert "self.irrigation = waterdemand_irrigation(self.model)" in water_demand_source
    assert "self.irrigation.initial()" in water_demand_source
    assert "self.irrigation.dynamic()" in water_demand_source


@pytest.mark.parametrize("module_class", [BioMassCalc, crop_WU, new_PM_ET])
def test_s_curve_parameters_reproduce_control_points(module_class):
    module = module_class.__new__(module_class)
    x1, y1 = module.scrp_to_xy(10.50)
    x2, y2 = module.scrp_to_xy(100.95)

    b1, b2 = module.calculate_s_curve_params(x1, y1, x2, y2)

    predicted_y1 = x1 / (x1 + np.exp(b1 - b2 * x1))
    predicted_y2 = x2 / (x2 + np.exp(b1 - b2 * x2))

    # The implementation stores the fitted parameters at six decimal places.
    assert predicted_y1 == pytest.approx(y1, abs=1e-5)
    assert predicted_y2 == pytest.approx(y2, abs=1e-5)
