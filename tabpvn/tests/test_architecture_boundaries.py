"""Architecture seams that keep runtime, proposers, and research code separated."""

import ast
import pathlib
import tomllib

import numpy as np

from tabpvn import TabPVN
from tabpvn.proposers import default_registry


def _imports(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    return imported


def test_fit_pipeline_records_stages_and_proposer_registry():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([0.0, 1.0, 2.1, 2.9, 4.0, 5.2])

    model = TabPVN(boost={"rounds": 5, "depth": 2, "leaf": 2, "patience": 2}).fit(X, y)

    stages = [stage["name"] for stage in model.fit_stages_]
    assert stages == ["schema", "candidate_gates", "certified_predictor", "confidence", "reports"]
    assert "symbolic_predicate_boost" in model.proposer_registry_
    assert "mdl_symbolic_beam" in model.proposer_registry_
    assert "mdl_recursive_dnf" in model.proposer_registry_
    assert "mdl_exception_program" in model.proposer_registry_
    assert "bayesian_expert_router" in model.proposer_registry_
    assert "hierarchical_proof_path_memory" in model.proposer_registry_
    assert "temporal_context_state" in model.proposer_registry_
    assert "temporal_suffix_tree" in model.proposer_registry_
    assert "categorical_hypergraph_posterior" in model.proposer_registry_
    assert model.fit_pipeline_["proposers"] == default_registry().describe()


def test_default_proposer_registry_names_are_unique_and_ordered():
    names = default_registry().names()

    assert len(names) == len(set(names))
    assert names[:8] == (
        "automatic_event_schema",
        "target_encoding",
        "compression_evidence",
        "temporal_laplace_evidence",
        "temporal_context_state",
        "temporal_suffix_tree",
        "auto_boost",
        "shallow_certified_boost",
    )
    assert "rare_rank_checkpoint" in names
    assert "rare_symbolic_predicate_boost" in names
    assert "multiclass_rank_checkpoint" in names
    assert "multiclass_residual_stump_head" in names
    assert "stratified_scenario_verifier" in names
    assert "multiclass_residual_predicate_boost" in names
    assert names[-7:] == (
        "smooth_knn",
        "hierarchical_proof_path_memory",
        "bayesian_expert_router",
        "global_affine_rank",
        "categorical_posterior",
        "categorical_hypergraph_posterior",
        "numeric_interval_decision",
    )


def test_preprocessing_and_adapters_have_independent_module_boundaries():
    from tabpvn.adapters import TabPVNMultiOutput
    from tabpvn.base import TabPVNMultiOutput as LegacyMultiOutput
    from tabpvn.base import _Preprocessor as LegacyPreprocessor
    from tabpvn.preprocessing import _Preprocessor

    assert LegacyPreprocessor is _Preprocessor
    assert LegacyMultiOutput is TabPVNMultiOutput
    assert _Preprocessor.__module__ == "tabpvn.preprocessing"
    assert TabPVNMultiOutput.__module__ == "tabpvn.adapters"


def test_runtime_dependencies_point_inward_and_never_reach_research_code():
    forbidden = ("benchmark", "certifiedlm", "model", "tabpvn.experiments")
    runtime_files = list(pathlib.Path("core").glob("*.py"))
    runtime_files += [
        path for path in pathlib.Path("tabpvn").glob("*.py") if path.name != "fol_regression.py"
    ]
    for path in runtime_files:
        imported = _imports(path)
        assert not any(
            module == prefix or module.startswith(prefix + ".") for module in imported for prefix in forbidden
        ), f"{path} imports research code: {sorted(imported)}"

    for path in (pathlib.Path("tabpvn/preprocessing.py"), pathlib.Path("tabpvn/relational.py")):
        imported = _imports(path)
        assert not any(module.startswith("tabpvn.base") for module in imported)
        assert not any(module.startswith("tabpvn.trees") for module in imported)
        assert not any(module.startswith("tabpvn.certified_boost") for module in imported)

    for path in pathlib.Path("core").glob("*.py"):
        assert not any(module.startswith("tabpvn") for module in _imports(path))


def test_runtime_wheel_excludes_research_packages():
    pyproject = tomllib.loads(pathlib.Path("pyproject.toml").read_text())
    wheel = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]
    packages = wheel["packages"]
    excluded = set(wheel["exclude"])

    assert packages == ["core", "tabpvn"]
    assert "certifiedlm" not in packages
    assert "benchmark" not in packages
    assert "model" not in packages
    assert "tabpvn/experiments/**" in excluded
    assert "tabpvn/tests/**" in excluded
    assert "tabpvn/fol_regression.py" in excluded

    sdist = pyproject["tool"]["hatch"]["build"]["targets"]["sdist"]
    assert sdist["include"] == ["LICENSE", "core/**", "tabpvn/**", "pyproject.toml"]
    assert set(sdist["exclude"]) == excluded
