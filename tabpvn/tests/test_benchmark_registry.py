"""The packaged TabPVN default remains available to the shared benchmark runner."""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

from benchmark.experiments import models
from tabpvn import TabPVN
from tabpvn.candidate_allocation import VerifierScore


def test_tabpvn_and_hgb_are_available_without_optional_backends():
    tabpvn_clf = models.build("tabpvn", "classification")
    tabpvn_reg = models.build("tabpvn", "regression")
    assert isinstance(tabpvn_clf, TabPVN)
    assert isinstance(tabpvn_reg, TabPVN)
    assert tabpvn_clf.task == "classification"
    assert tabpvn_reg.task == "regression"
    assert isinstance(models.build("tabpvn_legacy_allocation", "classification"), TabPVN)
    assert isinstance(models.build("tabpvn_base", "classification"), TabPVN)
    assert isinstance(models.build("tabpvn_freq", "classification"), TabPVN)
    assert isinstance(models.build("tabpvn_no_rare_architecture", "classification"), TabPVN)
    assert isinstance(models.build("tabpvn_no_adaptive_hard_pair", "classification"), TabPVN)
    assert isinstance(models.build("tabpvn_no_mdl_symbolic_beam", "classification"), TabPVN)
    assert isinstance(models.build("tabpvn_no_mdl_dnf", "classification"), TabPVN)
    assert isinstance(models.build("tabpvn_no_mdl_recursive_dnf", "classification"), TabPVN)
    assert isinstance(models.build("tabpvn_no_mdl_exception", "classification"), TabPVN)
    assert isinstance(models.build("tabpvn_no_bayesian_expert_router", "classification"), TabPVN)
    assert isinstance(models.build("tabpvn_no_hierarchical_path_memory", "classification"), TabPVN)
    assert isinstance(models.build("tabpvn_no_temporal_context_state", "classification"), TabPVN)
    assert isinstance(models.build("tabpvn_no_temporal_suffix_tree", "classification"), TabPVN)
    assert isinstance(models.build("tabpvn_no_categorical_hypergraph", "classification"), TabPVN)
    hgb_clf = models.build("hgb", "classification")
    hgb_reg = models.build("hgb", "regression")
    assert isinstance(hgb_clf, Pipeline)
    assert isinstance(hgb_reg, Pipeline)
    assert isinstance(hgb_clf[-1], HistGradientBoostingClassifier)
    assert isinstance(hgb_reg[-1], HistGradientBoostingRegressor)


def test_tabpfn_uses_the_local_public_checkpoint_when_available():
    model = models.build("tabpfn", "classification")
    assert str(model.model_path).endswith("external/tabpfn3/tabpfn-v3-classifier-v3_default.ckpt")
    assert model.device == "cpu"


def test_legacy_allocation_ablation_reproduces_absolute_top_half():
    model = models.build("tabpvn_legacy_allocation", "classification")
    scores = {
        0: VerifierScore((0.90,), (0.60, 0.95)),
        1: VerifierScore((0.80,), (0.80, 0.80)),
        2: VerifierScore((0.85,), (0.85, 0.85)),
    }

    decision = model._allocate_search_budget(
        scores,
        list(scores),
        1,
        keep=1,
        maximize=True,
        prune_dominated=True,
    )

    assert decision.promoted == (0, 1)
    assert decision.report["method"] == "legacy_absolute_top_half"


def test_multiclass_growth_ablations_do_not_inherit_the_promoted_controller(monkeypatch):
    monkeypatch.setattr(
        TabPVN,
        "_auto_tune_clf",
        lambda self, X, y: {
            "depth": 6,
            "max_leaves": 24,
            "best_first_pair": True,
            "adaptive_best_first_pair": True,
        },
    )
    X = np.zeros((450, 6))
    y = np.repeat(np.arange(3), 150)

    best_first = models.build("tabpvn_best_first_multiclass", "classification")._auto_tune_clf(X, y)
    hard_pair = models.build("tabpvn_hard_pair_best_first", "classification")._auto_tune_clf(X, y)
    ungated = models.build(
        "tabpvn_ungated_adaptive_hard_pair", "classification"
    )._with_adaptive_multiclass_pair_growth(X, y, {"depth": 6})

    assert best_first["best_first_pair"] is False
    assert best_first["adaptive_best_first_pair"] is False
    assert hard_pair["best_first_pair"] is True
    assert hard_pair["adaptive_best_first_pair"] is False
    assert ungated["adaptive_best_first_pair"] is True
    assert ungated["verifier_gated_pair_growth"] is False


def test_mdl_symbolic_beam_ablation_disables_only_beam_expansion():
    default = models.build("tabpvn", "classification")
    ablation = models.build("tabpvn_no_mdl_symbolic_beam", "classification")

    assert default._symbolic_mdl_beam is True
    assert ablation._symbolic_mdl_beam is False
    assert default._new_symbolic_predicate_map(seed=0).mdl_beam is True
    assert ablation._new_symbolic_predicate_map(seed=0).mdl_beam is False
    assert ablation._new_symbolic_predicate_map(seed=0).mdl_dnf is False
    assert ablation._new_symbolic_predicate_map(seed=0).mdl_recursive_dnf is False
    assert ablation._new_symbolic_predicate_map(seed=0).mdl_exception is False


def test_mdl_exception_ablation_retains_recursive_dnf_search():
    default = models.build("tabpvn", "classification")
    ablation = models.build("tabpvn_no_mdl_exception", "classification")

    assert default._symbolic_mdl_recursive_dnf is ablation._symbolic_mdl_recursive_dnf is True
    assert default._symbolic_mdl_exception is True
    assert ablation._symbolic_mdl_exception is False
    assert default._new_symbolic_predicate_map(seed=0).mdl_exception is True
    assert ablation._new_symbolic_predicate_map(seed=0).mdl_recursive_dnf is True
    assert ablation._new_symbolic_predicate_map(seed=0).mdl_exception is False


def test_mdl_dnf_ablation_retains_signed_conjunction_search():
    default = models.build("tabpvn", "classification")
    ablation = models.build("tabpvn_no_mdl_dnf", "classification")

    assert default._symbolic_mdl_beam is ablation._symbolic_mdl_beam is True
    assert default._symbolic_mdl_dnf is True
    assert ablation._symbolic_mdl_dnf is False
    assert default._new_symbolic_predicate_map(seed=0).mdl_dnf is True
    assert ablation._new_symbolic_predicate_map(seed=0).mdl_dnf is False
    assert ablation._new_symbolic_predicate_map(seed=0).mdl_recursive_dnf is False
    assert ablation._new_symbolic_predicate_map(seed=0).mdl_exception is True


def test_recursive_dnf_ablation_retains_pair_composition():
    default = models.build("tabpvn", "classification")
    ablation = models.build("tabpvn_no_mdl_recursive_dnf", "classification")

    assert default._symbolic_mdl_dnf is ablation._symbolic_mdl_dnf is True
    assert default._symbolic_mdl_recursive_dnf is True
    assert ablation._symbolic_mdl_recursive_dnf is False
    assert default._new_symbolic_predicate_map(seed=0).mdl_recursive_dnf is True
    assert ablation._new_symbolic_predicate_map(seed=0).mdl_dnf is True
    assert ablation._new_symbolic_predicate_map(seed=0).mdl_recursive_dnf is False
    assert ablation._new_symbolic_predicate_map(seed=0).mdl_exception is True


def test_bayesian_expert_router_ablation_retains_global_memory_blends():
    default = models.build("tabpvn", "classification")
    ablation = models.build("tabpvn_no_bayesian_expert_router", "classification")

    assert default._bayesian_expert_routing is True
    assert ablation._bayesian_expert_routing is False
    assert default._categorical_evidence is ablation._categorical_evidence is True
    assert default._proof_path_evidence is ablation._proof_path_evidence is True


def test_hierarchical_path_ablation_retains_the_local_certified_path_read():
    default = models.build("tabpvn", "classification")
    ablation = models.build("tabpvn_no_hierarchical_path_memory", "classification")

    assert default._hierarchical_proof_path_memory is True
    assert ablation._hierarchical_proof_path_memory is False
    assert default._proof_path_evidence is ablation._proof_path_evidence is True


def test_temporal_context_ablation_retains_causal_laplace_history():
    default = models.build("tabpvn", "classification")
    ablation = models.build("tabpvn_no_temporal_context_state", "classification")

    assert default._temporal_context_state is True
    assert ablation._temporal_context_state is False


def test_temporal_suffix_tree_ablation_retains_fixed_context_state():
    default = models.build("tabpvn", "classification")
    ablation = models.build("tabpvn_no_temporal_suffix_tree", "classification")

    assert default._temporal_context_tree is True
    assert ablation._temporal_context_tree is False
    assert default._temporal_context_state is ablation._temporal_context_state is True


def test_categorical_hypergraph_ablation_retains_pair_posteriors():
    default = models.build("tabpvn", "classification")
    ablation = models.build("tabpvn_no_categorical_hypergraph", "classification")

    assert default._categorical_hypergraph_posterior is True
    assert ablation._categorical_hypergraph_posterior is False
    assert default._categorical_posterior_evidence is True
    assert ablation._categorical_posterior_evidence is True
