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

    assert best_first["best_first_pair"] is False
    assert best_first["adaptive_best_first_pair"] is False
    assert hard_pair["best_first_pair"] is True
    assert hard_pair["adaptive_best_first_pair"] is False
