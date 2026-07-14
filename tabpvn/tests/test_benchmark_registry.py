"""The packaged TabPVN default remains available to the shared benchmark runner."""

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

from benchmark.experiments import models
from tabpvn import TabPVN


def test_tabpvn_and_hgb_are_available_without_optional_backends():
    tabpvn_clf = models.build("tabpvn", "classification")
    tabpvn_reg = models.build("tabpvn", "regression")
    assert isinstance(tabpvn_clf, TabPVN)
    assert isinstance(tabpvn_reg, TabPVN)
    assert tabpvn_clf.task == "classification"
    assert tabpvn_reg.task == "regression"
    assert isinstance(models.build("tabpvn_base", "classification"), TabPVN)
    assert isinstance(models.build("tabpvn_freq", "classification"), TabPVN)
    assert isinstance(models.build("tabpvn_no_rare_architecture", "classification"), TabPVN)
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
