"""OOF admission and serving invariants for multiclass prior-rank projection."""

import numpy as np
from sklearn.model_selection import StratifiedKFold

from tabpvn import TabPVN
from tabpvn.base import _multiclass_prior_rank_projection


def _imbalanced_rank_case():
    """A deterministic softmax surface whose dominant denominator obscures minority rank."""
    rng = np.random.default_rng(4)
    y = np.repeat([0, 1, 2], [30, 300, 30])
    prior = np.bincount(y).astype(float) / len(y)
    scores = None
    for _ in range(2):
        noise = rng.normal(0.0, 1.0, (len(y), 3))
        signal = rng.uniform(0.05, 1.5)
        scores = np.log(prior)[None, :] + noise
        scores[np.arange(len(y)), y] += signal
        scores[:, 1] += rng.normal(0.0, rng.uniform(0.5, 3.0), len(y))
    return scores, y, prior


def test_prior_rank_gate_selects_repeatable_class_preserving_oof_lift():
    scores, y, prior = _imbalanced_rank_case()
    splits = tuple(StratifiedKFold(3, shuffle=True, random_state=0).split(np.arange(len(y)), y))
    model = TabPVN(seed=0)
    model._pred = type("Predictor", (), {"classes_": np.array([0, 1, 2])})()
    model._prior_train = prior
    model._temp = 1.0
    precomp = {
        "scores": scores,
        "splits": splits,
        "evidence_rows": np.arange(len(y)),
    }

    assert model._multiclass_prior_rank_gate(y, precomp) is True
    assert model._prior_rank_strength == 0.5
    assert model.prior_rank_report_[-1]["rank_auc_delta"] > 0.01
    assert min(model.prior_rank_report_[-1]["fold_auc_delta"]) > 0.01

    base = np.exp(scores - scores.max(1, keepdims=True))
    base /= base.sum(1, keepdims=True)
    assert np.array_equal(base.argmax(1), model._prior_rank_oof_proba.argmax(1))


def test_prior_rank_projection_is_rejected_without_dominant_class():
    scores, y, _prior = _imbalanced_rank_case()
    model = TabPVN(seed=0)
    model._pred = type("Predictor", (), {"classes_": np.array([0, 1, 2])})()
    model._prior_train = np.full(3, 1.0 / 3.0)
    precomp = {
        "scores": scores,
        "splits": tuple(StratifiedKFold(3, shuffle=True, random_state=0).split(np.arange(len(y)), y)),
        "evidence_rows": np.arange(len(y)),
    }

    assert model._multiclass_prior_rank_gate(y, precomp) is False
    assert model.prior_rank_report_[0]["reason"] == "class_prior_not_dominant"


def test_prior_rank_gate_reads_the_final_selected_oof_stack(monkeypatch):
    scores, y, prior = _imbalanced_rank_case()
    base = np.exp(scores - scores.max(1, keepdims=True))
    base /= base.sum(1, keepdims=True)
    final_stack = np.roll(base, 1, axis=1)
    captured = {}

    def projection(probability, _prior, strength=0.5):
        captured["probability"] = probability
        return probability

    monkeypatch.setattr("tabpvn.base._multiclass_prior_rank_projection", projection)
    model = TabPVN(seed=0)
    model._pred = type("Predictor", (), {"classes_": np.array([0, 1, 2])})()
    model._prior_train = prior
    model._temp = 1.0
    model._smooth_oof_proba = final_stack
    splits = tuple(StratifiedKFold(3, shuffle=True, random_state=0).split(np.arange(len(y)), y))

    assert (
        model._multiclass_prior_rank_gate(
            y,
            {"scores": scores, "splits": splits, "evidence_rows": np.arange(len(y))},
        )
        is False
    )
    assert captured["probability"] is final_stack


def test_serving_projection_matches_oof_transform_and_keeps_the_class():
    scores, _y, prior = _imbalanced_rank_case()

    class Predictor:
        classes_ = np.array([0, 1, 2])

        @staticmethod
        def _scores(X):
            return X

    model = TabPVN(seed=0)
    model._pred = Predictor()
    model._temp = 1.0
    model._prior_train = prior
    model._prior_rank_strength = 0.5
    base = model._blended_proba(scores, include_prior_rank=False)
    projected = model._blended_proba(scores)

    assert np.allclose(projected, _multiclass_prior_rank_projection(base, prior))
    assert np.array_equal(projected.argmax(1), base.argmax(1))
    assert np.allclose(projected.sum(1), 1.0)


def test_certified_decision_bypasses_rank_projection_for_exact_bayes_update():
    scores, _y, prior = _imbalanced_rank_case()

    class Predictor:
        classes_ = np.array([0, 1, 2])

        @staticmethod
        def _scores(X):
            return X

    model = TabPVN(seed=0)
    model.mode = "classification"
    model._pred = Predictor()
    model._temp = 1.0
    model._prior_train = prior
    model._prior_rank_strength = 0.5
    model._cal_conf = None
    rows = scores[:12]

    calibrated = model._blended_proba(rows, include_prior_rank=False)
    ranked = model.predict_proba(rows)
    bundle = model.certified_decision(rows, prior=np.full(3, 1.0 / 3.0))

    assert not np.allclose(ranked, calibrated)
    assert np.allclose(model.predict_calibrated_proba(rows), calibrated)
    assert np.allclose(bundle["raw_proba"], calibrated)
    assert bundle["verified"] is True
