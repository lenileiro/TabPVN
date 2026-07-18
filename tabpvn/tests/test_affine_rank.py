from types import SimpleNamespace

import numpy as np

import tabpvn.base as base
from tabpvn import TabPVN
from tabpvn.proposers.affine import AffineLogitRead


def test_affine_logit_read_collapses_standardization_and_respects_class_order():
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(7)
    X = rng.normal(size=(180, 5))
    X[:, 4] = 3.0
    y = (1.2 * X[:, 0] - 0.8 * X[:, 1] + 0.2 * rng.normal(size=len(X)) > 0.0).astype(int)
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale <= 0.0] = 1.0
    expected_model = LogisticRegression(C=0.03, solver="lbfgs", max_iter=500, random_state=3).fit(
        (X - mean) / scale,
        y,
    )

    read = AffineLogitRead(seed=3).fit(X, y, classes=np.array([1, 0]))
    actual = read.proba(X[:20])
    expected = expected_model.predict_proba((X[:20] - mean) / scale)[:, ::-1]

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual.sum(axis=1), 1.0)
    assert read.report()["serving"] == "explicit_affine_softmax"


def test_affine_logit_read_matches_multiclass_softmax():
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(11)
    X = rng.normal(size=(240, 4))
    logits = np.column_stack((X[:, 0], X[:, 1] - X[:, 2], -X[:, 0] + X[:, 3]))
    y = logits.argmax(axis=1)
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    expected_model = LogisticRegression(C=0.03, solver="lbfgs", max_iter=500, random_state=5).fit(
        (X - mean) / scale,
        y,
    )

    read = AffineLogitRead(seed=5).fit(X, y, classes=np.array([2, 0, 1]))

    np.testing.assert_allclose(
        read.proba(X[:25]),
        expected_model.predict_proba((X[:25] - mean) / scale)[:, [2, 0, 1]],
        rtol=1e-12,
        atol=1e-12,
    )


def test_affine_prior_ratio_composition_matches_declared_probability_arithmetic():
    base_probability = np.array([[0.60, 0.40], [0.25, 0.75]])
    affine_probability = np.array([[0.70, 0.30], [0.45, 0.55]])
    prior = np.array([0.90, 0.10])
    weight = 0.5

    actual = AffineLogitRead.combine(
        base_probability,
        affine_probability,
        weight,
        composition="prior_ratio",
        prior=prior,
    )
    expected = base_probability * np.power(affine_probability / prior, weight)
    expected /= expected.sum(axis=1, keepdims=True)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual.sum(axis=1), 1.0, rtol=0.0, atol=1e-15)

    row_priors = np.array([[0.90, 0.10], [0.70, 0.30]])
    row_actual = AffineLogitRead.combine(
        base_probability,
        affine_probability,
        weight,
        composition="prior_ratio",
        prior=row_priors,
    )
    row_expected = base_probability * np.power(affine_probability / row_priors, weight)
    row_expected /= row_expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(row_actual, row_expected, rtol=1e-12, atol=1e-12)


def test_affine_gate_requires_every_rank_fold_and_can_earn_decision_authority(monkeypatch):
    class AffineStub:
        combine = staticmethod(AffineLogitRead.combine)

        def __init__(self, **_kwargs):
            pass

        def fit(self, _X, _y, *, classes):
            self.classes_ = np.asarray(classes)
            return self

        def proba(self, X):
            positive = np.asarray(X)[:, 0]
            return np.column_stack((1.0 - positive, positive))

        def report(self):
            return {"kind": "stub"}

    monkeypatch.setattr(base, "AffineLogitRead", AffineStub)
    y = np.tile(np.array([0, 1]), 120)
    X = y[:, None].astype(float)
    rows = np.arange(len(y))
    splits = [
        (np.arange(80, 240), np.arange(0, 80)),
        (np.concatenate((np.arange(0, 80), np.arange(160, 240))), np.arange(80, 160)),
        (np.arange(0, 160), np.arange(160, 240)),
    ]
    precomp = {
        "scores": np.zeros((len(y), 2)),
        "splits": splits,
        "evidence_rows": rows,
    }
    model = TabPVN(seed=0)
    model._pred = type("Pred", (), {"classes_": np.array([0, 1])})()
    model._temp = 1.0
    model._fit_validation = None
    model.feature_names_ = ["signal"]
    model._prep = None
    incumbent = model._oof_probability_stack(precomp)

    weight = model._global_affine_rank_gate(X, y, precomp)

    assert weight == base._affine_rank_weight(len(y))
    assert model.affine_rank_report_[-1]["selected"] is True
    assert model.affine_rank_report_[-1]["permission"] == "decision_and_rank"
    assert model.affine_rank_report_[-1]["accuracy_gain"] == 0.5
    assert model.affine_rank_report_[-1]["paired_z"] > 2.0
    assert min(model.affine_rank_report_[-1]["fold_auc_delta"]) > 0.0
    assert not np.array_equal(model._affine_rank_oof_proba.argmax(1), incumbent.argmax(1))

    losing = X.copy()
    losing[splits[1][1], 0] = 1.0 - losing[splits[1][1], 0]

    assert model._global_affine_rank_gate(losing, y, precomp) == 0.0
    assert model.affine_rank_report_[-1]["selected"] is False
    assert min(model.affine_rank_report_[-1]["fold_auc_delta"]) < 0.0


def test_affine_gate_keeps_rank_only_when_hard_accuracy_evidence_is_absent(monkeypatch):
    class AffineStub:
        combine = staticmethod(AffineLogitRead.combine)

        def __init__(self, **_kwargs):
            pass

        def fit(self, _X, _y, *, classes):
            self.classes_ = np.asarray(classes)
            return self

        @staticmethod
        def proba(X):
            positive = 0.02 + 0.96 * np.asarray(X)[:, 0]
            return np.column_stack((1.0 - positive, positive))

        @staticmethod
        def report():
            return {"kind": "stub"}

    monkeypatch.setattr(base, "AffineLogitRead", AffineStub)
    y = np.tile(np.array([0, 1]), 120)
    X = y[:, None].astype(float)
    splits = [
        (np.arange(80, 240), np.arange(0, 80)),
        (np.concatenate((np.arange(0, 80), np.arange(160, 240))), np.arange(80, 160)),
        (np.arange(0, 160), np.arange(160, 240)),
    ]
    precomp = {
        "scores": np.tile(np.array([0.0, -20.0]), (len(y), 1)),
        "splits": splits,
        "evidence_rows": np.arange(len(y)),
    }
    model = TabPVN(seed=0)
    model._pred = type("Pred", (), {"classes_": np.array([0, 1])})()
    model._temp = 1.0
    model._fit_validation = None
    model.feature_names_ = ["signal"]
    model._prep = None
    incumbent = model._oof_probability_stack(precomp)

    assert model._global_affine_rank_gate(X, y, precomp) > 0.0
    assert model._affine_rank_permission == "rank_only"
    assert model.affine_rank_report_[-1]["accuracy_gain"] == 0.0
    assert np.array_equal(model._affine_rank_oof_proba.argmax(1), incumbent.argmax(1))


def test_affine_gate_can_select_prior_corrected_decision_composition(monkeypatch):
    class AffineStub:
        combine = staticmethod(AffineLogitRead.combine)

        def __init__(self, **_kwargs):
            pass

        def fit(self, _X, _y, *, classes):
            self.classes_ = np.asarray(classes)
            return self

        @staticmethod
        def proba(X):
            positive = np.where(np.asarray(X)[:, 0] > 0.5, 0.30, 0.05)
            return np.column_stack((1.0 - positive, positive))

        @staticmethod
        def report():
            return {"kind": "stub"}

    monkeypatch.setattr(base, "AffineLogitRead", AffineStub)
    y = np.tile(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 30)
    X = y[:, None].astype(float)
    splits = [
        (np.arange(100, 300), np.arange(0, 100)),
        (np.concatenate((np.arange(0, 100), np.arange(200, 300))), np.arange(100, 200)),
        (np.arange(0, 200), np.arange(200, 300)),
    ]
    precomp = {
        "scores": np.tile(np.log([0.60, 0.40]), (len(y), 1)),
        "splits": splits,
        "evidence_rows": np.arange(len(y)),
    }
    model = TabPVN(seed=0)
    model._pred = type("Pred", (), {"classes_": np.array([0, 1])})()
    model._temp = 1.0
    model._fit_validation = None
    model._prior_train = np.array([0.90, 0.10])
    model.feature_names_ = ["rare_signal"]
    model._prep = None

    assert model._global_affine_rank_gate(X, y, precomp) > 0.0
    report = model.affine_rank_report_[-1]
    reports_by_composition = {candidate["composition"]: candidate for candidate in report["candidates"]}
    assert model._affine_composition == "prior_ratio"
    assert model._affine_rank_permission == "decision_and_rank"
    assert report["composition"] == "prior_ratio"
    assert reports_by_composition["arithmetic"]["decision_selected"] is False
    assert reports_by_composition["prior_ratio"]["decision_selected"] is True
    assert np.isclose(reports_by_composition["prior_ratio"]["accuracy_gain"], 0.1)


def test_affine_gate_uses_fold_prior_and_is_noop_outside_temporal_evidence(monkeypatch):
    class AffineStub:
        combine = staticmethod(AffineLogitRead.combine)

        def __init__(self, **_kwargs):
            pass

        def fit(self, _X, _y, *, classes):
            self.classes_ = np.asarray(classes)
            return self

        @staticmethod
        def proba(X):
            positive = 0.01 + 0.98 * np.asarray(X)[:, 0]
            return np.column_stack((1.0 - positive, positive))

        @staticmethod
        def report():
            return {"kind": "stub"}

    monkeypatch.setattr(base, "AffineLogitRead", AffineStub)
    y = np.tile(np.array([0, 1]), 150)
    X = y[:, None].astype(float)
    valid = np.arange(200, 300)
    precomp = {
        "scores": np.zeros((len(y), 2)),
        "splits": [(np.arange(0, 200), valid)],
        "evidence_rows": valid,
    }
    model = TabPVN(seed=0)
    model._pred = type("Pred", (), {"classes_": np.array([0, 1])})()
    model._temp = 1.0
    model._fit_validation = object()
    model.feature_names_ = ["signal"]
    model._prep = None
    incumbent = model._oof_probability_stack(precomp)

    assert model._global_affine_rank_gate(X, y, precomp) > 0.0
    np.testing.assert_array_equal(model._affine_rank_oof_proba[:200], incumbent[:200])
    assert model.affine_rank_report_[-1]["oof_prior_source"] == "fold_training_rows"


def test_affine_schema_profile_admits_structured_dominant_mixed_tables():
    model = TabPVN(seed=0)
    model._prep = SimpleNamespace(
        text_cols=["description"],
        text_feat={"description": SimpleNamespace(vocab=tuple(range(40)))},
    )

    admitted = model._affine_schema_profile(100)
    model._prep.text_feat["description"].vocab = tuple(range(80))
    rejected = model._affine_schema_profile(100)

    assert admitted == {
        "eligible": True,
        "features": 100,
        "schema": "mixed_structured_text",
        "structured_features": 60,
        "token_features": 40,
        "token_fraction": 0.4,
    }
    assert rejected["eligible"] is False
    assert rejected["structured_features"] == 20
    assert rejected["token_fraction"] == 0.8


def test_affine_rank_only_selects_best_admitted_composition(monkeypatch):
    class AffineStub:
        combine = staticmethod(AffineLogitRead.combine)

        def __init__(self, **_kwargs):
            pass

        def fit(self, _X, _y, *, classes):
            self.classes_ = np.asarray(classes)
            return self

        @staticmethod
        def proba(X):
            return np.tile(np.array([0.55, 0.45]), (len(X), 1))

        @staticmethod
        def report():
            return {"kind": "stub"}

    def evaluation(encoded, baseline, candidate, evidence_rows, splits, *, composition, **_kwargs):
        del evidence_rows
        score = 0.72 if composition == "prior_ratio" else 0.70
        baseline_prediction = baseline.argmax(axis=1)
        return {
            "composition": composition,
            "candidate": candidate,
            "projected": candidate,
            "baseline_prediction": baseline_prediction,
            "decision_prediction": baseline_prediction,
            "baseline_score": 0.60,
            "rank_score": score,
            "decision_score": score,
            "baseline_accuracy": 0.50,
            "decision_accuracy": 0.50,
            "accuracy_gain": 0.0,
            "baseline_loss": 0.70,
            "rank_loss": 0.65,
            "decision_loss": 0.65,
            "log_loss_tolerance": 0.01,
            "fold_accuracy_deltas": [0.0 for _ in splits],
            "fold_net_wins": [0 for _ in splits],
            "wins": 0,
            "losses": 0,
            "paired_z": 0.0,
            "decision_fold_deltas": [score - 0.60 for _ in splits],
            "rank_fold_deltas": [score - 0.60 for _ in splits],
            "decision_selected": False,
            "decision_rank_selected": False,
            "rank_selected": True,
        }

    monkeypatch.setattr(base, "AffineLogitRead", AffineStub)
    monkeypatch.setattr(base, "_global_probability_candidate_evaluation", evaluation)
    y = np.tile(np.array([0, 1]), 120)
    X = y[:, None].astype(float)
    splits = [
        (np.arange(80, 240), np.arange(0, 80)),
        (np.concatenate((np.arange(0, 80), np.arange(160, 240))), np.arange(80, 160)),
        (np.arange(0, 160), np.arange(160, 240)),
    ]
    precomp = {
        "scores": np.zeros((len(y), 2)),
        "splits": splits,
        "evidence_rows": np.arange(len(y)),
    }
    model = TabPVN(seed=0)
    model._pred = SimpleNamespace(classes_=np.array([0, 1]))
    model._temp = 1.0
    model._fit_validation = None
    model.feature_names_ = ["signal"]
    model._prep = None

    assert model._global_affine_rank_gate(X, y, precomp) > 0.0
    assert model._affine_rank_permission == "rank_only"
    assert model._affine_composition == "prior_ratio"
    assert model.affine_rank_report_[-1]["rank_only_composition"] == "prior_ratio"


def test_affine_decision_evidence_recomputes_prediction_and_rejects_tampering():
    rng = np.random.default_rng(17)
    X = rng.normal(size=(240, 2))
    y = (X[:, 0] > 0.0).astype(int)
    read = AffineLogitRead(seed=2).fit(X, y, classes=np.array([0, 1]))
    query = np.array([[4.0, 0.0]])
    base_probability = np.array([[0.55, 0.45]])
    base_proof = {"class": 0, "n_stages": 0, "terms_shown": []}

    evidence = read.evidence(
        query,
        0,
        base_probability,
        0.5,
        base_proof=base_proof,
        verify_base=TabPVN.check_proof,
    )

    assert evidence["override"] is True
    assert evidence["prediction"] == 1
    assert evidence["verified"] is True
    assert TabPVN.verify_decision_evidence(evidence)
    assert TabPVN.check_proof(evidence)

    tampered = dict(evidence)
    tampered["coefficients"] = [row[:] for row in evidence["coefficients"]]
    tampered["coefficients"][0][0] *= -1.0
    assert not TabPVN.verify_decision_evidence(tampered)
    assert not TabPVN.check_proof(tampered)


def test_affine_prior_ratio_evidence_rejects_training_prior_tampering():
    rng = np.random.default_rng(19)
    X = rng.normal(size=(240, 2))
    y = (X[:, 0] > 0.0).astype(int)
    read = AffineLogitRead(seed=2).fit(X, y, classes=np.array([0, 1]))
    evidence = read.evidence(
        np.array([[4.0, 0.0]]),
        0,
        np.array([[0.60, 0.40]]),
        0.5,
        base_proof={"class": 0, "n_stages": 0, "terms_shown": []},
        composition="prior_ratio",
        prior=np.array([0.90, 0.10]),
        verify_base=TabPVN.check_proof,
    )

    assert evidence["override"] is True
    assert evidence["verified"] is True
    tampered = dict(evidence)
    tampered["prior"] = [0.80, 0.20]
    assert not AffineLogitRead.verify_evidence(tampered, verify_base=TabPVN.check_proof)
    assert not TabPVN.check_proof(tampered)


def test_decision_only_affine_changes_predict_and_proof_but_not_predict_proba():
    rng = np.random.default_rng(23)
    X = rng.normal(size=(240, 2))
    y = (X[:, 0] > 0.0).astype(int)

    class Baseline:
        classes_ = np.array([0, 1])

        @staticmethod
        def _scores(rows):
            return np.tile(np.log([0.55, 0.45]), (len(rows), 1))

        @staticmethod
        def predict(rows):
            return np.zeros(len(rows), dtype=int)

        @staticmethod
        def proof(_rows, _row):
            return {"class": 0, "n_stages": 0, "terms_shown": []}

    model = TabPVN(seed=0)
    model.mode = "classification"
    model._pred = Baseline()
    model._temp = 1.0
    model._conf = None
    model._prep = None
    model._interactions = None
    model.n_features_in_ = 2
    model.n_input_features_ = 2
    model.feature_names_ = ["signal", "noise"]
    model._affine_rank = AffineLogitRead(seed=2).fit(X, y, classes=np.array([0, 1]))
    model._affine_rank_weight = 0.5
    model._affine_rank_permission = "decision_only"
    query = np.array([[4.0, 0.0]])

    np.testing.assert_allclose(model.predict_proba(query), [[0.55, 0.45]])
    assert model.predict(query).tolist() == [1]
    evidence = model.affine_evidence(query, 0)
    assert evidence["base_prediction"] == 0
    assert evidence["prediction"] == 1
    response = model.proof(query, 0)
    artifact = model.proof_artifact(query, 0)
    assert response["prediction"]["value"] == 1
    assert artifact["machine_proof"]["prediction"]["kind"] == "affine_logit_decision"
    assert TabPVN.check_proof(response, artifact=artifact)
    assert model.reason(query, 0)["kind"] == "affine_override"
    certificate = model.certificate(query, 0)
    assert certificate["prediction"] == 1
    assert certificate["affine_evidence"]["verified"] is True
    assert TabPVN.check_proof(certificate["proof"], artifact=artifact)


def test_affine_rank_weight_tapers_to_zero():
    assert base._affine_rank_weight(1_000) == 0.5
    assert 0.0 < base._affine_rank_weight(6_500) < 0.25
    assert base._affine_rank_weight(10_000) == 0.0
    assert base._affine_rank_weight(100_000) == 0.0


def test_binary_class_projection_is_pointwise_and_leaves_valid_rows_unchanged():
    incumbent = np.array(
        [
            [0.8, 0.2],
            [0.7, 0.3],
            [0.4, 0.6],
            [0.1, 0.9],
        ]
    )
    candidate = np.array(
        [
            [0.1, 0.9],
            [0.8, 0.2],
            [0.8, 0.2],
            [0.05, 0.95],
        ]
    )

    batched = base._preserve_certified_class(incumbent, candidate)
    pointwise = np.vstack(
        [
            base._preserve_certified_class(
                incumbent[row : row + 1],
                candidate[row : row + 1],
            )[0]
            for row in range(len(incumbent))
        ]
    )

    np.testing.assert_array_equal(batched, pointwise)
    np.testing.assert_array_equal(batched.argmax(1), incumbent.argmax(1))
    np.testing.assert_allclose(batched[[1, 3]], candidate[[1, 3]], rtol=0.0, atol=1e-15)
    assert incumbent[0, 1] <= batched[0, 1] < 0.5
    assert 0.5 < batched[2, 1] <= incumbent[2, 1]
