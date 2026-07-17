"""Bounded contracts for the installable TabPVN runtime."""

import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from tabpvn import (
    TabPVN,
    TabPVNMultiOutput,
    TabPVNOrdinal,
    TabPVNTextPair,
    TabularDecisionClient,
    load_model,
    save_model,
)
from tabpvn.bayes import check_prior_shift, prior_shift
from tabpvn.bayes import test_posterior as bayes_test_posterior
from tabpvn.pricing import check_decision, check_no_arbitrage, no_arbitrage_report


@pytest.fixture(scope="module")
def fitted_classifier():
    rng = np.random.default_rng(17)
    X = rng.normal(size=(180, 5))
    y = (X[:, 0] + 0.4 * X[:, 1] > 0.0).astype(int)
    model = TabPVN(boost={"rounds": 20, "depth": 2, "leaf": 4}).fit(X, y)
    return model, X, y


def test_package_metadata_import_is_lazy():
    code = (
        "import sys, tabpvn; "
        "assert tabpvn.__version__; "
        "assert 'tabpvn.base' not in sys.modules; "
        "assert 'sklearn' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_estimator_parameters_clone_and_score(fitted_classifier):
    model, X, y = fitted_classifier
    fresh = TabPVN(seed=9)
    cloned = clone(fresh)
    deterministic_refit = clone(model).fit(X, y)

    assert cloned.get_params() == fresh.get_params()
    np.testing.assert_array_equal(model.predict_proba(X[:20]), deterministic_refit.predict_proba(X[:20]))
    assert fresh.set_params(alpha=0.2) is fresh
    assert fresh.alpha == 0.2
    assert 0.0 <= model.score(X, y) <= 1.0
    with pytest.raises(ValueError, match="invalid parameter"):
        fresh.set_params(unknown=True)


def test_adaptive_depth_report_is_a_read_only_exactness_audit(fitted_classifier):
    model, X, _ = fitted_classifier

    report = model.adaptive_depth_report(X[:30])

    assert report["rows"] == 30
    assert report["probability_path"] == "full_forest_required"
    assert isinstance(report["public_predict_eligible"], bool)
    assert isinstance(report["public_predict_uses_adaptive_depth"], bool)
    if report["active"]:
        assert report["predictions_match_full_forest"] is True


def test_pickle_and_versioned_file_round_trip(fitted_classifier, tmp_path):
    model, X, _ = fitted_classifier
    expected_labels = model.predict(X[:12])
    expected_probability = model.predict_proba(X[:12])

    restored = pickle.loads(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
    np.testing.assert_array_equal(restored.predict(X[:12]), expected_labels)
    np.testing.assert_array_equal(restored.predict_proba(X[:12]), expected_probability)
    assert restored.certify(X[:12]) == 1.0

    destination = tmp_path / "classifier.tabpvn"
    assert save_model(model, destination) == destination
    loaded = load_model(destination)
    np.testing.assert_array_equal(loaded.predict_proba(X[:12]), expected_probability)
    np.testing.assert_array_equal(TabPVN.load(destination).predict(X[:12]), expected_labels)


def test_decision_api_uses_the_same_fitted_runtime(fitted_classifier):
    model, X, _ = fitted_classifier
    client = TabularDecisionClient(models={"risk": model})

    response = client.decisions.create(model="risk", rows=X[:4], reward=1.0, penalty=2.0)

    assert response.verified
    assert len(response.results) == 4
    assert TabularDecisionClient.verify_decision(response)

    with pytest.raises(RuntimeError, match="not fitted"):
        TabularDecisionClient(models={"fresh": TabPVN()})
    with pytest.raises(TypeError, match="not a TabPVN"):
        TabularDecisionClient(models={"bad": object()})


def test_dataframe_schema_is_strict_and_transform_does_not_mutate_input():
    X = pd.DataFrame(
        {
            "amount": [1.0, 2.0, np.inf, 4.0, 5.0, 6.0, 7.0, 8.0],
            "kind": ["a", "a", "b", "b", "a", "b", "a", "b"],
        }
    )
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    model = TabPVN(boost={"rounds": 8, "depth": 2, "leaf": 1}).fit(X, y)
    query = X.iloc[:2].copy()
    before = query.copy(deep=True)

    model.predict(query)
    pd.testing.assert_frame_equal(query, before)
    with pytest.raises(ValueError, match="columns differ from fit"):
        model.predict(query.drop(columns=["kind"]))
    with pytest.raises(ValueError, match="columns differ from fit"):
        model.predict(query.assign(extra=1))


def test_invalid_runtime_inputs_fail_before_training_or_prediction(fitted_classifier):
    with pytest.raises(ValueError, match="archived research path"):
        TabPVN(additive=False)
    with pytest.raises(RuntimeError, match="not fitted"):
        TabPVN().predict(np.zeros((1, 2)))
    with pytest.raises(ValueError, match="one-dimensional"):
        TabPVN().fit(np.zeros((4, 2)), np.zeros((4, 1)))
    with pytest.raises(ValueError, match="NaN or infinite"):
        TabPVN().fit(np.array([[0.0], [np.nan]]), np.array([0, 1]))

    model, _, _ = fitted_classifier
    with pytest.raises(ValueError, match="expects 5"):
        model.predict(np.zeros((2, 4)))


def test_decision_arithmetic_rejects_malformed_inputs_and_claims():
    with pytest.raises(ValueError, match="expected 2"):
        prior_shift([0.4, 0.6], [1.0], [0.5, 0.5])
    with pytest.raises(ValueError, match="finite"):
        prior_shift([np.nan, 0.5], [0.5, 0.5], [0.5, 0.5])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        bayes_test_posterior(1.2, 0.9, 0.9)
    assert not check_prior_shift([0.4, 0.6], [0.5, 0.5], [0.5, 0.5], [0.4])
    assert not check_decision([True], [0.9, 0.8], 1.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="same length"):
        no_arbitrage_report([0.9], [1.0, 0.0])
    assert not check_no_arbitrage({}, 0.1)


def test_proof_verifier_fails_closed_on_malformed_and_cyclic_inputs():
    from core.kernel_fol import FOLKernel, check_proof

    kernel = FOLKernel([(("q", "X"), [("p", "X")])])
    facts, provenance = kernel.closure([("p", "a")])
    assert ("q", "a") in facts
    assert check_proof(kernel.proof(("q", "a"), provenance), [("p", "a")])
    assert not check_proof(("malformed",))

    cyclic = [("q", "a"), (("q", "X"), [("q", "X")]), []]
    cyclic[2].append(cyclic)
    assert not check_proof(cyclic)


def test_composed_estimators_validate_their_input_contracts():
    with pytest.raises(ValueError, match="at least one target"):
        TabPVNMultiOutput().fit(np.zeros((2, 1)), np.empty((2, 0)))
    with pytest.raises(ValueError, match="one-dimensional"):
        TabPVNOrdinal().fit(np.zeros((2, 1)), np.zeros((2, 1)))
    with pytest.raises(ValueError, match="positive integer"):
        TabPVNTextPair(max_tokens=0)
    with pytest.raises(ValueError, match="same number of rows"):
        TabPVNTextPair().fit(["a"], [], np.array([0]))
