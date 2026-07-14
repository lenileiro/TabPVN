"""Bayesian posterior correction: base-rate (prior-probability) shift + sequential evidence updating, and its
composition with the fair-price decision. Reproduces the Veritasium disease-test example exactly."""

import numpy as np

from tabpvn import prior_shift, sequential_test
from tabpvn.bayes import check_prior_shift, check_sequential_test
from tabpvn.bayes import (
    test_posterior as one_test_posterior,  # aliased: a bare test_* name is collected by pytest
)


def test_disease_example_matches_bayes():
    # 99% sensitivity & specificity, 0.1% prevalence -> ~9% posterior after one positive test
    assert abs(one_test_posterior(0.001, 0.99, 0.99) - 0.0902) < 1e-3
    # a second independent positive -> ~91% (the posterior becomes the next prior)
    traj = sequential_test(0.001, [(0.99, 0.99, True), (0.99, 0.99, True)])
    assert abs(traj[0] - 0.0902) < 1e-3 and abs(traj[1] - 0.9075) < 1e-3
    assert check_sequential_test(0.001, [(0.99, 0.99, True), (0.99, 0.99, True)], traj)


def test_prior_shift_corrects_and_renormalises():
    # a model very confident 'positive' (0.99) deployed where positives are rare (0.1%) -> heavily discounted
    p = np.array([0.01, 0.99])
    q = prior_shift(p, [0.5, 0.5], [0.999, 0.001])
    assert abs(q.sum() - 1.0) < 1e-9 and abs(q[1] - 0.0902) < 1e-3
    assert check_prior_shift(p, [0.5, 0.5], [0.999, 0.001], q)  # re-verifies
    # identical train/deploy priors -> no change
    assert np.allclose(prior_shift(p, [0.5, 0.5], [0.5, 0.5]), p)


def test_prior_shift_strength_is_a_geometric_interpolation():
    p = np.array([[0.2, 0.5, 0.3]])
    train = np.array([0.1, 0.8, 0.1])
    deploy = np.full(3, 1.0 / 3.0)
    expected = p * np.sqrt(deploy / train)
    expected /= expected.sum(1, keepdims=True)

    assert np.allclose(prior_shift(p, train, deploy, strength=0.5), expected)
    assert np.allclose(prior_shift(p, train, deploy, strength=0.0), p)


def test_tabpvn_posterior_and_decide_compose():
    from sklearn.datasets import make_classification

    from tabpvn import TabPVN

    X, y = make_classification(n_samples=5000, n_features=20, n_informative=8, random_state=0)
    m = TabPVN().fit(X[:4000], y[:4000])
    Xte = X[4000:]
    raw = m.predict_proba(Xte)
    post = m.posterior(Xte, prior={0: 0.99, 1: 0.01})  # deploy where class 1 is rare
    assert post[:, 1].mean() < raw[:, 1].mean()  # base-rate correction pulls P(rare) down
    assert check_prior_shift(raw[0], m._prior_train, [0.99, 0.01], post[0])
    # decide on the posterior predicts the rare class far less often than on the raw proba
    n_raw = sum(1 for p in m.decide(Xte, reward=1.0, penalty=5.0)["prediction"] if p == 1)
    n_post = sum(
        1 for p in m.decide(Xte, reward=1.0, penalty=5.0, prior={0: 0.99, 1: 0.01})["prediction"] if p == 1
    )
    assert n_post < n_raw
    assert TabPVN.check_decision(m.decide(Xte, reward=1.0, penalty=5.0, prior={0: 0.99, 1: 0.01}))
