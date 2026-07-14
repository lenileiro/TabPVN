"""Fair-price selective prediction: the option-exercise decision (Chow's rule = the break-even strike) and the
no-arbitrage / coherence certificate on the model's confidences (the fair-bet guarantee)."""

import numpy as np

from tabpvn import fair_strike, no_arbitrage_report
from tabpvn.pricing import check_decision, check_no_arbitrage, decide


def test_fair_strike_is_the_break_even():
    assert fair_strike(1.0, 1.0, 0.0) == 0.5  # symmetric, free abstain -> argmax cut
    assert fair_strike(1.0, 4.0, 0.0) == 0.8  # a 4x-costly error pushes the strike up
    assert fair_strike(1.0, 9.0, 0.0) == 0.9
    assert 0.0 <= fair_strike(1.0, 1.0, 5.0) <= 1.0  # big abstain cost is clamped, never negative
    # break-even: at p*, answering and abstaining have equal expected value
    from tabpvn.pricing import answer_ev

    p = fair_strike(1.0, 3.0, 0.2)
    assert abs(answer_ev(p, 1.0, 3.0) - (-0.2)) < 1e-9


def test_decide_answers_above_the_strike_and_recheck():
    conf = [0.55, 0.72, 0.81, 0.95]
    actions, strike, evs = decide(conf, reward=1.0, penalty=4.0)  # strike 0.8
    assert actions == [False, False, True, True]
    assert check_decision(actions, conf, 1.0, 4.0, 0.0)  # re-verifies
    tampered = [True, False, True, True]  # claim we answered a below-strike row
    assert not check_decision(tampered, conf, 1.0, 4.0, 0.0)


def test_no_arbitrage_passes_on_calibrated_prices():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.5, 1.0, size=8000)
    correct = (rng.uniform(size=8000) < conf).astype(float)  # calibrated: P(correct)=stated confidence
    rep = no_arbitrage_report(conf, correct, n_bins=10, delta=0.05)
    assert rep["empirical_edge"] < 0.06  # no bin is meaningfully exploitable
    assert check_no_arbitrage(rep, epsilon=0.1) and rep["certified_edge"] <= 0.1


def test_no_arbitrage_flags_miscalibrated_prices():
    rng = np.random.default_rng(1)
    conf = rng.uniform(0.85, 0.95, size=4000)  # claims ~90% sure
    correct = (rng.uniform(size=4000) < 0.5).astype(float)  # but only right half the time -> overpriced
    rep = no_arbitrage_report(conf, correct, n_bins=10)
    assert rep["empirical_edge"] > 0.3  # a bettor profits ~0.4 per bet against it
    assert not check_no_arbitrage(rep, epsilon=0.1)
    assert rep["arbitrage_bin"]["gap"] < 0  # overpriced: accuracy below stated confidence


def test_tabpvn_decide_and_no_arbitrage_end_to_end():
    from sklearn.datasets import make_classification

    from tabpvn import TabPVN

    X, y = make_classification(n_samples=5000, n_features=20, n_informative=8, random_state=0)
    m = TabPVN().fit(X[:4000], y[:4000])
    Xte, yte = X[4000:], y[4000:]
    strict = m.decide(Xte, reward=1.0, penalty=9.0, abstain_cost=0.1)  # errors 9x costly -> only sure bets
    loose = m.decide(Xte, reward=1.0, penalty=1.0)  # symmetric -> answer everything
    assert strict["strike"] > loose["strike"] and loose["answer"].all()
    assert strict["answer"].sum() < len(yte)  # the option declines unfavourable bets
    ans = strict["answer"]
    acc_ans = np.mean([strict["prediction"][i] == yte[i] for i in range(len(yte)) if ans[i]])
    acc_all = np.mean(loose["prediction"] == yte)
    assert acc_ans > acc_all  # answered subset is more accurate
    assert TabPVN.check_decision(strict)  # decision re-verifies
    cert = m.no_arbitrage_certificate(epsilon=0.1, n_bins=5)
    assert cert is not None and cert["holds"] and TabPVN.check_no_arbitrage(cert)  # prices certified fair


def test_certified_decision_bundle_verifies_end_to_end():
    """The whole from the parts: one bundle composes Bayes posterior + fair strike + no-arbitrage and re-verifies
    with NO model. Tampering any part — answers, posterior, or the fairness cert — breaks verification."""
    import numpy as np
    from sklearn.datasets import make_classification

    from tabpvn import TabPVN

    X, y = make_classification(n_samples=5000, n_features=20, n_informative=8, random_state=0)
    m = TabPVN().fit(X[:4000], y[:4000])
    b = m.certified_decision(
        X[4000:], reward=1.0, penalty=9.0, abstain_cost=0.1, prior={0: 0.9, 1: 0.1}, epsilon=0.1, n_bins=5
    )
    assert b["verified"] and TabPVN.verify_decision(b)  # third party re-checks all three, no model
    assert b["posterior"] is not None and b["no_arbitrage"]["holds"]
    # tamper the answers -> fair-price step no longer matches the strike
    bad = dict(b, decision=dict(b["decision"], answer=~np.asarray(b["decision"]["answer"])))
    assert not TabPVN.verify_decision(bad)
    # tamper the posterior -> no longer the exact Bayes prior-shift of the raw proba
    assert not TabPVN.verify_decision(dict(b, posterior=b["posterior"] * 0 + 0.5))
    # tamper the fairness certificate -> certified_edge now exceeds epsilon
    na_bad = dict(b["no_arbitrage"], certified_edge=b["no_arbitrage"]["epsilon"] + 1.0)
    assert not TabPVN.verify_decision(dict(b, no_arbitrage=na_bad))


def test_decision_layer_is_opt_in():
    """The default predictor is fully automatic; decision economics start only when explicitly configured."""
    from sklearn.datasets import make_classification

    from tabpvn import TabPVN

    X, y = make_classification(n_samples=5000, n_features=20, n_informative=8, random_state=0)
    m = TabPVN().fit(X[:4000], y[:4000])
    Xte = X[4000:4100]

    plain = m.predict(Xte)
    assert plain.dtype != object and all(p is not None for p in plain)
    assert m.last_decision() is None

    m.configure_decisions(reward=1.0, penalty=9.0, abstain_cost=0.1, prior={0: 0.9, 1: 0.1})
    pol = m.predict(Xte)  # a real cost -> it starts abstaining
    assert any(p is None for p in pol)
    b = m.last_decision()
    assert b["verified"] and TabPVN.verify_decision(b)

    m.clear_decisions()
    assert m.last_decision() is None
    assert all(p is not None for p in m.predict(Xte))  # back to plain argmax
