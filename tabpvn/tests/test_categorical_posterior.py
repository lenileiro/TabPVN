"""Focused tests for the class-changing categorical posterior challenger."""

import copy

import numpy as np

from tabpvn import TabPVN
from tabpvn.proposers import CategoricalPosteriorChallenger


def _xor_categories(repeats=60):
    codes = np.tile(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), (repeats, 1))
    y = (codes[:, 0] ^ codes[:, 1]).astype(int)
    X = np.zeros((len(y), 4), dtype=float)
    X[np.arange(len(y)), codes[:, 0]] = 1.0
    X[np.arange(len(y)), 2 + codes[:, 1]] = 1.0
    return X, y


def _three_way_parity_categories(repeats=60):
    patterns = np.array(
        [[first, second, third] for first in range(2) for second in range(2) for third in range(2)]
    )
    codes = np.tile(patterns, (repeats, 1))
    y = (codes[:, 0] ^ codes[:, 1] ^ codes[:, 2]).astype(int)
    X = np.zeros((len(y), 6), dtype=float)
    for group in range(3):
        X[np.arange(len(y)), 2 * group + codes[:, group]] = 1.0
    groups = ((0, 1), (2, 3), (4, 5))
    return X, y, groups


def _balanced_parity_splits(repeats):
    fold = np.repeat(np.arange(repeats) % 3, 8)
    return [(np.flatnonzero(fold != index), np.flatnonzero(fold == index)) for index in range(3)]


def _category_model():
    model = TabPVN(seed=0)
    model.mode = "classification"
    model._pred = type("Pred", (), {"classes_": [0, 1]})()
    model._temp = 1.0
    model._prep = type(
        "Prep",
        (),
        {
            "num_cols": [],
            "na_cols": [],
            "cat_cols": ["left", "right"],
            "onehot": {"left": ["off", "on"], "right": ["off", "on"]},
            "target_encoding": {},
            "byte_cols": [],
            "compression_enabled": {},
        },
    )()
    return model


def _three_splits(n):
    fold = np.arange(n) % 3
    return [(np.flatnonzero(fold != index), np.flatnonzero(fold == index)) for index in range(3)]


def _rank_categories():
    """A category ranks risk strongly but cannot overcome a low base logit."""
    n = 600
    signal = np.zeros(n, dtype=int)
    y = np.zeros(n, dtype=int)
    for fold in range(3):
        rows = np.arange(fold, n, 3)
        signal[rows[100:]] = 1
        y[rows[:20]] = 1
        y[rows[100:180]] = 1
    X = np.zeros((n, 4), dtype=float)
    X[np.arange(n), signal] = 1.0
    X[:, 2] = 1.0
    return X, y


def _independent_categories(rows=1_800, seed=27):
    rng = np.random.default_rng(seed)
    y = rng.integers(2, size=rows)
    codes = np.column_stack([np.where(rng.random(rows) < 0.64, y, 1 - y) for _ in range(6)])
    X = np.zeros((rows, 12), dtype=float)
    for group in range(6):
        X[np.arange(rows), 2 * group + codes[:, group]] = 1.0
    return X, y


def _sparse_additive_categories(train_rows=3_000, test_rows=4_000, seed=81):
    rng = np.random.default_rng(seed)
    groups, levels = 4, 30
    effects = rng.normal(0.0, 0.9, size=(groups, levels))

    def generate(rows):
        codes = rng.integers(levels, size=(rows, groups))
        score = sum(effects[group, codes[:, group]] for group in range(groups))
        probability = 1.0 / (1.0 + np.exp(-score))
        y = (rng.random(rows) < probability).astype(int)
        X = np.zeros((rows, groups * levels), dtype=float)
        for group in range(groups):
            X[np.arange(rows), group * levels + codes[:, group]] = 1.0
        return X, y

    X_train, y_train = generate(train_rows)
    X_test, y_test = generate(test_rows)
    category_groups = tuple(tuple(range(group * levels, (group + 1) * levels)) for group in range(groups))
    return X_train, y_train, X_test, y_test, category_groups


def test_dirichlet_category_pair_recovers_interaction_and_verifies_evidence():
    X, y = _xor_categories()
    challenger = CategoricalPosteriorChallenger(
        X,
        y,
        [0, 1],
        ((0, 1), (2, 3)),
        metadata=(
            {"name": "left", "levels": ("off", "on")},
            {"name": "right", "levels": ("off", "on")},
        ),
    )
    base = np.tile([0.55, 0.45], (len(y), 1))
    combined = challenger.combine(base, X, weight=1.0)

    assert challenger.hierarchical_candidate is False
    assert (combined.argmax(1) == y).mean() == 1.0
    evidence = challenger.evidence(X, 1, base, weight=1.0)
    assert evidence["override"] is True
    assert evidence["class_counts"] == [0, 60]
    assert evidence["conditions"] == [
        {"group": 0, "name": "left", "level_index": 0, "level": "off"},
        {"group": 1, "name": "right", "level_index": 1, "level": "on"},
    ]
    assert TabPVN.verify_posterior_evidence(evidence)
    assert TabPVN.check_proof(evidence)

    tampered = dict(evidence, combined_probability=[0.9, 0.1])
    assert not TabPVN.verify_posterior_evidence(tampered)


def test_categorical_hyperedge_recovers_three_way_parity_and_verifies_evidence():
    X, y, groups = _three_way_parity_categories()
    pair_only = CategoricalPosteriorChallenger(X, y, [0, 1], groups, _hypergraph=False)
    hypergraph = CategoricalPosteriorChallenger(
        X,
        y,
        [0, 1],
        groups,
        metadata=tuple({"name": name, "levels": ("off", "on")} for name in ("first", "second", "third")),
        aggregation=CategoricalPosteriorChallenger.HYPER_STRONGEST,
    )
    base = np.tile([0.55, 0.45], (len(y), 1))

    pair_probability = pair_only.combine(base, X, weight=1.0)
    hyper_probability = hypergraph.combine(base, X, weight=1.0)

    assert (pair_probability.argmax(1) == y).mean() == 0.5
    assert (hyper_probability.argmax(1) == y).mean() == 1.0
    report = hypergraph.report()
    assert report["hyperedge_families"] == 1
    assert report["hyperedge_mdl_gain"]["(0, 1, 2)"] > 0.0
    assert report["hyperedge_mdl_penalty"]["(0, 1, 2)"] > 0.0
    assert report["hyperedge_parent_family"]["(0, 1, 2)"] == [0, 1]
    evidence = hypergraph.evidence(X, 1, base, weight=1.0)
    assert evidence["aggregation"] == CategoricalPosteriorChallenger.HYPER_STRONGEST
    assert evidence["family"] == [0, 1, 2]
    assert len(evidence["parent_factors"]) == 0
    assert TabPVN.verify_posterior_evidence(evidence)
    assert TabPVN.check_proof(evidence)

    tampered = copy.deepcopy(evidence)
    tampered["class_counts"][0] += 1
    assert not TabPVN.verify_posterior_evidence(tampered)

    mislabeled = copy.deepcopy(evidence)
    mislabeled["aggregation"] = "strongest"
    assert not TabPVN.verify_posterior_evidence(mislabeled)


def test_posterior_gate_selects_hypergraph_only_for_transferable_three_way_signal():
    repeats = 60
    X, y, groups = _three_way_parity_categories(repeats)
    base = np.tile([0.55, 0.45], (len(y), 1))
    precomputed = {"scores": np.log(base), "splits": _balanced_parity_splits(repeats)}

    model = _category_model()
    model._prep.cat_cols = ["first", "second", "third"]
    model._prep.onehot = {name: ["off", "on"] for name in model._prep.cat_cols}
    weight = model._category_posterior_gate(X, y, precomputed)

    ablation = _category_model()
    ablation._prep.cat_cols = ["first", "second", "third"]
    ablation._prep.onehot = {name: ["off", "on"] for name in ablation._prep.cat_cols}
    ablation._categorical_hypergraph_posterior = False
    ablation_weight = ablation._category_posterior_gate(X, y, precomputed)

    assert groups == ((0, 1), (2, 3), (4, 5))
    assert weight == 1.0
    assert model._category_posterior_permission == "class_change"
    assert model._category_posterior_aggregation == CategoricalPosteriorChallenger.HYPER_STRONGEST
    assert (model._category_posterior_oof_proba.argmax(1) == y).mean() == 1.0
    assert ablation_weight == 0.0
    assert ablation._category_posterior_permission is None


def test_disjoint_posterior_pool_combines_multiple_facts_and_verifies_every_factor():
    X, y = _independent_categories(rows=1_200)
    groups = tuple((2 * group, 2 * group + 1) for group in range(6))
    challenger = CategoricalPosteriorChallenger(X, y, [0, 1], groups, aggregation="disjoint_pool")
    base = np.tile([0.5, 0.5], (len(y), 1))
    strongest = challenger.combine(base, X, weight=1.0, aggregation="strongest")
    pooled = challenger.combine(base, X, weight=1.0)

    assert (pooled.argmax(1) == y).mean() > (strongest.argmax(1) == y).mean() + 0.02
    evidence = challenger.evidence(X, 0, base, weight=1.0)
    assert evidence["kind"] == "categorical_dirichlet_posterior_pool"
    assert len(evidence["factors"]) == challenger.MAX_EVIDENCE_FACTORS
    assert len(evidence["family"]) == len(set(evidence["family"]))
    assert TabPVN.verify_posterior_evidence(evidence)
    assert TabPVN.check_proof(evidence)

    tampered = copy.deepcopy(evidence)
    tampered["factors"][0]["class_counts"][0] += 1
    assert not TabPVN.verify_posterior_evidence(tampered)


def test_hierarchical_pairs_back_off_to_verified_single_category_parents():
    X, y, X_test, y_test, groups = _sparse_additive_categories()
    base = np.tile([0.5, 0.5], (len(y_test), 1))
    global_challenger = CategoricalPosteriorChallenger(X, y, [0, 1], groups, aggregation="disjoint_pool")
    hierarchical = CategoricalPosteriorChallenger(
        X,
        y,
        [0, 1],
        groups,
        aggregation="disjoint_pool",
        smoothing="hierarchical",
    )
    assert hierarchical.hierarchical_candidate is True
    global_probability = global_challenger.combine(base, X_test, weight=1.0)
    hierarchical_probability = hierarchical.combine(base, X_test, weight=1.0)

    global_accuracy = (global_probability.argmax(1) == y_test).mean()
    hierarchical_accuracy = (hierarchical_probability.argmax(1) == y_test).mean()
    assert hierarchical_accuracy > global_accuracy + 0.008

    evidence = hierarchical.evidence(X_test[:1], 0, base[:1], weight=1.0)
    pair_factors = [factor for factor in evidence["factors"] if len(factor["family"]) == 2]
    assert pair_factors and all(len(factor["parent_factors"]) == 2 for factor in pair_factors)
    assert TabPVN.verify_posterior_evidence(evidence)

    tampered = copy.deepcopy(evidence)
    tampered["factors"][0]["parent_factors"][0]["class_counts"][0] += 1
    assert not TabPVN.verify_posterior_evidence(tampered)


def test_posterior_gate_requires_transferable_top1_gain():
    X, y = _xor_categories()
    model = _category_model()
    base = np.tile([0.55, 0.45], (len(y), 1))
    precomputed = {"scores": np.log(base), "splits": _three_splits(len(y))}

    weight = model._category_posterior_gate(X, y, precomputed)

    assert weight == 1.0
    assert model._category_posterior_permission == "class_change"
    assert model.category_posterior_report_[-1]["selected"] is True
    assert model.category_posterior_report_[-1]["permission"] == "class_change"
    assert model.category_posterior_report_[-1]["reason"] == "consistent_oof_accuracy_gain"
    assert (model._category_posterior_oof_proba.argmax(1) == y).mean() == 1.0
    assert all(net > 0 for net in model.category_posterior_report_[-1]["candidates"][-1]["fold_net_wins"])
    assert all(
        candidate["paired_z"] >= model.category_posterior_report_[-1]["minimum_paired_z"]
        for candidate in model.category_posterior_report_[-1]["candidates"]
        if candidate["class_change_accepted"]
    )


def test_posterior_gate_rejects_when_booster_already_gets_the_classes_right():
    X, y = _xor_categories()
    model = _category_model()
    base = np.where(y[:, None] == 0, np.array([0.8, 0.2]), np.array([0.2, 0.8]))
    precomputed = {"scores": np.log(base), "splits": _three_splits(len(y))}

    weight = model._category_posterior_gate(X, y, precomputed)

    assert weight == 0.0
    assert model._category_posterior_oof_proba is None
    assert model.category_posterior_report_[-1]["selected"] is False
    assert model.category_posterior_report_[-1]["reason"] == "no_transferable_accuracy_or_rank_gain"


def test_posterior_gate_can_admit_rank_without_class_change_authority():
    X, y = _rank_categories()
    model = _category_model()
    base = np.tile([0.9, 0.1], (len(y), 1))
    precomputed = {"scores": np.log(base), "splits": _three_splits(len(y))}

    weight = model._category_posterior_gate(X, y, precomputed)

    report = model.category_posterior_report_[-1]
    assert weight == 1.0
    assert model._category_posterior_permission == "rank_only"
    assert report["permission"] == "rank_only"
    assert report["reason"] == "consistent_oof_rank_gain"
    assert np.all(model._category_posterior_oof_proba.argmax(1) == 0)
    assert report["mean_score"] == 0.8
    assert all(
        delta >= report["minimum_fold_rank_gain"] for delta in report["candidates"][-1]["fold_rank_auc_delta"]
    )


def test_posterior_gate_selects_disjoint_pool_when_multiple_facts_transfer():
    X, y = _independent_categories()
    model = _category_model()
    model._prep.cat_cols = [f"group_{index}" for index in range(6)]
    model._prep.onehot = {name: ["off", "on"] for name in model._prep.cat_cols}
    base = np.tile([0.5, 0.5], (len(y), 1))
    precomputed = {"scores": np.log(base), "splits": _three_splits(len(y))}

    weight = model._category_posterior_gate(X, y, precomputed)

    assert weight == 1.0
    assert model._category_posterior_permission == "class_change"
    assert model._category_posterior_aggregation == "disjoint_pool"
    report = model.category_posterior_report_[-1]
    pooled = next(
        candidate
        for candidate in report["candidates"]
        if candidate["aggregation"] == "disjoint_pool" and candidate["weight"] == 1.0
    )
    strongest = next(
        candidate
        for candidate in report["candidates"]
        if candidate["aggregation"] == "strongest" and candidate["weight"] == 1.0
    )
    assert pooled["oof_accuracy"] > strongest["oof_accuracy"] + 0.02
    assert all(net > 0 for net in pooled["fold_net_wins"])


def test_posterior_gate_selects_hierarchical_backoff_on_sparse_pairs():
    X, y, _X_test, _y_test, _groups = _sparse_additive_categories(test_rows=1)
    model = _category_model()
    names = [f"group_{index}" for index in range(4)]
    model._prep.cat_cols = names
    model._prep.onehot = {name: [str(level) for level in range(30)] for name in names}
    base = np.tile([0.5, 0.5], (len(y), 1))
    precomputed = {"scores": np.log(base), "splits": _three_splits(len(y))}

    weight = model._category_posterior_gate(X, y, precomputed)

    assert weight == 0.5
    assert model._category_posterior_permission == "class_change"
    assert model._category_posterior_aggregation == "disjoint_pool"
    assert model._category_posterior_smoothing == "hierarchical"
    report = model.category_posterior_report_[-1]
    selected = next(
        candidate
        for candidate in report["candidates"]
        if candidate["smoothing"] == "hierarchical"
        and candidate["aggregation"] == "disjoint_pool"
        and candidate["weight"] == weight
    )
    assert selected["class_change_accepted"] is True
    assert all(net > 0 for net in selected["fold_net_wins"])


def test_rank_only_posterior_changes_probability_but_not_prediction_or_proof_path():
    X, y = _rank_categories()
    model = _category_model()

    class Baseline:
        classes_ = [0, 1]

        @staticmethod
        def _scores(rows):
            return np.tile(np.log([0.9, 0.1]), (len(rows), 1))

        @staticmethod
        def predict(rows):
            return np.zeros(len(rows), dtype=int)

    model._prep = None
    model._pred = Baseline()
    model._category_posterior = CategoricalPosteriorChallenger(X, y, [0, 1], ((0, 1), (2, 3)))
    model._category_posterior_w = 1.0
    model._category_posterior_permission = "rank_only"
    model._smooth = None
    model._category_memory = None
    model._proof_path_memory = None

    baseline = model._blended_proba(X, include_posterior=False)
    probability = model.predict_proba(X)

    assert not np.allclose(probability, baseline)
    assert np.all(model.predict(X) == 0)
    assert np.all(probability.argmax(1) == baseline.argmax(1))
    assert model.posterior_evidence(X, 0) is None


def test_selected_posterior_changes_standard_predict_and_predict_proba():
    X, y = _xor_categories(repeats=20)
    model = _category_model()

    class Baseline:
        classes_ = [0, 1]

        @staticmethod
        def _scores(rows):
            return np.tile(np.log([0.55, 0.45]), (len(rows), 1))

        @staticmethod
        def predict(rows):
            return np.zeros(len(rows), dtype=int)

    model._prep = None
    model._pred = Baseline()
    model._category_posterior = CategoricalPosteriorChallenger(X, y, [0, 1], ((0, 1), (2, 3)))
    model._category_posterior_w = 1.0
    model._smooth = None
    model._category_memory = None
    model._proof_path_memory = None

    probability = model.predict_proba(X)
    prediction = model.predict(X)

    assert np.allclose(probability.sum(1), 1.0)
    assert (prediction == y).mean() == 1.0
    evidence = model.posterior_evidence(X, 1)
    assert evidence["override"] is True
    response = model.proof(X, 1)
    artifact = model.proof_artifact(X, 1)
    assert response["prediction"]["value"] == 1
    assert all(
        set(condition) == {"feature", "operator", "value", "observed"} and condition["operator"] == "eq"
        for condition in response["reasons"][0]["conditions"]
    )
    assert artifact["machine_proof"]["prediction"]["kind"] == "categorical_dirichlet_posterior"
    assert TabPVN.check_proof(response, artifact=artifact)
    assert model.reason(X, 1)["kind"] == "posterior_override"
    assert model.robustness(X, 1)["certified_stable"] is False
    assert model.recourse(X, 1, target=0)["reachable"] is False
    certificate = model.certificate(X, 1)
    assert certificate["prediction"] == 1
    assert certificate["posterior_evidence"]["verified"] is True
    assert certificate["sufficient_reason"]["kind"] == "posterior_override"
