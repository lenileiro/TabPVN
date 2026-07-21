"""Native proof-carrying categorical partitions in the deployed classifier."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from tabpvn import TabPVN
from tabpvn.base import _CategoricalEvidenceMemory, _onehot_groups, _Preprocessor
from tabpvn.certified_boost import AdditiveCertifiedClassifier


def _contains_category_node(tree):
    if tree[0] == "cat":
        return True
    return tree[0] == "node" and (_contains_category_node(tree[3]) or _contains_category_node(tree[4]))


def test_discontiguous_category_partition_is_native_and_kernel_checked():
    # A one-hot stump would need multiple stages for {north, west}; one native
    # category-in-set node expresses the finite partition directly.
    level = np.tile(["north", "east", "south", "west"], 40)
    X = pd.DataFrame({"segment": level})
    y = np.isin(level, ["north", "west"]).astype(int)
    prep = _Preprocessor(target_encoding=False)
    Xenc = prep.fit_transform(X, y)
    model = AdditiveCertifiedClassifier(
        seed=4,
        rounds=80,
        lr=0.12,
        depth=2,
        leaf=8,
        patience=12,
        refit=False,
        categorical_groups=_onehot_groups(prep),
    ).fit(Xenc, y)

    assert any(_contains_category_node(tree) for _, tree in model.trees_)
    assert (model.predict(Xenc) == y).mean() > 0.99
    assert model.kernel_certify(Xenc[:30])["scores_reproduced"] == 1.0

    proof = model.proof(Xenc, 1)
    assert proof["terms_shown"]
    assert all(TabPVN.check_proof(term["proof"]) for term in proof["terms_shown"])


def test_honest_category_partition_is_native_and_kernel_checked():
    # The level order/set is proposed from one deterministic row half and
    # evaluated on the other. It must retain the same finite proof semantics.
    level = np.tile(["north", "east", "south", "west"], 40)
    X = pd.DataFrame({"segment": level})
    y = np.isin(level, ["north", "west"]).astype(int)
    prep = _Preprocessor(target_encoding=False)
    Xenc = prep.fit_transform(X, y)
    model = AdditiveCertifiedClassifier(
        seed=4,
        rounds=80,
        lr=0.12,
        depth=2,
        leaf=8,
        patience=12,
        refit=False,
        categorical_groups=_onehot_groups(prep),
        honest_categorical=True,
    ).fit(Xenc, y)

    assert any(_contains_category_node(tree) for _, tree in model.trees_)
    assert (model.predict(Xenc) == y).mean() > 0.99
    assert model.kernel_certify(Xenc[:30])["scores_reproduced"] == 1.0

    proof = model.proof(Xenc, 1)
    assert proof["terms_shown"]
    assert all(TabPVN.check_proof(term["proof"]) for term in proof["terms_shown"])


def test_categorical_evidence_memory_reads_atomic_category_facts():
    level = np.tile(["north", "east", "south", "west"], 40)
    X = pd.DataFrame({"segment": level})
    y = np.isin(level, ["north", "west"]).astype(int)
    prep = _Preprocessor(target_encoding=False)
    Xenc = prep.fit_transform(X, y)

    memory = _CategoricalEvidenceMemory(Xenc, y, [0, 1], _onehot_groups(prep), seed=4)
    proba = memory.proba(Xenc)

    assert proba.shape == (len(y), 2)
    assert np.allclose(proba.sum(1), 1.0)
    assert (proba.argmax(1) == y).mean() > 0.99


def test_categorical_evidence_memory_can_weight_label_informative_facts():
    y = np.tile([0, 1, 0, 1], 50)
    informative = y.copy()
    nuisance = np.tile([0, 0, 1, 1], 50)
    blocks = []
    groups = []
    for codes in (informative, nuisance):
        start = sum(block.shape[1] for block in blocks)
        block = np.zeros((len(y), 2), dtype=float)
        block[np.arange(len(y)), codes] = 1.0
        blocks.append(block)
        groups.append((start, start + 1))
    X = np.column_stack(blocks)

    rarity = _CategoricalEvidenceMemory(X, y, [0, 1], groups, seed=4)
    informed = _CategoricalEvidenceMemory(
        X,
        y,
        [0, 1],
        groups,
        seed=4,
        metric="rarity_plus_label_information",
    )

    assert np.all(informed.fact_weights[0] > rarity.fact_weights[0])
    np.testing.assert_allclose(informed.fact_weights[1], rarity.fact_weights[1])
    assert informed.index_report()["metric"] == "rarity_plus_label_information"
    np.testing.assert_allclose(informed.proba(X).sum(1), 1.0)


def test_categorical_evidence_memory_can_recover_excess_pair_information():
    combinations = np.tile(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), (80, 1))
    y = np.bitwise_xor(combinations[:, 0], combinations[:, 1])
    blocks = []
    groups = []
    for codes in combinations.T:
        start = sum(block.shape[1] for block in blocks)
        block = np.zeros((len(y), 2), dtype=float)
        block[np.arange(len(y)), codes] = 1.0
        blocks.append(block)
        groups.append((start, start + 1))
    X = np.column_stack(blocks)

    memory = _CategoricalEvidenceMemory(
        X,
        y,
        [0, 1],
        groups,
        seed=4,
        metric="rarity_plus_label_information_pairs",
    )
    proba = memory.proba(X)
    report = memory.index_report()

    assert len(memory.pair_facts) == 1
    assert np.count_nonzero(memory.pair_facts[0]["weights"]) == 4
    assert (proba.argmax(1) == y).mean() > 0.99
    np.testing.assert_allclose(proba.sum(1), 1.0)
    assert report["pair_information_strength"] > 0.0
    assert report["pair_families"][0]["groups"] == [0, 1]


def test_categorical_evidence_memory_bounds_high_cardinality_pairs():
    rows = 130
    width = 65
    blocks = []
    groups = []
    for offset in range(4):
        codes = (np.arange(rows) + offset) % width
        start = sum(block.shape[1] for block in blocks)
        block = np.zeros((rows, width), dtype=float)
        block[np.arange(rows), codes] = 1.0
        blocks.append(block)
        groups.append(tuple(range(start, start + width)))
    X = np.column_stack(blocks)
    y = np.arange(rows) % 2

    memory = _CategoricalEvidenceMemory(
        X,
        y,
        [0, 1],
        groups,
        seed=4,
        metric="rarity_plus_label_information_pairs",
    )

    assert memory.pair_facts == ()
    assert memory.index_report()["pair_postings"] == 0
    np.testing.assert_allclose(memory.proba(X[:8]).sum(1), 1.0)


def test_category_memory_gate_selects_pair_information_for_pure_interaction():
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 2, size=(600, 14))
    y = np.bitwise_xor(codes[:, 0], codes[:, 1])
    raw = pd.DataFrame(
        {
            f"category_{column}": pd.Categorical(codes[:, column].astype(str))
            for column in range(codes.shape[1])
        }
    )
    prep = _Preprocessor(target_encoding=False, task="classification")
    X = prep.fit_transform(raw, y)
    model = TabPVN(seed=3, task="classification")
    model._prep = prep
    model._pred = SimpleNamespace(classes_=np.array([0, 1]))
    splits = list(StratifiedKFold(3, shuffle=True, random_state=3).split(X, y))

    weight = model._category_memory_gate(
        X,
        y,
        {"scores": np.zeros((len(y), 2)), "splits": splits},
    )
    report = model.category_memory_report_[-1]

    assert weight == 0.1
    assert model._category_memory_metric == "rarity_plus_label_information_pairs"
    assert model.category_memory_report_[0]["oof_rank_auc"] == 0.5
    assert report["oof_rank_auc"] == 1.0
    assert min(report["fold_auc_delta"]) == 0.5
    assert report["pair_information_strength"] > 0.0
    assert all(parameter["pair_families"] > 0 for parameter in report["parameters"])


def test_category_memory_gate_selects_information_metric_automatically(monkeypatch):
    y = np.tile([0, 1], 120)
    raw = pd.DataFrame(
        {
            "signal": np.where(y == 1, "positive", "negative"),
            "context_a": np.tile(["a", "b", "c", "d"], 60),
            "context_b": np.tile(["e", "f", "g"], 80),
            "context_c": np.tile(["h", "i", "j", "k", "l"], 48),
        }
    )
    prep = _Preprocessor(target_encoding=False, task="classification")
    X = prep.fit_transform(raw, y)
    signal_group = _onehot_groups(prep)[0]
    positive_column = signal_group[prep.onehot["signal"].index("positive")]

    class FakeMemory:
        def __init__(self, _X, _y, _classes, _groups, seed=0, metric="rarity"):
            self.metric = metric
            self.k = 8
            self.temp = 1.0

        def proba(self, query):
            positive = np.asarray(query)[:, positive_column] > 0.5
            p1 = np.where(positive, 0.9, 0.1)
            if self.metric == "rarity":
                p1[:] = 0.5
            return np.column_stack((1.0 - p1, p1))

    monkeypatch.setattr("tabpvn.base._CategoricalEvidenceMemory", FakeMemory)
    model = TabPVN(seed=3, task="classification")
    model._prep = prep
    model._pred = SimpleNamespace(classes_=np.array([0, 1]))
    splits = list(StratifiedKFold(3, shuffle=True, random_state=3).split(X, y))
    precomp = {"scores": np.zeros((len(y), 2)), "splits": splits}

    weight = model._category_memory_gate(X, y, precomp)

    assert weight == 0.1
    assert model._category_memory_metric == "rarity_plus_label_information"
    assert model.category_memory_report_[-1]["selected"] is True
