"""Native proof-carrying categorical partitions in the deployed classifier."""

import numpy as np
import pandas as pd

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
