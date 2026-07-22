"""Focused tests for the deterministic certified proof-path memory."""

import numpy as np

import tabpvn.base as base
from tabpvn import TabPVN
from tabpvn.base import _ProofPathMemory
from tabpvn.trees import _flat_leaf_ids


def _stump():
    return (
        np.array([0, -1, -1], dtype=np.int64),
        np.array([0.0, 0.0, 0.0]),
        np.array([1, -1, -1], dtype=np.int64),
        np.array([2, -1, -1], dtype=np.int64),
        np.zeros(3),
    )


def _depth_two_tree():
    return (
        np.array([0, 1, -1, -1, -1], dtype=np.int64),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([1, 3, -1, -1, -1], dtype=np.int64),
        np.array([2, 4, -1, -1, -1], dtype=np.int64),
        np.zeros(5),
    )


class _FlatPredictor:
    linear_ = False

    def __init__(self):
        self._trees = [_stump()] * 4

    def _flats(self):
        return self._trees


def _branch_data(n_per_branch=120):
    X = np.r_[np.full((n_per_branch, 1), -1.0), np.full((n_per_branch, 1), 1.0)]
    y = np.r_[np.zeros(n_per_branch, dtype=int), np.ones(n_per_branch, dtype=int)]
    return X, y


def test_flat_leaf_ids_preserve_the_certified_split_route():
    leaves = _flat_leaf_ids(_stump(), np.array([[-0.1], [0.0], [0.1]]))

    assert leaves.tolist() == [1, 1, 2]


def test_proof_path_memory_reads_exact_support_bounded_prefixes():
    X, y = _branch_data()
    memory = _ProofPathMemory(_FlatPredictor(), X, y, [0, 1])
    assert memory.index_report()["hierarchy_compiled"] is False
    proba = memory.proba(np.array([[-1.0], [1.0]]))

    assert memory.index_report()["hierarchy_compiled"] is False
    assert np.allclose(proba.sum(1), 1.0)
    assert proba.argmax(1).tolist() == [0, 1]
    assert memory.index_report()["path_facts"] == 16
    assert memory.index_report()["leaf_facts"] == 8
    assert memory.index_report()["prefix_facts"] == 8
    assert memory.index_report()["neighbourhood"] == 16


def test_hierarchical_path_memory_backs_a_sparse_leaf_with_its_parent_region():
    tree = _depth_two_tree()
    predictor = _FlatPredictor()
    predictor._trees = [tree]
    X = np.r_[
        np.tile([-1.0, -1.0], (20, 1)),
        [[-1.0, 1.0]],
        np.tile([1.0, 0.0], (20, 1)),
    ]
    # One rare positive occupies its own deep leaf even though both broader
    # regions are negative. Prior-normalized local voting overreacts to that
    # singleton; recursive parent evidence should not.
    y = np.r_[np.zeros(20, dtype=int), 1, np.zeros(20, dtype=int)]

    memory = _ProofPathMemory(predictor, X, y, [0, 1])
    modes = memory.proba_modes(np.array([[-1.0, 1.0], [-1.0, -1.0], [1.0, 0.0]]))
    sparse_leaf = int(_flat_leaf_ids(tree, np.array([[-1.0, 1.0]]))[0])

    assert memory.indexes[0].hierarchy_support[sparse_leaf] == 1
    assert memory.index_report()["hierarchy_compiled"] is True
    assert memory.indexes[0].hierarchy_posterior[sparse_leaf, 0] > 0.5
    assert modes["local_vote"].argmax(1).tolist() == [1, 0, 0]
    assert modes["hierarchical_posterior"].argmax(1).tolist() == [0, 0, 0]
    expected = np.zeros(3, dtype=int)
    assert np.mean(modes["local_vote"].argmax(1) == expected) == 2 / 3
    assert np.mean(modes["hierarchical_posterior"].argmax(1) == expected) == 1.0
    assert np.allclose(modes["hierarchical_posterior"].sum(1), 1.0)
    assert memory.index_report()["hierarchical_facts"] == 4


def test_proof_path_gate_reuses_fold_models_and_requires_oof_rank_gain():
    X, y = _branch_data()
    folds = []
    rows = np.arange(len(y))
    for fold in range(3):
        val = np.r_[np.arange(fold, 120, 3), np.arange(120 + fold, 240, 3)]
        folds.append((np.setdiff1d(rows, val), val))
    model = TabPVN(seed=0)
    model._prep = None
    model._pred = type("Pred", (), {"classes_": [0, 1]})()
    model._temp = 1.0
    precomp = {
        "scores": np.zeros((len(y), 2)),
        "splits": folds,
        "models": [_FlatPredictor(), _FlatPredictor(), _FlatPredictor()],
    }

    weight = model._proof_path_memory_gate(X, y, precomp)

    assert weight > 0.0
    assert model.proof_path_memory_report_[-1]["selected"] is True
    assert model._proof_path_oof_proba is not None
    assert np.array_equal(model._proof_path_oof_proba.argmax(1), np.zeros(len(y), dtype=int))

    # The same path read must be judged against a preceding selected member,
    # not against the raw scores again. This category-like surface already has
    # perfect rank while retaining the certified class, so there is no residual
    # gain for the path member to claim.
    p1 = np.where(y == 1, 0.49, 0.1)
    model._category_memory_oof_proba = np.column_stack([1.0 - p1, p1])

    assert model._proof_path_memory_gate(X, y, precomp) == 0.0
    assert model.proof_path_memory_report_[-1]["selected"] is False


def test_proof_path_gate_selects_hierarchical_memory_from_shared_oof_evidence(monkeypatch):
    class MemoryStub:
        def __init__(self, *_args):
            pass

        def proba_modes(self, X):
            local = np.tile([0.5, 0.5], (len(X), 1))
            positive = np.where(X[:, 0] > 0.5, 0.9, 0.1)
            return {
                "local_vote": local,
                "hierarchical_posterior": np.column_stack([1.0 - positive, positive]),
            }

        def index_report(self):
            return {"anchors": 1, "hierarchical_facts": 3}

    monkeypatch.setattr(base, "_ProofPathMemory", MemoryStub)
    n = 300
    rows = np.arange(n)
    y = np.tile([0, 1], n // 2)
    X = y[:, None].astype(float)
    splits = [(np.setdiff1d(rows, rows[i::3]), rows[i::3]) for i in range(3)]
    model = TabPVN(seed=0)
    model._bayesian_expert_routing = False
    model._prep = None
    model._pred = type("Pred", (), {"classes_": [0, 1]})()
    model._temp = 1.0
    precomp = {
        "scores": np.zeros((n, 2)),
        "splits": splits,
        "models": [object(), object(), object()],
    }

    weight = model._proof_path_memory_gate(X, y, precomp)

    assert weight > 0.0
    assert model._proof_path_memory_mode == "hierarchical_posterior"
    assert model.proof_path_memory_report_[-1]["mode"] == "hierarchical_posterior"
    assert {row["mode"] for row in model.proof_path_memory_report_[-1]["candidates"]} == {
        "local_vote",
        "hierarchical_posterior",
    }
    assert np.array_equal(model._proof_path_oof_proba.argmax(1), np.zeros(n, dtype=int))


def test_proof_path_gate_rejects_an_aggregate_win_with_one_weak_fold(monkeypatch):
    class MemoryStub:
        def __init__(self, *_args):
            pass

        def proba(self, X):
            return np.tile([0.4, 0.6], (len(X), 1))

        def index_report(self):
            return {"anchors": 1}

    # Base score, three base folds, then global/fold scores at each of the
    # five candidate weights. The best global lift is material, but its second
    # independent fold is below the proof-path minimum.
    rank_scores = iter(
        [
            0.900,
            0.900,
            0.900,
            0.900,
            0.907,
            0.904,
            0.902,
            0.904,
            0.906,
            0.904,
            0.904,
            0.904,
            0.905,
            0.904,
            0.904,
            0.904,
            0.904,
            0.904,
            0.904,
            0.904,
            0.903,
            0.904,
            0.904,
            0.904,
        ]
    )
    monkeypatch.setattr(base, "_ProofPathMemory", MemoryStub)
    monkeypatch.setattr(base, "_classification_rank_score", lambda *_args: next(rank_scores))

    n = 300
    rows = np.arange(n)
    y = np.tile([0, 1], n // 2)
    splits = [(np.setdiff1d(rows, rows[i::3]), rows[i::3]) for i in range(3)]
    model = TabPVN(seed=0)
    model._bayesian_expert_routing = False
    model._prep = None
    model._pred = type("Pred", (), {"classes_": [0, 1]})()
    model._temp = 1.0
    precomp = {
        "scores": np.zeros((n, 2)),
        "splits": splits,
        "models": [object(), object(), object()],
    }

    assert model._proof_path_memory_gate(np.zeros((n, 1)), y, precomp) == 0.0
    assert model.proof_path_memory_report_[-1]["selected"] is False


def test_proof_path_blend_cannot_change_the_certified_class():
    class Predictor:
        def _scores(self, X):
            return np.column_stack([-X[:, 0], X[:, 0]])

    class MemoryStub:
        def proba(self, X):
            # Deliberately prefer the opposite label so the projection is exercised.
            return np.column_stack([X[:, 0] > 0.0, X[:, 0] <= 0.0]).astype(float)

    X = np.array([[-2.0], [-0.5], [0.5], [2.0]])
    model = TabPVN(seed=0)
    model._pred = Predictor()
    model._temp = 1.0
    model._proof_path_memory = MemoryStub()
    model._proof_path_memory_w = 0.5

    proba = model._blended_proba(X)

    assert np.array_equal(proba.argmax(1), np.array([0, 0, 1, 1]))
    assert np.allclose(proba.sum(1), 1.0)


def test_hierarchical_path_mode_is_replayed_at_inference():
    class Predictor:
        def _scores(self, X):
            return np.column_stack([-X[:, 0], X[:, 0]])

    class MemoryStub:
        def proba(self, X):
            raise AssertionError("local path vote must not be used")

        def hierarchical_proba(self, X):
            return np.column_stack([X[:, 0] > 0.0, X[:, 0] <= 0.0]).astype(float)

    X = np.array([[-2.0], [-0.5], [0.5], [2.0]])
    model = TabPVN(seed=0)
    model._pred = Predictor()
    model._temp = 1.0
    model._proof_path_memory = MemoryStub()
    model._proof_path_memory_mode = "hierarchical_posterior"
    model._proof_path_memory_w = 0.5

    proba = model._blended_proba(X)

    assert np.array_equal(proba.argmax(1), np.array([0, 0, 1, 1]))
    assert np.allclose(proba.sum(1), 1.0)
