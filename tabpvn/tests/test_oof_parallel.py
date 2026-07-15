"""Deterministic and backend-safe coverage for deployed-config OOF fits."""

import threading

import numba
import numpy as np
import pytest

import tabpvn.base as base_module
import tabpvn.trees as trees_module
from tabpvn import TabPVN


def _data(seed=59):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(180, 9))
    margin = 1.5 * X[:, 0] - X[:, 1] + 0.7 * X[:, 2] * X[:, 3] + rng.normal(size=len(X)) * 0.3
    return X, (margin > np.median(margin)).astype(int)


def test_pmap_serializes_numba_workqueue(monkeypatch):
    monkeypatch.setattr(base_module, "_SEARCH_WORKERS", 3)
    monkeypatch.setattr(trees_module, "warm_numba", lambda: None)
    monkeypatch.setattr(numba, "threading_layer", lambda: "workqueue")
    caller = threading.get_ident()

    result = base_module._pmap({i: threading.get_ident for i in range(3)})

    assert list(result) == [0, 1, 2]
    assert set(result.values()) == {caller}


def test_numba_threading_safety_fails_closed_for_workqueue_or_unknown(monkeypatch):
    monkeypatch.setattr(trees_module, "_HAS_NUMBA", True)
    monkeypatch.setattr(numba, "threading_layer", lambda: "workqueue")
    assert not trees_module._numba_threading_is_threadsafe()

    def uninitialized():
        raise ValueError("threading layer is not initialized")

    monkeypatch.setattr(numba, "threading_layer", uninitialized)
    assert not trees_module._numba_threading_is_threadsafe()

    monkeypatch.setattr(numba, "threading_layer", lambda: "tbb")
    assert trees_module._numba_threading_is_threadsafe()


def test_pmap_retains_threads_for_threadsafe_numba_backend(monkeypatch):
    monkeypatch.setattr(base_module, "_SEARCH_WORKERS", 2)
    monkeypatch.setattr(trees_module, "warm_numba", lambda: None)
    monkeypatch.setattr(numba, "threading_layer", lambda: "tbb")
    rendezvous = threading.Barrier(2)

    def worker_id():
        rendezvous.wait(timeout=2)
        return threading.get_ident()

    result = base_module._pmap({0: worker_id, 1: worker_id})

    assert list(result) == [0, 1]
    assert len(set(result.values())) == 2


@pytest.mark.parametrize("linear_leaf", [False, True])
def test_oof_parallelism_preserves_scores_predictions_and_fold_order(monkeypatch, linear_leaf):
    X, y = _data()
    model = TabPVN(
        seed=3,
        boost={"rounds": 80, "lr": 0.05, "depth": 4, "leaf": 12, "patience": 10, "linear_leaf": linear_leaf},
    ).fit(X, y)

    monkeypatch.setattr(base_module, "_SEARCH_WORKERS", 1)
    serial = model._clf_oof(X, y)
    monkeypatch.setattr(base_module, "_SEARCH_WORKERS", 3)
    concurrent = model._clf_oof(X, y)

    np.testing.assert_array_equal(concurrent["scores"], serial["scores"])
    np.testing.assert_array_equal(concurrent["pred"], serial["pred"])
    assert len(concurrent["models"]) == len(serial["models"]) == 3
    assert [len(model.trees_) for model in concurrent["models"]] == [
        len(model.trees_) for model in serial["models"]
    ]
    for (serial_train, serial_valid), (parallel_train, parallel_valid) in zip(
        serial["splits"], concurrent["splits"], strict=False
    ):
        np.testing.assert_array_equal(parallel_train, serial_train)
        np.testing.assert_array_equal(parallel_valid, serial_valid)
