"""Shared candidate-evidence geometry and weighted metric tests."""

import numpy as np
from sklearn.metrics import average_precision_score

from tabpvn.proposers import ClassificationEvidenceWorkspace
from tabpvn.trees import _multiclass_ovo_auc


def test_crossfit_workspace_aligns_classes_weights_and_caches_predictions():
    y = np.array([0] * 80 + [1] * 20)
    weight = np.where(y == 1, 0.25, 1.1875)
    signal = np.linspace(0.0, 1.0, len(y))
    workspace = ClassificationEvidenceWorkspace(
        y,
        seed=3,
        metric="average_precision",
        positive_class=1,
        sample_weight=weight,
    )
    calls = []

    def evaluate(_train, valid):
        calls.append(valid.copy())
        positive = signal[valid]
        # Deliberately reverse class order; the workspace must realign it.
        return np.column_stack([positive, 1.0 - positive]), np.array([1, 0])

    evidence = workspace.evaluate("candidate", evaluate)
    cached = workspace.evaluate("candidate", lambda *_args: (_ for _ in ()).throw(AssertionError()))

    assert cached is evidence
    assert len(calls) == 2
    for fold_index, (_train, valid) in enumerate(workspace.splits):
        expected = average_precision_score(
            y[valid] == 1,
            signal[valid],
            sample_weight=weight[valid],
        )
        assert evidence.fold_scores[fold_index] == expected
    np.testing.assert_allclose(evidence.probabilities[:, 1], signal)


def test_crossfit_workspace_requires_material_gain_on_every_fold():
    y = np.array([0, 1] * 20)
    workspace = ClassificationEvidenceWorkspace(y, seed=2, metric="roc_auc")

    def probabilities(scale):
        def evaluate(_train, valid):
            positive = 0.5 + scale * np.where(y[valid] == 1, 1.0, -1.0)
            return np.column_stack([1.0 - positive, positive]), np.array([0, 1])

        return evaluate

    baseline = workspace.evaluate("baseline", probabilities(0.0))
    candidate = workspace.evaluate("candidate", probabilities(0.2))
    selected, deltas = workspace.accepts(
        candidate,
        baseline,
        min_fold_gain=0.1,
        min_mean_gain=0.1,
    )

    assert selected is True
    assert np.all(deltas == 0.5)


def test_crossfit_workspace_scores_weighted_multiclass_ovo_evidence():
    y = np.tile(np.arange(3), 30)
    weight = np.linspace(0.5, 2.0, len(y))
    probability = np.full((len(y), 3), 0.1)
    probability[np.arange(len(y)), y] = 0.8
    workspace = ClassificationEvidenceWorkspace(
        y,
        seed=9,
        metric="roc_auc",
        sample_weight=weight,
    )

    def evaluate(_train, valid):
        return probability[valid], np.array([0, 1, 2])

    evidence = workspace.evaluate("weighted_multiclass", evaluate)

    for fold_index, (_train, valid) in enumerate(workspace.splits):
        expected = _multiclass_ovo_auc(
            np.log(probability[valid]),
            y[valid],
            weight[valid],
        )
        assert evidence.fold_scores[fold_index] == expected
