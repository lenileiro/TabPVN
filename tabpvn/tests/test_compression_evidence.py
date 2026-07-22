"""Leakage, selectivity, and runtime contracts for compression evidence."""

import pickle

import numpy as np
import pandas as pd

from tabpvn import TabPVN
from tabpvn.compression_evidence import CompressionEvidenceMap
from tabpvn.preprocessing import _Preprocessor


def _order_corpus(rows_per_class=80):
    forward = [f"alpha first then omega route sample{index} signal marker" for index in range(rows_per_class)]
    reverse = [f"omega first then alpha route sample{index} signal marker" for index in range(rows_per_class)]
    return (
        pd.DataFrame({"message": forward + reverse}),
        np.array([0] * rows_per_class + [1] * rows_per_class),
    )


def test_phrase_map_is_class_balanced_and_order_sensitive():
    X, y = _order_corpus()
    evidence = CompressionEvidenceMap(max_reference_bytes=2_048).fit(X["message"], y)

    transformed = evidence.transform(
        np.array(
            [
                "alpha first then omega route signal marker",
                "omega first then alpha route signal marker",
            ],
            dtype=object,
        )
    )

    assert evidence.is_active_
    assert evidence.reference_bytes_ == 2_048
    assert len(evidence.keys_) <= evidence.max_features
    assert np.all(evidence.keys_[1:] > evidence.keys_[:-1])
    assert transformed[0, 0] > transformed[0, 1]
    assert transformed[1, 1] > transformed[1, 0]
    assert np.all(transformed[:, -2:] > 0.0)


def test_preprocessor_crossfits_and_selects_sequence_evidence():
    X, y = _order_corpus()
    preprocessor = _Preprocessor(task="classification")

    training = preprocessor.fit_transform(X, y)
    compression_columns = preprocessor.compression_indices["message"]
    inference = preprocessor.transform(X)

    assert preprocessor.byte_cols == ["message"]
    assert preprocessor.compression_enabled == {"message": True}
    assert preprocessor.compression_report[-1]["selected"] is True
    assert min(preprocessor.compression_report[-1]["fold_scores"]) > 0.99
    assert not any("token" in str(name) or "~" in str(name) for name in preprocessor.names)
    assert not np.allclose(
        training[:, compression_columns],
        inference[:, compression_columns],
    )
    assert np.isfinite(training).all()


def test_preprocessor_rejects_balanced_text_without_sequence_signal():
    documents = [f"alpha beta gamma delta route sample{index} signal marker" for index in range(100)]
    X = pd.DataFrame({"message": documents + documents})
    y = np.array([0] * len(documents) + [1] * len(documents))

    preprocessor = _Preprocessor(task="classification")
    encoded = preprocessor.fit_transform(X, y)

    assert preprocessor.compression_enabled == {}
    assert preprocessor.compression_maps == {}
    assert preprocessor.compression_report[0]["selected"] is False
    assert preprocessor.compression_report[0]["reason"] == "no_repeated_discriminative_phrases"
    assert preprocessor.compression_report[0]["reference_bytes_per_class"] > 0
    assert not any("__compression" in str(name) for name in preprocessor.names)
    assert preprocessor.names == ["message__freq"]
    assert encoded.shape[1] == len(preprocessor.names)


def test_preprocessor_selects_multiclass_sequence_evidence():
    orders = (
        ("red", "green", "blue"),
        ("green", "blue", "red"),
        ("blue", "red", "green"),
    )
    documents = []
    labels = []
    for label, order in enumerate(orders):
        documents.extend(
            [
                f"{order[0]} then {order[1]} then {order[2]} route sample{index} signal marker"
                for index in range(90)
            ]
        )
        labels.extend([label] * 90)
    X = pd.DataFrame({"message": documents})
    y = np.asarray(labels)

    preprocessor = _Preprocessor(task="classification")
    encoded = preprocessor.fit_transform(X, y)
    compression_columns = preprocessor.compression_indices["message"]

    assert preprocessor.compression_report[-1]["selected"] is True
    assert preprocessor.compression_report[-1]["metric"] == "macro_ovo_auc"
    assert min(preprocessor.compression_report[-1]["fold_scores"]) > 0.99
    assert (encoded[:, compression_columns[:3]].argmax(axis=1) == y).mean() > 0.99


def test_tabpvn_deploys_compression_features_inside_certified_booster():
    X, y = _order_corpus()
    model = TabPVN(
        seed=0,
        boost={"rounds": 30, "depth": 2, "leaf": 4, "patience": 8},
    ).fit(X, y)

    assert model.compression_evidence_report_[-1]["selected"] is True
    assert any("__compression_bits" in str(name) for name in model.feature_names_)
    assert model.score(X, y) == 1.0
    assert model.certify(X.iloc[:20]) == 1.0
    groups, metadata, _widen = model._reason_groups()
    assert sorted(column for group in groups for column in group) == list(range(len(model.feature_names_)))
    assert any(group["kind"] == "byte_evidence" for group in metadata)
    assert "byte evidence" in model.reason(X.iloc[:1], 0)["rule"]
    restored = pickle.loads(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
    np.testing.assert_array_equal(restored.predict(X.iloc[:20]), model.predict(X.iloc[:20]))
