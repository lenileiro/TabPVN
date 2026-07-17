"""Transparent class-conditional compression evidence for text columns.

The map learns a bounded dictionary of discriminative byte phrases. Each
phrase carries a quantized log-likelihood ratio against a class-balanced
global code. Transforming a document is therefore a replayable sum of integer
phrase contributions, not a call to an opaque compressor or a nearest-neighbor
scan over training rows.
"""

from __future__ import annotations

import math
import unicodedata
from collections import Counter
from typing import Any, Self

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray

_BITS_SCALE = 256
_MIN_NGRAM = 3
_MAX_NGRAM = 5
_MAX_FEATURES = 2_048
_MAX_REFERENCE_BYTES = 64 * 1_024
_MAX_TOTAL_REFERENCE_BYTES = 512 * 1_024
_MIN_REFERENCE_BYTES = 1_024
_MAX_REFERENCE_ROWS = 4_096
_MAX_DOCUMENT_BYTES = 1_024
_TRANSFORM_BATCH_ROWS = 8_192
_MIN_DOCUMENT_FREQUENCY = 2
_MIN_WEIGHT_SPAN_BITS = 0.5
_MAX_WEIGHT_BITS = 12.0
_SMOOTHING = 0.5


def _as_vector(values: ArrayLike, name: str) -> NDArray[Any]:
    array = np.asarray(values, dtype=object)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def _canonical_bytes(value: Any, max_bytes: int) -> bytes:
    """Return a bounded canonical byte sequence with unambiguous boundaries."""
    if not isinstance(value, str):
        return b"\xff\xfe"
    payload = unicodedata.normalize("NFKC", value).encode("utf-8", errors="replace")[:max_bytes]
    # 0xfe and 0xff never occur in valid UTF-8, so phrases cannot confuse a
    # document boundary with user text.
    return b"\xff" + payload + b"\xfe"


def _pack_ngram(document: bytes, start: int, length: int) -> int:
    key = length << 56
    for offset in range(length):
        key |= document[start + offset] << (8 * offset)
    return key


@njit(cache=True)
def _find_key(keys: NDArray[np.uint64], key: np.uint64) -> int:
    low = 0
    high = len(keys)
    while low < high:
        middle = (low + high) // 2
        if keys[middle] < key:
            low = middle + 1
        else:
            high = middle
    if low < len(keys) and keys[low] == key:
        return low
    return -1


@njit(cache=True)
def _score_batch(
    data: NDArray[np.uint8],
    offsets: NDArray[np.int64],
    keys: NDArray[np.uint64],
    weights: NDArray[np.int16],
) -> NDArray[np.float64]:
    n_rows = len(offsets) - 1
    n_classes = weights.shape[1]
    output = np.zeros((n_rows, n_classes + 2), dtype=np.float64)
    for row in range(n_rows):
        start = offsets[row]
        end = offsets[row + 1]
        document_length = end - start
        body_length = max(1, document_length - 2)
        class_score = np.zeros(n_classes, dtype=np.int64)
        matched = 0
        positions = 0
        for length in range(_MIN_NGRAM, _MAX_NGRAM + 1):
            n_positions = document_length - length + 1
            if n_positions <= 0:
                continue
            positions += n_positions
            for position in range(start, end - length + 1):
                key = np.uint64(length) << np.uint64(56)
                for offset in range(length):
                    key |= np.uint64(data[position + offset]) << np.uint64(8 * offset)
                phrase = _find_key(keys, key)
                if phrase < 0:
                    continue
                matched += 1
                for class_index in range(n_classes):
                    class_score[class_index] += weights[phrase, class_index]

        denominator = float(_BITS_SCALE * body_length)
        best = -np.inf
        second = -np.inf
        for class_index in range(n_classes):
            score = class_score[class_index] / denominator
            output[row, class_index] = score
            if score > best:
                second = best
                best = score
            elif score > second:
                second = score
        output[row, n_classes] = best - second
        output[row, n_classes + 1] = matched / max(1, positions)
    return output


class CompressionEvidenceMap:
    """Bounded class-conditional byte-phrase evidence map.

    Reference material is sampled and then truncated to the same byte budget
    for every class. Phrase weights are quantized bits saved relative to a
    class-balanced global phrase code.
    """

    def __init__(
        self,
        *,
        seed: int = 0,
        max_features: int = _MAX_FEATURES,
        max_reference_bytes: int = _MAX_REFERENCE_BYTES,
        max_document_bytes: int = _MAX_DOCUMENT_BYTES,
    ) -> None:
        if max_features <= 0:
            raise ValueError("max_features must be positive")
        if max_reference_bytes < _MIN_REFERENCE_BYTES:
            raise ValueError(f"max_reference_bytes must be at least {_MIN_REFERENCE_BYTES}")
        if max_document_bytes < _MAX_NGRAM:
            raise ValueError(f"max_document_bytes must be at least {_MAX_NGRAM}")
        self.seed = int(seed)
        self.max_features = int(max_features)
        self.max_reference_bytes = int(max_reference_bytes)
        self.max_document_bytes = int(max_document_bytes)

    @property
    def n_features_out_(self) -> int:
        self._require_fitted()
        return len(self.classes_) + 2

    def _require_fitted(self) -> None:
        if not hasattr(self, "classes_"):
            raise RuntimeError("compression evidence map is not fitted")

    def _sample_documents(
        self,
        values: NDArray[Any],
        target: NDArray[Any],
        label: Any,
        class_index: int,
    ) -> list[bytes]:
        rows = np.flatnonzero(target == label)
        if len(rows) > _MAX_REFERENCE_ROWS:
            rng = np.random.default_rng(self.seed + 104_729 * (class_index + 1))
            rows = rng.choice(rows, _MAX_REFERENCE_ROWS, replace=False)
        return [_canonical_bytes(values[row], self.max_document_bytes) for row in rows]

    @staticmethod
    def _truncate_documents(documents: list[bytes], budget: int) -> list[bytes]:
        selected = []
        remaining = budget
        for document in documents:
            if remaining <= 0:
                break
            payload = document[1:-1]
            if not payload:
                continue
            take = min(len(payload), remaining)
            selected.append(b"\xff" + payload[:take] + b"\xfe")
            remaining -= take
        return selected

    @staticmethod
    def _count_phrases(
        references: list[list[bytes]],
    ) -> tuple[list[Counter[int]], list[Counter[int]], NDArray[np.int64]]:
        n_classes = len(references)
        counts = [Counter[int]() for _ in range(n_classes)]
        documents = [Counter[int]() for _ in range(n_classes)]
        positions: NDArray[np.int64] = np.zeros((n_classes, _MAX_NGRAM + 1), dtype=np.int64)
        for class_index, class_documents in enumerate(references):
            for document in class_documents:
                seen: set[int] = set()
                for length in range(_MIN_NGRAM, _MAX_NGRAM + 1):
                    n_positions = len(document) - length + 1
                    if n_positions <= 0:
                        continue
                    positions[class_index, length] += n_positions
                    for start in range(n_positions):
                        key = _pack_ngram(document, start, length)
                        counts[class_index][key] += 1
                        seen.add(key)
                documents[class_index].update(seen)
        return counts, documents, positions

    @staticmethod
    def _candidate_weights(
        key: int,
        counts: list[Counter[int]],
        positions: NDArray[np.int64],
        vocabulary_sizes: dict[int, int],
    ) -> NDArray[np.float64]:
        length = key >> 56
        vocabulary = vocabulary_sizes[length]
        probabilities = np.array(
            [
                (class_counts.get(key, 0) + _SMOOTHING)
                / (positions[class_index, length] + _SMOOTHING * vocabulary)
                for class_index, class_counts in enumerate(counts)
            ],
            dtype=float,
        )
        global_probability = float(probabilities.mean())
        return np.log2(probabilities / global_probability)

    def _select_phrases(
        self,
        counts: list[Counter[int]],
        document_counts: list[Counter[int]],
        positions: NDArray[np.int64],
    ) -> tuple[NDArray[np.uint64], NDArray[np.int16]]:
        candidates = set().union(*(class_counts.keys() for class_counts in counts))
        vocabulary_sizes = {
            length: sum((key >> 56) == length for key in candidates)
            for length in range(_MIN_NGRAM, _MAX_NGRAM + 1)
        }
        ranked: list[tuple[float, int, NDArray[np.float64]]] = []
        for key in candidates:
            document_frequency = sum(class_counts.get(key, 0) for class_counts in document_counts)
            if document_frequency < _MIN_DOCUMENT_FREQUENCY:
                continue
            phrase_weights = self._candidate_weights(key, counts, positions, vocabulary_sizes)
            span = float(phrase_weights.max() - phrase_weights.min())
            if span < _MIN_WEIGHT_SPAN_BITS:
                continue
            length = key >> 56
            rank = span * math.sqrt(document_frequency) * (1.0 + 0.15 * (length - _MIN_NGRAM))
            ranked.append((rank, key, phrase_weights))
        ranked.sort(key=lambda candidate: (-candidate[0], candidate[1]))
        selected = ranked[: self.max_features]
        selected.sort(key=lambda candidate: candidate[1])
        keys = np.asarray([key for _, key, _ in selected], dtype=np.uint64)
        weights = np.asarray(
            [
                np.rint(np.clip(phrase_weights, -_MAX_WEIGHT_BITS, _MAX_WEIGHT_BITS) * _BITS_SCALE).astype(
                    np.int16
                )
                for _, _, phrase_weights in selected
            ],
            dtype=np.int16,
        )
        if not selected:
            weights = np.zeros((0, len(counts)), dtype=np.int16)
        return keys, weights

    def fit(self, values: ArrayLike, y: ArrayLike) -> Self:
        """Fit a class-balanced phrase dictionary from bounded references."""
        documents = _as_vector(values, "values")
        target = _as_vector(y, "y")
        if len(documents) != len(target):
            raise ValueError("values and y must have the same length")
        self.classes_ = np.unique(target)
        if len(self.classes_) < 2:
            raise ValueError("compression evidence requires at least two classes")

        sampled = [
            self._sample_documents(documents, target, label, class_index)
            for class_index, label in enumerate(self.classes_)
        ]
        available_bytes = [sum(max(0, len(document) - 2) for document in group) for group in sampled]
        class_budget = max(_MIN_REFERENCE_BYTES, _MAX_TOTAL_REFERENCE_BYTES // len(self.classes_))
        self.reference_bytes_ = min(
            self.max_reference_bytes,
            class_budget,
            min(available_bytes, default=0),
        )
        self.reference_total_bytes_ = self.reference_bytes_ * len(self.classes_)
        self.reference_documents_ = tuple(0 for _ in self.classes_)
        self.keys_: NDArray[np.uint64] = np.zeros(0, dtype=np.uint64)
        self.weights_: NDArray[np.int16] = np.zeros((0, len(self.classes_)), dtype=np.int16)
        self.is_active_ = False
        self.inactive_reason_: str | None = "insufficient_reference_bytes"
        if self.reference_bytes_ < _MIN_REFERENCE_BYTES:
            return self

        references = [self._truncate_documents(group, self.reference_bytes_) for group in sampled]
        self.reference_documents_ = tuple(len(group) for group in references)
        counts, document_counts, positions = self._count_phrases(references)
        self.keys_, self.weights_ = self._select_phrases(counts, document_counts, positions)
        self.is_active_ = bool(len(self.keys_))
        self.inactive_reason_ = None if self.is_active_ else "no_repeated_discriminative_phrases"
        return self

    def transform(self, values: ArrayLike) -> NDArray[np.float64]:
        """Return class bits-saved scores, top-two margin, and phrase coverage."""
        self._require_fitted()
        documents = _as_vector(values, "values")
        output: NDArray[np.float64] = np.zeros((len(documents), self.n_features_out_), dtype=float)
        if not self.is_active_ or not len(documents):
            return output
        for start in range(0, len(documents), _TRANSFORM_BATCH_ROWS):
            stop = min(start + _TRANSFORM_BATCH_ROWS, len(documents))
            encoded = [_canonical_bytes(value, self.max_document_bytes) for value in documents[start:stop]]
            offsets: NDArray[np.int64] = np.zeros(len(encoded) + 1, dtype=np.int64)
            offsets[1:] = np.cumsum([len(document) for document in encoded], dtype=np.int64)
            data = np.frombuffer(b"".join(encoded), dtype=np.uint8)
            output[start:stop] = _score_batch(data, offsets, self.keys_, self.weights_)
        return output

    def feature_names(self, column: Any) -> list[str]:
        """Return stable feature names without entering the bag-of-words namespace."""
        self._require_fitted()
        prefix = str(column)
        names = [f"{prefix}__compression_bits={label}" for label in self.classes_]
        return names + [f"{prefix}__compression_margin", f"{prefix}__compression_coverage"]


__all__ = ["CompressionEvidenceMap"]
