"""Deterministic preprocessing for TabPVN's tabular estimators.

This module owns raw-table schema compilation. It intentionally contains no
predictor imports, which keeps preprocessing reusable by both ``TabPVN`` and
``CertifiedAttention`` without creating a dependency cycle.
"""

from __future__ import annotations

import re
import warnings
from collections import Counter
from typing import Any, Literal, Self

import numpy as np
from numba import njit
from numpy.typing import NDArray

from tabpvn.compression_evidence import CompressionEvidenceMap

Task = Literal["auto", "classification", "regression"]


@njit(cache=True, nogil=True)
def _causal_target_scan(
    order: NDArray[np.int64],
    codes: NDArray[np.int64],
    groups: NDArray[np.int64],
    target: NDArray[np.float64],
    smooth: float,
    initial_prior: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Emit smoothed category statistics before updating one timestamp batch."""
    category_count = int(codes.max()) + 1
    width = target.shape[1]
    counts: NDArray[np.float64] = np.zeros(category_count, dtype=np.float64)
    sums = np.zeros((category_count, width), dtype=np.float64)
    global_sum = np.zeros(width, dtype=np.float64)
    global_count = 0.0
    output = np.zeros((len(codes), width), dtype=np.float64)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        timestamp = groups[order[cursor]]
        while end < len(order) and groups[order[end]] == timestamp:
            end += 1
        prior = initial_prior if global_count == 0.0 else global_sum / global_count
        for position in range(cursor, end):
            row = order[position]
            code = codes[row]
            denominator = counts[code] + smooth
            for column in range(width):
                output[row, column] = (sums[code, column] + smooth * prior[column]) / denominator
        for position in range(cursor, end):
            row = order[position]
            code = codes[row]
            counts[code] += 1.0
            global_count += 1.0
            for column in range(width):
                value = target[row, column]
                sums[code, column] += value
                global_sum[column] += value
        cursor = end
    return output


@njit(cache=True, nogil=True)
def _causal_gaussian_target_scan(
    order: NDArray[np.int64],
    codes: NDArray[np.int64],
    groups: NDArray[np.int64],
    target: NDArray[np.float64],
    smooth: float,
) -> NDArray[np.float64]:
    """Emit a variance-aware Gaussian posterior before updating one timestamp batch."""
    category_count = int(codes.max()) + 1
    counts: NDArray[np.float64] = np.zeros(category_count, dtype=np.float64)
    sums: NDArray[np.float64] = np.zeros(category_count, dtype=np.float64)
    sum_squares: NDArray[np.float64] = np.zeros(category_count, dtype=np.float64)
    global_sum = 0.0
    global_sum_squares = 0.0
    global_count = 0.0
    output: NDArray[np.float64] = np.zeros((len(codes), 1), dtype=np.float64)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        timestamp = groups[order[cursor]]
        while end < len(order) and groups[order[end]] == timestamp:
            end += 1

        prior = 0.0 if global_count == 0.0 else global_sum / global_count
        variance = 0.0
        if global_count > 1.0:
            variance = (global_sum_squares - global_sum * global_sum / global_count) / (global_count - 1.0)
            variance = max(variance, 1e-12 * max(1.0, abs(prior)) ** 2)

        for position in range(cursor, end):
            row = order[position]
            code = codes[row]
            count = counts[code]
            if count == 0.0:
                output[row, 0] = prior
            elif variance == 0.0:
                output[row, 0] = (sums[code] + smooth * prior) / (count + smooth)
            else:
                raw_mean = sums[code] / count
                within_sum_squares = max(
                    sum_squares[code] - sums[code] * sums[code] / count,
                    0.0,
                )
                local_variance = (within_sum_squares + smooth * variance) / (max(count - 1.0, 0.0) + smooth)
                local_variance = min(
                    variance * 1e3,
                    max(variance * 1e-3, local_variance),
                )
                prior_precision = smooth / variance
                data_precision = count / local_variance
                output[row, 0] = (prior_precision * prior + data_precision * raw_mean) / (
                    prior_precision + data_precision
                )

        for position in range(cursor, end):
            row = order[position]
            code = codes[row]
            value = target[row]
            counts[code] += 1.0
            sums[code] += value
            sum_squares[code] += value * value
            global_count += 1.0
            global_sum += value
            global_sum_squares += value * value
        cursor = end
    return output


def _is_classification(y: Any) -> bool:
    """Return whether a target should use the finite-class prediction path."""
    values = np.asarray(y)
    if values.dtype.kind == "b":
        return True
    if values.dtype.kind in "iu":
        return len(np.unique(values)) <= 20
    if values.dtype.kind == "f":
        return bool(np.allclose(values, np.round(values)) and len(np.unique(values)) <= 20)
    return True


def target_encode(
    train_ids: Any,
    y: Any,
    test_ids: Any | None = None,
    folds: int = 5,
    smooth: float = 20,
    seed: int = 0,
) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build leak-safe smoothed target encodings for high-cardinality IDs.

    Training rows are encoded out of fold. When ``test_ids`` is supplied, a
    second vector encoded from all training rows is returned.
    """
    import pandas as pd
    from sklearn.model_selection import KFold

    train_ids = np.asarray(train_ids)
    target = np.asarray(y, dtype=float)
    if train_ids.ndim != 1 or target.ndim != 1:
        raise ValueError("train_ids and y must be one-dimensional")
    if len(train_ids) != len(target):
        raise ValueError("train_ids and y must have the same length")
    if not 2 <= folds <= len(target):
        raise ValueError("folds must be between 2 and the number of training rows")
    if smooth < 0:
        raise ValueError("smooth must be non-negative")
    if not np.isfinite(target).all():
        raise ValueError("y must contain only finite values")

    global_mean = float(target.mean())
    encoded_train: NDArray[np.float64] = np.full(len(target), global_mean, dtype=float)
    for train, validation in KFold(folds, shuffle=True, random_state=seed).split(train_ids):
        grouped = pd.Series(target[train]).groupby(train_ids[train])
        count = grouped.count()
        mean = grouped.mean()
        encoding = (mean * count + global_mean * smooth) / (count + smooth)
        encoded_train[validation] = (
            pd.Series(train_ids[validation]).map(encoding).fillna(global_mean).to_numpy()
        )
    if test_ids is None:
        return encoded_train

    grouped = pd.Series(target).groupby(train_ids)
    count = grouped.count()
    mean = grouped.mean()
    encoding = (mean * count + global_mean * smooth) / (count + smooth)
    encoded_test = pd.Series(np.asarray(test_ids)).map(encoding).fillna(global_mean).to_numpy()
    return encoded_train, encoded_test


_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")


def _tokenize(value: Any) -> list[str]:
    return _TOKEN_RE.findall(value.lower()) if isinstance(value, str) else []


def _looks_like_structured_identifier(series: Any) -> bool:
    """Return whether a high-cardinality string column contains reusable ID parts."""
    values = series.dropna().astype(str)
    if len(values) < 8 or values.nunique() < 8 or values.nunique() / len(values) < 0.5:
        return False
    sample = values.sample(min(len(values), 1_000), random_state=0)
    if sample.str.contains(r"[-_:/]", regex=True).mean() < 0.8:
        return False
    documents = [_tokenize(value) for value in sample]
    token_counts = np.asarray([len(tokens) for tokens in documents], dtype=int)
    if np.mean((token_counts >= 2) & (token_counts <= 8)) < 0.8:
        return False
    frequency = Counter(token for tokens in documents for token in set(tokens))
    maximum_reusable_frequency = 0.8 * len(documents)
    return any(2 <= count <= maximum_reusable_frequency for count in frequency.values())


class _TextFeaturizer:
    """Deterministic binary bag-of-words featurizer for one text column."""

    def __init__(self, max_features: int = 300):
        if max_features <= 0:
            raise ValueError("max_features must be positive")
        self.max_features = int(max_features)

    def _docsets(self, series: Any) -> tuple[NDArray[Any], list[set[str]]]:
        values = series.astype("object").where(series.notna(), "").to_numpy()
        return values, [set(_tokenize(value)) for value in values]

    def fit(
        self,
        series: Any,
        y: Any = None,
        is_classification: bool | None = None,
    ) -> Self:
        _, documents = self._docsets(series)
        n_documents = max(1, len(documents))
        document_frequency: dict[str, int] = {}
        for tokens in documents:
            for token in tokens:
                document_frequency[token] = document_frequency.get(token, 0) + 1

        target_is_classification = (
            _is_classification(np.asarray(y))
            if y is not None and is_classification is None
            else bool(is_classification)
        )
        is_regression = y is not None and not target_is_classification
        minimum_frequency = 2 if is_regression else max(2, int(0.005 * n_documents))
        maximum_frequency = (0.9 if is_regression else 0.5) * n_documents
        candidates = [
            token
            for token, count in document_frequency.items()
            if minimum_frequency <= count <= maximum_frequency
        ]
        candidate_set = set(candidates)

        if candidates and y is not None and target_is_classification:
            target = np.asarray(y)
            classes = sorted(set(target.tolist()))
            class_index = {value: index for index, value in enumerate(classes)}
            encoded_target = np.array([class_index[value] for value in target])
            class_count = len(classes)
            totals = np.bincount(encoded_target, minlength=class_count).astype(float)
            present_by_class = {token: np.zeros(class_count) for token in candidates}
            for row, tokens in enumerate(documents):
                target_class = encoded_target[row]
                for token in tokens:
                    if token in candidate_set:
                        present_by_class[token][target_class] += 1.0

            def chi_squared(token: str) -> float:
                observed_present = present_by_class[token]
                present = observed_present.sum()
                observed_absent = totals - observed_present
                expected_present = present * totals / n_documents
                expected_absent = (n_documents - present) * totals / n_documents
                return float(
                    np.sum((observed_present - expected_present) ** 2 / (expected_present + 1e-9))
                    + np.sum((observed_absent - expected_absent) ** 2 / (expected_absent + 1e-9))
                )

            candidates.sort(key=lambda token: (-chi_squared(token), token))
        elif candidates and y is not None:
            centered_target = np.asarray(y, dtype=float) - float(np.mean(y))
            target_scale = float(np.std(centered_target)) + 1e-9
            target_sum = dict.fromkeys(candidates, 0.0)
            for row, tokens in enumerate(documents):
                for token in tokens:
                    if token in candidate_set:
                        target_sum[token] += centered_target[row]

            def correlation(token: str) -> float:
                present = document_frequency[token]
                denominator = np.sqrt(present * (n_documents - present)) * target_scale + 1e-9
                return abs(target_sum[token]) / denominator

            candidates.sort(key=lambda token: (-correlation(token), token))
        else:
            candidates.sort(key=lambda token: (-document_frequency[token], token))

        self.vocab = candidates[: self.max_features]
        self.tok2col = {token: index for index, token in enumerate(self.vocab)}
        return self

    def transform(self, series: Any) -> NDArray[np.float64]:
        values, documents = self._docsets(series)
        output = np.zeros((len(values), len(self.vocab)))
        for row, tokens in enumerate(documents):
            for token in tokens:
                column = self.tok2col.get(token, -1)
                if column >= 0:
                    output[row, column] = 1.0
        return output


class _DateTimeFeaturizer:
    """Compact, deterministic order and calendar facts for one datetime column."""

    _COMPONENT_NAMES = (
        "elapsed_days",
        "annual_sin",
        "annual_cos",
        "weekly_sin",
        "weekly_cos",
        "daily_sin",
        "daily_cos",
        "isna",
    )

    @staticmethod
    def _coerce(series: Any) -> Any:
        import pandas as pd

        if pd.api.types.is_numeric_dtype(series.dtype) and not pd.api.types.is_datetime64_any_dtype(
            series.dtype
        ):
            raise TypeError("fitted datetime columns cannot be supplied as ambiguous numeric timestamps")
        options = {"format": "mixed"} if series.dtype == object else {}
        return pd.to_datetime(series, errors="coerce", utc=True, **options)

    @staticmethod
    def _cycle(position: NDArray[np.float64], period: float) -> tuple[Any, Any]:
        angle = 2.0 * np.pi * position / period
        return np.sin(angle), np.cos(angle)

    def _components(self, series: Any) -> tuple[NDArray[np.float64], NDArray[np.bool_], Any]:
        import pandas as pd

        timestamps = self._coerce(series)
        missing = timestamps.isna().to_numpy()
        origin = pd.Timestamp(self.origin_ns_, unit="ns", tz="UTC")
        filled = timestamps.fillna(origin)
        values_ns = filled.astype("int64").to_numpy(dtype=np.int64, copy=False)
        elapsed_days = (values_ns.astype(np.float64) - float(self.origin_ns_)) / 86_400_000_000_000.0
        hour = (
            filled.dt.hour.to_numpy(dtype=float)
            + filled.dt.minute.to_numpy(dtype=float) / 60.0
            + filled.dt.second.to_numpy(dtype=float) / 3_600.0
            + filled.dt.microsecond.to_numpy(dtype=float) / 3_600_000_000.0
        )
        annual_position = filled.dt.dayofyear.to_numpy(dtype=float) - 1.0 + hour / 24.0
        weekly_position = filled.dt.dayofweek.to_numpy(dtype=float) + hour / 24.0
        annual_sin, annual_cos = self._cycle(annual_position, 365.2425)
        weekly_sin, weekly_cos = self._cycle(weekly_position, 7.0)
        daily_sin, daily_cos = self._cycle(hour, 24.0)
        block = np.column_stack(
            (
                elapsed_days,
                annual_sin,
                annual_cos,
                weekly_sin,
                weekly_cos,
                daily_sin,
                daily_cos,
                missing.astype(float),
            )
        )
        return block, missing, filled

    def fit(self, series: Any) -> Self:
        timestamps = self._coerce(series)
        valid = timestamps.notna().to_numpy()
        if valid.any():
            valid_ns = np.sort(timestamps[valid].astype("int64").to_numpy(dtype=np.int64, copy=False))
            self.origin_ns_ = int(valid_ns[len(valid_ns) // 2])
        else:
            self.origin_ns_ = 0

        _, missing, filled = self._components(series)
        selected = []
        if valid.any():
            selected.append(0)
            valid_filled = filled[valid]
            if valid_filled.dt.normalize().nunique() > 1:
                selected.extend((1, 2))
            if valid_filled.dt.dayofweek.nunique() > 1:
                selected.extend((3, 4))
            seconds = valid_filled.dt.hour * 3_600 + valid_filled.dt.minute * 60 + valid_filled.dt.second
            if seconds.nunique() > 1:
                selected.extend((5, 6))
        if missing.any():
            selected.append(7)
        self.selected_ = tuple(selected)
        self.n_features_out_ = len(self.selected_)
        return self

    def transform(self, series: Any) -> NDArray[np.float64]:
        block, _, _ = self._components(series)
        return block[:, self.selected_]

    def feature_names(self, column: Any) -> list[str]:
        return [f"{column}__datetime_{self._COMPONENT_NAMES[index]}" for index in self.selected_]


class _Preprocessor:
    """Compile a raw DataFrame into the numeric matrix used by TabPVN."""

    def __init__(
        self,
        target_encoding: bool = True,
        task: Task = "auto",
        compression_evidence: bool = True,
        gaussian_target_statistics: bool = False,
    ):
        if task not in {"auto", "classification", "regression"}:
            raise ValueError("task must be 'auto', 'classification', or 'regression'")
        self.target_encoding_enabled = bool(target_encoding)
        self.compression_evidence_enabled = bool(compression_evidence)
        self.gaussian_target_statistics_enabled = bool(gaussian_target_statistics)
        self.task = task

    @staticmethod
    def _category_keys(series: Any) -> NDArray[Any]:
        return series.astype("object").where(series.notna(), "__nan__").to_numpy(dtype=object)

    @staticmethod
    def _target_stats(
        keys: Any,
        y: Any,
        is_classification: bool,
        classes: Any,
        smooth: float,
        gaussian: bool = False,
    ) -> dict[str, Any]:
        """Return smoothed category statistics computed from supplied rows."""
        import pandas as pd

        keys = np.asarray(keys, dtype=object)
        target = np.asarray(y)
        codes, levels = pd.factorize(keys, sort=False)
        counts = np.bincount(codes, minlength=len(levels)).astype(float)
        if is_classification:
            outputs = np.asarray(classes)[1:]
            prior = np.array([(target == value).mean() for value in outputs], dtype=float)
            sums = np.column_stack(
                [
                    np.bincount(
                        codes,
                        weights=(target == value).astype(float),
                        minlength=len(levels),
                    )
                    for value in outputs
                ]
            )
            labels = [str(value) for value in outputs]
        else:
            prior = np.array([float(np.mean(target))])
            sums = np.bincount(codes, weights=target.astype(float), minlength=len(levels))[:, None]
            labels = ["mean"]
        values = (sums + smooth * prior[None, :]) / (counts[:, None] + smooth)
        if gaussian and not is_classification:
            variance = float(np.var(target.astype(float), ddof=1)) if len(target) > 1 else 0.0
            variance = max(variance, 1e-12 * max(1.0, abs(float(prior[0]))) ** 2)
            sum_squares = np.bincount(
                codes,
                weights=target.astype(float) ** 2,
                minlength=len(levels),
            )
            raw_mean = sums[:, 0] / counts
            within_sum_squares = np.maximum(
                sum_squares - sums[:, 0] ** 2 / counts,
                0.0,
            )
            local_variance = (within_sum_squares + smooth * variance) / (
                np.maximum(counts - 1.0, 0.0) + smooth
            )
            local_variance = np.clip(local_variance, variance * 1e-3, variance * 1e3)
            prior_precision = smooth / variance
            data_precision = counts / local_variance
            values = (
                (prior_precision * prior[0] + data_precision * raw_mean) / (prior_precision + data_precision)
            )[:, None]
            labels = ["gaussian_mean"]
        return {
            "table": {level: values[index] for index, level in enumerate(levels)},
            "prior": prior,
            "labels": labels,
        }

    @staticmethod
    def _apply_target_stats(keys: Any, stats: dict[str, Any]) -> NDArray[np.float64]:
        prior = stats["prior"]
        return np.stack([stats["table"].get(key, prior) for key in keys], axis=0)

    def _causal_target_values(
        self,
        keys: Any,
        target: NDArray[Any],
        groups: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        """Encode each row from labels at strictly earlier timestamps."""
        import pandas as pd

        codes, _levels = pd.factorize(np.asarray(keys, dtype=object), sort=False)
        if np.any(codes < 0):
            raise ValueError("causal target encoding received an unsupported category value")
        rows: NDArray[np.int64] = np.arange(len(groups), dtype=np.int64)
        order = np.lexsort((rows, groups)).astype(np.int64, copy=False)
        if self._target_is_classification:
            classes = self._target_classes
            if classes is None:
                raise ValueError("classification target classes are unavailable")
            outputs = np.asarray(classes)[1:]
            target_matrix = np.column_stack([(target == value).astype(float) for value in outputs])
            initial_prior: NDArray[np.float64] = np.full(len(outputs), 1.0 / len(classes), dtype=float)
        else:
            if self.gaussian_target_statistics_enabled:
                return _causal_gaussian_target_scan(
                    order,
                    np.asarray(codes, dtype=np.int64),
                    np.asarray(groups, dtype=np.int64),
                    np.asarray(target, dtype=float),
                    self._target_smoothing,
                )
            target_matrix = np.asarray(target, dtype=float)[:, None]
            initial_prior = np.zeros(1, dtype=float)
        return _causal_target_scan(
            order,
            np.asarray(codes, dtype=np.int64),
            np.asarray(groups, dtype=np.int64),
            np.asarray(target_matrix, dtype=float),
            self._target_smoothing,
            initial_prior,
        )

    def _target_signal(self, encoded: NDArray[np.float64], y: Any) -> bool:
        """Return whether an encoding has meaningful out-of-fold association."""
        target = np.asarray(y)
        try:
            if self._target_is_classification:
                from sklearn.metrics import roc_auc_score

                classes = self._target_classes
                if classes is None:
                    return False
                scores = []
                for column, target_class in enumerate(classes[1:]):
                    binary_target = (target == target_class).astype(int)
                    if 0 < binary_target.sum() < len(binary_target):
                        scores.append(abs(float(roc_auc_score(binary_target, encoded[:, column])) - 0.5))
                return bool(scores and max(scores) >= 0.03)
            correlation = float(np.corrcoef(encoded[:, 0], target.astype(float))[0, 1])
            return bool(abs(correlation) >= 0.05)
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _compression_fold_score(
        encoded: NDArray[np.float64],
        y: Any,
        classes: NDArray[Any],
    ) -> float:
        """Score correctly oriented class codelength evidence on one OOF fold."""
        from sklearn.metrics import roc_auc_score

        target = np.asarray(y)
        class_scores = encoded[:, : len(classes)]
        if len(classes) == 2:
            margin = class_scores[:, 1] - class_scores[:, 0]
            return float(roc_auc_score(target == classes[1], margin))
        shifted = class_scores - class_scores.max(axis=1, keepdims=True)
        probability = np.exp(shifted)
        probability /= probability.sum(axis=1, keepdims=True)
        class_index = {label: index for index, label in enumerate(classes)}
        encoded_target = np.asarray([class_index[label] for label in target], dtype=int)
        return float(
            roc_auc_score(
                encoded_target,
                probability,
                labels=np.arange(len(classes)),
                multi_class="ovo",
                average="macro",
            )
        )

    @staticmethod
    def _compression_signal(fold_scores: list[float]) -> bool:
        """Admit only broad compression signal, not a lucky single partition."""
        return bool(fold_scores and min(fold_scores) >= 0.52 and float(np.mean(fold_scores)) >= 0.56)

    def _set_feature_names(self) -> None:
        datetime_names = [
            name for column in self.datetime_cols for name in self.datetime_feat[column].feature_names(column)
        ]
        category_names = []
        for column in self.cat_cols:
            if column in self.onehot:
                category_names += [f"{column}={value}" for value in self.onehot[column]]
            else:
                category_names.append(f"{column}__freq")
                category_names += [
                    f"{column}__target={label}"
                    for label in self.target_encoding.get(column, {}).get("labels", [])
                ]
        text_names = [
            (f"{column}__token={token}" if column in self.structured_id_cols else f"{column}~{token}")
            for column in self.text_cols
            for token in self.text_feat[column].vocab
        ]
        compression_names = [
            name
            for column in self.text_cols
            if self.compression_enabled.get(column, False)
            for name in self.compression_maps[column].feature_names(column)
        ]
        self.names = (
            list(self.num_cols)
            + datetime_names
            + [f"{column}__isna" for column in self.na_cols]
            + category_names
            + text_names
            + compression_names
        )
        self.target_indices = {}
        self.datetime_indices = {}
        offset = len(self.num_cols)
        for column in self.datetime_cols:
            width = self.datetime_feat[column].n_features_out_
            self.datetime_indices[column] = np.arange(offset, offset + width)
            offset += width
        offset += len(self.na_cols)
        for column in self.cat_cols:
            if column in self.onehot:
                offset += len(self.onehot[column])
            else:
                offset += 1
                if column in self.target_encoding:
                    width = len(self.target_encoding[column]["prior"])
                    self.target_indices[column] = np.arange(offset, offset + width)
                    offset += width
        offset += len(text_names)
        self.compression_indices = {}
        for column in self.text_cols:
            if self.compression_enabled.get(column, False):
                width = self.compression_maps[column].n_features_out_
                self.compression_indices[column] = np.arange(offset, offset + width)
                offset += width

    def fit(self, X: Any, y: Any = None) -> Self:
        import pandas as pd

        frame = pd.DataFrame(X).copy()
        self.input_cols = list(frame.columns)
        self.datetime_cols = [
            column for column in frame.columns if pd.api.types.is_datetime64_any_dtype(frame[column].dtype)
        ]
        self.datetime_feat = {
            column: _DateTimeFeaturizer().fit(frame[column]) for column in self.datetime_cols
        }
        for column in frame.columns:
            if frame[column].dtype == object:
                converted = pd.to_numeric(frame[column], errors="coerce")
                if converted.notna().mean() > 0.9:
                    leading_zero_rate = frame[column].astype(str).str.match(r"0\d").mean()
                    if leading_zero_rate < 0.05:
                        frame[column] = converted
                    else:
                        warnings.warn(
                            f"column {column!r} is numeric-parseable but has leading-zero codes "
                            "(e.g. '00042'); treating it as a category, not a numeric magnitude. "
                            "Pass it pre-encoded to override.",
                            stacklevel=2,
                        )

        self.num_cols = [
            column
            for column in frame.columns
            if column not in self.datetime_cols and pd.api.types.is_numeric_dtype(frame[column])
        ]
        for column in self.num_cols:
            values = pd.to_numeric(frame[column], errors="coerce")
            frame[column] = values.where(np.isfinite(values), np.nan)
        self._target_is_classification = bool(
            y is not None
            and (self.task == "classification" or (self.task == "auto" and _is_classification(np.asarray(y))))
        )

        self.text_cols = []
        self.structured_id_cols = []
        self.text_feat = {}
        self.cat_cols = []
        for column in (
            column
            for column in frame.columns
            if column not in self.num_cols and column not in self.datetime_cols
        ):
            series = frame[column].dropna().astype(str)
            sample = series.sample(min(len(series), 1000), random_state=0) if len(series) else series
            mean_tokens = float(np.mean([len(_tokenize(value)) for value in sample])) if len(sample) else 0.0
            structured_identifier = _looks_like_structured_identifier(frame[column])
            if (mean_tokens >= 4 and frame[column].nunique(dropna=True) > 20) or structured_identifier:
                self.text_cols.append(column)
                if structured_identifier:
                    self.structured_id_cols.append(column)
                max_features = 1200 if y is not None and not self._target_is_classification else 300
                self.text_feat[column] = _TextFeaturizer(max_features=max_features).fit(
                    frame[column],
                    y,
                    is_classification=self._target_is_classification,
                )
            else:
                self.cat_cols.append(column)

        self.compression_maps = {}
        self.compression_enabled = {}
        self.compression_report = []
        if y is not None and self._target_is_classification and self.compression_evidence_enabled:
            for column in self.text_cols:
                if column in self.structured_id_cols:
                    continue
                try:
                    evidence_map = CompressionEvidenceMap().fit(frame[column].to_numpy(dtype=object), y)
                except (TypeError, ValueError, FloatingPointError, OverflowError) as error:
                    self.compression_report.append(
                        {
                            "column": str(column),
                            "selected": False,
                            "reason": f"fit_failed:{type(error).__name__}",
                        }
                    )
                    continue
                if evidence_map.is_active_:
                    self.compression_maps[column] = evidence_map
                    self.compression_enabled[column] = False
                else:
                    self.compression_report.append(
                        {
                            "column": str(column),
                            "selected": False,
                            "reason": evidence_map.inactive_reason_,
                            "reference_bytes_per_class": int(evidence_map.reference_bytes_),
                        }
                    )

        self.medians = {
            column: (float(frame[column].median()) if np.isfinite(frame[column].median()) else 0.0)
            for column in self.num_cols
        }
        self.na_cols = [column for column in self.num_cols if frame[column].isna().any()]
        self.onehot = {}
        self.freq = {}
        self.target_encoding = {}
        self.target_enabled = {}
        self._target_classes = np.unique(y) if self._target_is_classification else None
        self._target_smoothing = float(np.clip(len(frame) ** 0.25, 5.0, 20.0))
        for column in self.cat_cols:
            values = self._category_keys(frame[column])
            counts = pd.Series(values).value_counts()
            if len(counts) <= 20:
                self.onehot[column] = list(counts.index)
            else:
                self.freq[column] = {key: value / len(frame) for key, value in counts.items()}
                if y is not None and self.target_encoding_enabled:
                    self.target_encoding[column] = self._target_stats(
                        values,
                        y,
                        self._target_is_classification,
                        self._target_classes,
                        self._target_smoothing,
                        gaussian=(
                            self.gaussian_target_statistics_enabled and not self._target_is_classification
                        ),
                    )
                    self.target_enabled[column] = False
        self.onehot_idx = {
            column: {value: index for index, value in enumerate(self.onehot[column])}
            for column in self.onehot
        }
        self._set_feature_names()
        return self

    def transform(self, X: Any) -> NDArray[np.float64]:
        import pandas as pd

        if isinstance(X, pd.DataFrame):
            if X.columns.duplicated().any():
                duplicates = list(X.columns[X.columns.duplicated()])
                raise ValueError(f"predict-time DataFrame has duplicate columns: {duplicates[:5]}")
            extra = [column for column in X.columns if column not in self.input_cols]
            missing = [column for column in self.input_cols if column not in X.columns]
            if extra or missing:
                raise ValueError(
                    f"predict-time columns differ from fit: {len(extra)} unexpected {extra[:5]}, "
                    f"{len(missing)} missing {missing[:5]}"
                )
            frame = X.reindex(columns=self.input_cols).copy()
        else:
            values = np.asarray(X)
            if values.ndim != 2:
                raise ValueError(f"X must be a 2-D table; got shape {values.shape}")
            if values.shape[1] != len(self.input_cols):
                raise ValueError(
                    f"X has {values.shape[1]} columns, but the fitted model expects {len(self.input_cols)}"
                )
            frame = pd.DataFrame(values, columns=self.input_cols)

        n_rows = len(frame)
        for column in self.num_cols:
            values = pd.to_numeric(frame[column], errors="coerce")
            frame[column] = values.where(np.isfinite(values), np.nan)
        columns = [frame[column].fillna(self.medians[column]).to_numpy(float) for column in self.num_cols]
        columns += [self.datetime_feat[column].transform(frame[column]) for column in self.datetime_cols]
        columns += [frame[column].isna().to_numpy(float) for column in self.na_cols]

        small_batch = n_rows <= 64
        for column in self.cat_cols:
            if small_batch:
                raw = frame[column].to_numpy(dtype=object)
                keys = ["__nan__" if pd.isna(value) else value for value in raw]
                if column in self.onehot:
                    block = np.zeros((n_rows, len(self.onehot[column])))
                    index = self.onehot_idx[column]
                    for row, value in enumerate(keys):
                        position = index.get(value, -1)
                        if position >= 0:
                            block[row, position] = 1.0
                    columns.append(block)
                else:
                    frequency = self.freq[column]
                    columns.append(np.array([frequency.get(value, 0.0) for value in keys], dtype=float))
                    if column in self.target_encoding:
                        stats = self.target_encoding[column]
                        values = self._apply_target_stats(keys, stats)
                        if not self.target_enabled.get(column, False):
                            values[:] = stats["prior"]
                        columns.append(values)
                continue

            values = frame[column].astype("object").where(frame[column].notna(), "__nan__")
            if column in self.onehot:
                block = np.zeros((n_rows, len(self.onehot[column])))
                positions = values.map(self.onehot_idx[column]).to_numpy()
                valid = ~pd.isna(positions)
                block[np.flatnonzero(valid), positions[valid].astype(int)] = 1.0
                columns.append(block)
            else:
                columns.append(values.map(self.freq[column]).fillna(0.0).to_numpy(float))
                if column in self.target_encoding:
                    stats = self.target_encoding[column]
                    encoded = self._apply_target_stats(values.to_numpy(dtype=object), stats)
                    if not self.target_enabled.get(column, False):
                        encoded[:] = stats["prior"]
                    columns.append(encoded)

        for column in self.text_cols:
            columns.append(self.text_feat[column].transform(frame[column]))
        for column in self.text_cols:
            if self.compression_enabled.get(column, False):
                columns.append(self.compression_maps[column].transform(frame[column].to_numpy(dtype=object)))
        return np.column_stack(columns) if columns else np.zeros((n_rows, 0))

    def fit_transform(  # noqa: C901 - target/compression evidence state machine
        self,
        X: Any,
        y: Any = None,
        *,
        validation_groups: Any | None = None,
    ) -> NDArray[np.float64]:
        """Fit predict-time maps while encoding training rows out of fold."""
        import pandas as pd

        self.fit(X, y)
        if y is None:
            return self.transform(X)

        frame = pd.DataFrame(X).reindex(columns=self.input_cols)
        target = np.asarray(y)
        groups = None if validation_groups is None else np.asarray(validation_groups, dtype=np.int64)
        if groups is not None and (groups.ndim != 1 or len(groups) != len(target)):
            raise ValueError("validation_groups must have one value per preprocessing row")
        target_out_of_fold = {}
        if self.target_encoding:
            if groups is not None:
                target_splits = []
            elif self._target_is_classification:
                from sklearn.model_selection import StratifiedKFold

                _, counts = np.unique(target, return_counts=True)
                folds = min(5, int(counts.min()))
                target_splits = (
                    list(StratifiedKFold(folds, shuffle=True, random_state=0).split(frame, target))
                    if folds >= 2
                    else []
                )
            else:
                from sklearn.model_selection import KFold

                folds = min(5, len(target))
                target_splits = (
                    list(KFold(folds, shuffle=True, random_state=0).split(frame)) if folds >= 2 else []
                )

            for column, full_stats in self.target_encoding.items():
                keys = self._category_keys(frame[column])
                if groups is not None:
                    encoded = self._causal_target_values(keys, target, groups)
                    enabled = self._target_signal(encoded, target)
                else:
                    encoded = np.tile(full_stats["prior"], (len(target), 1))
                    for train, validation in target_splits:
                        fold_stats = self._target_stats(
                            keys[train],
                            target[train],
                            self._target_is_classification,
                            self._target_classes,
                            self._target_smoothing,
                            gaussian=(
                                self.gaussian_target_statistics_enabled and not self._target_is_classification
                            ),
                        )
                        encoded[validation] = self._apply_target_stats(keys[validation], fold_stats)
                    enabled = bool(target_splits) and self._target_signal(encoded, target)
                self.target_enabled[column] = enabled
                if enabled:
                    target_out_of_fold[column] = encoded

        for column in [column for column, enabled in self.target_enabled.items() if not enabled]:
            del self.target_encoding[column]
            del self.target_enabled[column]

        compression_out_of_fold = {}
        if self.compression_maps and groups is not None:
            for column, full_map in self.compression_maps.items():
                self.compression_report.append(
                    {
                        "column": str(column),
                        "selected": False,
                        "phrases": int(len(full_map.keys_)),
                        "reference_bytes_per_class": int(full_map.reference_bytes_),
                        "reason": "temporal_compression_requires_causal_replay",
                    }
                )
                self.compression_enabled[column] = False
        elif self.compression_maps:
            from sklearn.model_selection import StratifiedKFold

            _, counts = np.unique(target, return_counts=True)
            folds = min(2, int(counts.min()))
            compression_splits = (
                list(StratifiedKFold(folds, shuffle=True, random_state=0).split(frame, target))
                if folds >= 2
                else []
            )
            for column, full_map in list(self.compression_maps.items()):
                values = frame[column].to_numpy(dtype=object)
                encoded = np.zeros((len(target), full_map.n_features_out_), dtype=float)
                fold_scores = []
                failure_reason = None
                for train, validation in compression_splits:
                    try:
                        fold_map = CompressionEvidenceMap(
                            seed=full_map.seed,
                            max_features=full_map.max_features,
                            max_reference_bytes=full_map.max_reference_bytes,
                            max_document_bytes=full_map.max_document_bytes,
                        ).fit(values[train], target[train])
                        if not fold_map.is_active_:
                            failure_reason = fold_map.inactive_reason_
                            break
                        if not np.array_equal(fold_map.classes_, full_map.classes_):
                            failure_reason = "fold_class_mismatch"
                            break
                        fold_values = fold_map.transform(values[validation])
                        encoded[validation] = fold_values
                        fold_scores.append(
                            self._compression_fold_score(
                                fold_values,
                                target[validation],
                                fold_map.classes_,
                            )
                        )
                    except (TypeError, ValueError, FloatingPointError, OverflowError) as error:
                        failure_reason = f"oof_failed:{type(error).__name__}"
                        break
                selected = (
                    failure_reason is None
                    and len(fold_scores) == len(compression_splits)
                    and self._compression_signal(fold_scores)
                )
                self.compression_report.append(
                    {
                        "column": str(column),
                        "selected": bool(selected),
                        "metric": "roc_auc" if len(full_map.classes_) == 2 else "macro_ovo_auc",
                        "mean_score": float(np.mean(fold_scores)) if fold_scores else None,
                        "fold_scores": [float(score) for score in fold_scores],
                        "phrases": int(len(full_map.keys_)),
                        "reference_bytes_per_class": int(full_map.reference_bytes_),
                        "reason": failure_reason or (None if selected else "oof_signal_below_gate"),
                    }
                )
                self.compression_enabled[column] = bool(selected)
                if selected:
                    compression_out_of_fold[column] = encoded

        for column in [column for column, enabled in self.compression_enabled.items() if not enabled]:
            del self.compression_maps[column]
            del self.compression_enabled[column]
        self._set_feature_names()
        output = self.transform(X)
        for column, encoded in target_out_of_fold.items():
            output[:, self.target_indices[column]] = encoded
        for column, encoded in compression_out_of_fold.items():
            output[:, self.compression_indices[column]] = encoded
        return output


def _onehot_group_metadata(
    preprocessor: _Preprocessor | None,
) -> tuple[tuple[tuple[int, ...], ...], tuple[dict[str, Any], ...]]:
    """Return eligible one-hot blocks and their original declarations."""
    if preprocessor is None:
        return (), ()
    groups = []
    metadata = []
    datetime_width = sum(
        preprocessor.datetime_feat[column].n_features_out_
        for column in getattr(preprocessor, "datetime_cols", [])
    )
    index = len(preprocessor.num_cols) + datetime_width + len(preprocessor.na_cols)
    for column in preprocessor.cat_cols:
        if column in preprocessor.onehot:
            width = len(preprocessor.onehot[column])
            if 2 <= width <= 63:
                columns = tuple(range(index, index + width))
                groups.append(columns)
                metadata.append(
                    {
                        "name": str(column),
                        "levels": tuple(preprocessor.onehot[column]),
                        "columns": columns,
                    }
                )
            index += width
        else:
            index += 1 + len(preprocessor.target_encoding.get(column, {}).get("prior", []))
    return tuple(groups), tuple(metadata)


def _onehot_groups(preprocessor: _Preprocessor | None) -> tuple[tuple[int, ...], ...]:
    """Return encoded-column blocks representing one atomic raw category."""
    return _onehot_group_metadata(preprocessor)[0]


__all__ = ["target_encode"]
