"""Dataset suites.

Two loaders:

* ``sklearn_suite`` — small, network-free datasets bundled with scikit-learn. Used
  for smoke-testing the harness and for fast local iteration. NOT a serious
  benchmark — the datasets are tiny and well-worn.

* ``openml_suite`` — loads datasets by OpenML id. This is the seam where the
  official **TabArena** suite will be wired in once its exact task list / splitting
  protocol is confirmed. The ids below are a small placeholder spanning
  classification and regression, not the TabArena set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Dataset:
    name: str
    X: Any
    y: np.ndarray
    task: str  # "classification" | "regression"
    # None means the runner owns the repeated random splits. TabArena provides
    # official OpenML task folds, which must be reused verbatim for a fair
    # comparison to published results.
    splits: tuple[tuple[np.ndarray, np.ndarray], ...] | None = None


def sklearn_suite() -> list[Dataset]:
    from sklearn.datasets import (
        load_breast_cancer,
        load_diabetes,
        load_digits,
        load_iris,
        load_wine,
    )

    out: list[Dataset] = []
    for name, loader, task in [
        ("iris", load_iris, "classification"),
        ("wine", load_wine, "classification"),
        ("breast_cancer", load_breast_cancer, "classification"),
        ("digits", load_digits, "classification"),
        ("diabetes", load_diabetes, "regression"),
    ]:
        bunch = loader()
        out.append(Dataset(name, bunch.data, bunch.target, task))
    return out


# Placeholder OpenML ids — REPLACE with the official TabArena suite before drawing
# any conclusions. Format: (openml_dataset_id, task).
_OPENML_PLACEHOLDER = [
    (31, "classification"),  # credit-g
    (1590, "classification"),  # adult
]


def _encode_target(y, task: str) -> np.ndarray:
    """Create a stable numeric target without touching the feature table."""
    import pandas as pd

    if task == "classification":
        return np.asarray(pd.Categorical(y).codes)
    y = np.asarray(y, dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("OpenML regression target contains missing or non-finite values")
    return y


def _load_openml_dataset(dataset_id: int, task: str, name: str | None = None) -> Dataset:
    """Load raw OpenML features.

    Feature transforms belong inside each estimator's fit, so categorical
    encoding and imputation never inspect validation/test rows.
    """
    import openml

    ds = openml.datasets.get_dataset(dataset_id)
    X, y, _, _ = ds.get_data(target=ds.default_target_attribute)
    return Dataset(name or f"openml-{dataset_id}", X, _encode_target(y, task), task)


def openml_suite(ids: list[tuple[int, str]] | None = None) -> list[Dataset]:
    try:
        import openml  # noqa: F401
    except ImportError as e:
        raise ImportError("openml not installed (uv sync --extra openml)") from e

    ids = ids or _OPENML_PLACEHOLDER
    return [_load_openml_dataset(did, task) for did, task in ids]


_TABARENA_META = "benchmark/data/tabarena_metadata.csv"

# Size buckets matching the TabPFN-3 report's framing (the report's "largest data
# subset" = 10k–100k samples, where TabPFN-3-Plus claims +420 Elo).
_SIZE_BUCKETS = {
    "le10k": lambda n: n <= 10_000,
    "10k-100k": lambda n: 10_000 < n <= 100_000,
    "gt100k": lambda n: n > 100_000,
    "all": lambda n: True,
}


def tabarena_suite(
    size: str = "all",
    problem_types: list[str] | None = None,
    runnable: str | None = None,
    max_datasets: int | None = None,
    dataset_names: list[str] | None = None,
) -> list[Dataset]:
    """Load TabArena datasets (suite 'tabarena-v0.1') via OpenML by dataset id.

    Filters:
      size:         one of _SIZE_BUCKETS ('all' | 'le10k' | '10k-100k' | 'gt100k').
                    '10k-100k' == the TabPFN-3 report's headline "largest" subset.
      problem_types: subset of {'binary','multiclass','regression'} (None = all).
      runnable:      None | 'tabpfnv2' | 'tabicl' — keep only datasets flagged
                     runnable by that model in the TabArena metadata.
      max_datasets:  cap for quick smoke runs.
      dataset_names: optional ordered task names for a fixed, reproducible
                     promotion slice. Names are resolved after the other filters.

    The returned splits are the official OpenML task folds. The runner reuses them
    exactly, rather than generating fresh train/test splits. ``max_datasets`` and
    its CLI split cap are only for smoke runs; omit the cap for reportable results.
    """
    try:
        import openml
    except ImportError as e:
        raise ImportError("openml not installed (uv sync --extra openml)") from e
    import pandas as pd

    meta = pd.read_csv(_TABARENA_META)
    if size not in _SIZE_BUCKETS:
        raise ValueError(f"size must be one of {sorted(_SIZE_BUCKETS)}")
    meta = meta[meta.num_instances.apply(_SIZE_BUCKETS[size])]
    if problem_types is not None:
        meta = meta[meta.problem_type.isin(problem_types)]
    if runnable == "tabpfnv2":
        meta = meta[meta.can_run_tabpfnv2]
    elif runnable == "tabicl":
        meta = meta[meta.can_run_tabicl]
    if dataset_names is not None:
        names = list(dataset_names)
        available = set(meta.dataset_name)
        missing = [name for name in names if name not in available]
        if missing:
            raise ValueError(f"TabArena datasets unavailable after filters: {missing}")
        meta = meta.set_index("dataset_name").loc[names].reset_index()
    if max_datasets is not None:
        meta = meta.head(max_datasets)

    out: list[Dataset] = []
    for row in meta.itertuples():
        task = "regression" if row.problem_type == "regression" else "classification"
        task_obj = openml.tasks.get_task(int(row.task_id), download_splits=True)
        ds = task_obj.get_dataset()
        X, y, _, _ = ds.get_data(target=task_obj.target_name)
        official_splits = []
        for repeat in range(int(row.tabarena_num_repeats)):
            for fold in range(int(row.num_folds)):
                train, test = task_obj.get_train_test_split_indices(repeat=repeat, fold=fold, sample=0)
                official_splits.append((np.asarray(train, dtype=int), np.asarray(test, dtype=int)))
        out.append(
            Dataset(
                row.dataset_name,
                X,
                _encode_target(y, task),
                task,
                tuple(official_splits),
            )
        )
    return out


@dataclass
class KaggleSpec:
    """One Kaggle validation dataset.

    ref:    "owner/dataset-slug" (a Kaggle *dataset*) — or a competition slug if
            is_competition=True.
    csv:    filename to read from the downloaded archive.
    target: target column name.
    task:   "classification" | "regression".

    LEAKAGE DISCIPLINE: prefer datasets/competitions released AFTER the training
    cutoff of every model under test. Popular older Kaggle sets are likely in the
    pretraining corpus of real-data foundation models (TabDPT, Mitra, TabPFN
    variants) — validating on them is contaminated. Kaggle is for fresh, messy,
    and/or large holdouts; TabArena remains the headline benchmark.
    """

    ref: str
    csv: str
    target: str
    task: str
    is_competition: bool = False
    drop: tuple[str, ...] = ()  # columns to drop before encoding: free-text/IDs/dates that would
    #                             explode one-hot, and post-outcome columns that would leak the target.


_KAGGLE_CACHE = "data/kaggle"


def kaggle_suite(specs: list[KaggleSpec]) -> list[Dataset]:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as e:
        raise ImportError("kaggle not installed (uv sync --extra kaggle)") from e

    import os

    import pandas as pd

    api = KaggleApi()
    api.authenticate()  # needs ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY

    out: list[Dataset] = []
    for spec in specs:
        dest = os.path.join(_KAGGLE_CACHE, spec.ref.replace("/", "__"))
        os.makedirs(dest, exist_ok=True)
        if spec.is_competition:
            api.competition_download_files(spec.ref, path=dest, quiet=True)
        else:
            api.dataset_download_files(spec.ref, path=dest, quiet=True, unzip=True)
        # unzip=False for competitions leaves a .zip; pandas reads it inside the dir
        csv_path = os.path.join(dest, spec.csv)
        df = pd.read_csv(csv_path)

        if spec.drop:  # remove free-text/ID/date columns (one-hot blowup) and any leaky post-outcome columns
            df = df.drop(columns=[c for c in spec.drop if c in df.columns])
        y_raw = df.pop(spec.target)
        # Minimal, honest preprocessing (match TabArena protocol before reporting).
        X = pd.get_dummies(df, dummy_na=True).to_numpy(dtype=float)
        if spec.task == "classification":
            y = np.asarray(pd.Categorical(y_raw).codes)
        else:
            y = y_raw.to_numpy(dtype=float)
            m = ~np.isnan(
                y
            )  # a NaN regression target is unusable (e.g. unresolved tickets) — drop those rows
            X, y = X[m], y[m]
        name = f"kaggle-{spec.ref.split('/')[-1]}"
        out.append(Dataset(name, X, y, spec.task))
    return out
