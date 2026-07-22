"""Model registry.

Every model is exposed as a factory that returns an sklearn-compatible estimator
for a given task ("classification" | "regression"). Factories import their backend
lazily so the harness runs with whatever subset of libraries is installed — a model
whose library is missing is reported as "skipped", never a crash.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# A factory takes the task type and returns a fitted-able sklearn estimator.
Factory = Callable[[str], object]


class ModelUnavailable(ImportError):
    """Raised by a factory when its backend library is not installed."""


class _CategoricalObjectCaster(BaseEstimator, TransformerMixin):
    """Present pandas categorical columns to sklearn imputers as ordinary objects."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if hasattr(X, "astype"):
            return X.astype(object)
        return np.asarray(X, dtype=object)


class _FoldTabularEncoder(BaseEstimator, TransformerMixin):
    """Fold-fitted numeric/categorical preprocessing for sklearn estimators.

    TabPVN and TabPFN receive raw DataFrames and own their respective native
    preprocessing. Classical estimators need a numeric matrix; keeping this
    transformer inside an sklearn Pipeline prevents it from observing a task's
    validation or test rows.
    """

    def fit(self, X, y=None):
        import pandas as pd
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OneHotEncoder

        self._is_frame = isinstance(X, pd.DataFrame)
        if not self._is_frame:
            self._encoder = SimpleImputer(strategy="median", add_indicator=True).fit(
                np.asarray(X, dtype=float)
            )
            return self
        self._columns = list(X.columns)
        numeric = list(X.select_dtypes(include=np.number).columns)
        categorical = [c for c in self._columns if c not in numeric]
        transforms = []
        if numeric:
            transforms.append(("numeric", SimpleImputer(strategy="median", add_indicator=True), numeric))
        if categorical:
            transforms.append(
                (
                    "categorical",
                    make_pipeline(
                        _CategoricalObjectCaster(),
                        SimpleImputer(strategy="most_frequent"),
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                    categorical,
                )
            )
        self._encoder = ColumnTransformer(transforms, sparse_threshold=0.0).fit(X)
        return self

    def transform(self, X):
        if not self._is_frame:
            return self._encoder.transform(np.asarray(X, dtype=float))
        return self._encoder.transform(X.reindex(columns=self._columns))


def _fold_preprocessed(estimator):
    from sklearn.pipeline import make_pipeline

    return make_pipeline(_FoldTabularEncoder(), estimator)


# --- sklearn baselines (always available; the floor every real model must clear) ---


def _sklearn_baseline(task: str):
    if task == "classification":
        from sklearn.ensemble import RandomForestClassifier

        return _fold_preprocessed(RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0))
    from sklearn.ensemble import RandomForestRegressor

    return _fold_preprocessed(RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=0))


def _logreg_or_ridge(task: str):
    if task == "classification":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return _fold_preprocessed(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)))
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return _fold_preprocessed(make_pipeline(StandardScaler(), Ridge()))


# --- gradient-boosted trees (the incumbents to beat on large data) ---


def _hist_gradient_boosting(task: str):
    """A built-in, fixed-config GBDT baseline with no optional dependency."""
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    cls = HistGradientBoostingClassifier if task == "classification" else HistGradientBoostingRegressor
    return _fold_preprocessed(cls(learning_rate=0.06, max_iter=400, l2_regularization=1.0, random_state=0))


def _xgboost(task: str):
    try:
        import xgboost as xgb
    except ImportError as e:
        raise ModelUnavailable("xgboost not installed (uv sync --extra gbdt)") from e
    cls = xgb.XGBClassifier if task == "classification" else xgb.XGBRegressor
    return _fold_preprocessed(cls(n_estimators=500, tree_method="hist", n_jobs=-1, random_state=0))


def _lightgbm(task: str):
    try:
        import lightgbm as lgb
    except ImportError as e:
        raise ModelUnavailable("lightgbm not installed (uv sync --extra gbdt)") from e
    cls = lgb.LGBMClassifier if task == "classification" else lgb.LGBMRegressor
    return _fold_preprocessed(cls(n_estimators=500, n_jobs=-1, random_state=0, verbose=-1))


def _catboost(task: str):
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
    except ImportError as e:
        raise ModelUnavailable("catboost not installed (uv sync --extra gbdt)") from e
    cls = CatBoostClassifier if task == "classification" else CatBoostRegressor
    return _fold_preprocessed(cls(iterations=500, random_state=0, verbose=False, allow_writing_files=False))


# --- TabPFN (the foundation-model target) ---


def _tabpfn(task: str):
    try:
        from tabpfn import TabPFNClassifier, TabPFNRegressor
    except ImportError as e:
        raise ModelUnavailable("tabpfn not installed (uv sync --extra pfn)") from e
    cls = TabPFNClassifier if task == "classification" else TabPFNRegressor
    checkpoint = (
        Path(__file__).resolve().parents[2]
        / "external"
        / "tabpfn3"
        / f"tabpfn-v3-{'classifier' if task == 'classification' else 'regressor'}-v3_default.ckpt"
    )
    # Use the checked-in public TabPFN-3 weights when available. CPU is deliberate:
    # the local benchmark measures both foundation models under the same hardware.
    if checkpoint.is_file():
        return cls(
            model_path=checkpoint,
            device="cpu",
            ignore_pretraining_limits=True,
            random_state=0,
            show_progress_bar=False,
        )
    return cls(random_state=0, show_progress_bar=False)


def _tabpvn(task: str):
    """The packaged zero-knob TabPVN estimator, including its proof-carrying default path."""
    from tabpvn import TabPVN

    return TabPVN(seed=0, task=task)


def _tabpvn_legacy_allocation(task: str):
    """Research ablation: pre-verifier absolute top-half search allocation."""
    from tabpvn import TabPVN
    from tabpvn.candidate_allocation import AllocationDecision, FinalistDecision, score_values

    class _LegacyAllocation(TabPVN):
        def _allocate_search_budget(
            self,
            scores,
            candidates,
            baseline_index,
            *,
            keep,
            maximize,
            prune_dominated,
        ):
            del prune_dominated
            ranked = sorted(
                candidates,
                key=lambda index: score_values(scores[index]),
                reverse=maximize,
            )
            promoted = ranked[:keep]
            if baseline_index not in promoted:
                promoted.append(baseline_index)
            return AllocationDecision(
                tuple(promoted),
                {
                    "stage": "candidate_allocation",
                    "method": "legacy_absolute_top_half",
                    "absolute_anchor": "certified_baseline",
                    "evaluated_candidates": int(len(candidates)),
                    "promoted_candidates": int(len(promoted)),
                    "baseline_retained": True,
                },
            )

        def _select_search_finalist(self, scores, candidates, baseline_index, *, maximize):
            ranked = sorted(
                candidates,
                key=lambda index: score_values(scores[index]),
                reverse=maximize,
            )
            return FinalistDecision(
                ranked[0],
                {
                    "stage": "finalist_selection",
                    "method": "legacy_absolute_finalist",
                    "evaluated_candidates": int(len(candidates)),
                    "absolute_leader": int(ranked[0]),
                    "selected_candidate": int(ranked[0]),
                    "selection_changed": False,
                    "baseline_candidate": int(baseline_index),
                },
            )

    return _LegacyAllocation(seed=0, task=task)


def _tabpvn_independent_linear_gate(task: str):
    """Research ablation: refit both sides of the post-tuning linear-leaf gate."""
    from tabpvn import TabPVN

    class _IndependentLinearGate(TabPVN):
        _reuse_tuning_linear_leaf_evidence = False

    return _IndependentLinearGate(seed=0, task=task)


def _tabpvn_coarse_memory_blends(task: str):
    """Research ablation: retain the original five-point memory blend grid."""
    from tabpvn import TabPVN

    class _CoarseMemoryBlends(TabPVN):
        _memory_blend_weights = (0.1, 0.2, 0.3, 0.4, 0.5)

    return _CoarseMemoryBlends(seed=0, task=task)


def _tabpvn_strict_target_encoding(task: str):
    """Research ablation: require the legacy fixed 0.003 AUC gain."""
    from tabpvn import TabPVN

    class _StrictTargetEncoding(TabPVN):
        _target_encoding_proper_score_support = False

    return _StrictTargetEncoding(seed=0, task=task)


def _tabpvn_base(task: str):
    """Research-only ablation: the packaged default without symbolic programs."""
    from tabpvn import TabPVN

    class _NoSymbolicPrograms(TabPVN):
        def _auto_interactions(self, X, y, boost):
            return None

        def _auto_rare_interactions(self, X, y, boost):
            return None

    return _NoSymbolicPrograms(seed=0, task=task)


def _tabpvn_frequency_only(task: str):
    """Research-only ablation: legacy frequency encoding with no new proposers."""
    from tabpvn import TabPVN

    class _FrequencyOnly(TabPVN):
        _target_encoding = False

        def _auto_interactions(self, X, y, boost):
            return None

        def _auto_rare_interactions(self, X, y, boost):
            return None

    return _FrequencyOnly(seed=0, task=task)


def _tabpvn_shared_multiclass(task: str):
    """Research candidate: one proof region partition per multiclass boosting round."""
    from tabpvn import TabPVN

    class _SharedMulticlass(TabPVN):
        def _classifier(self, **kwargs):
            kwargs["shared_structure"] = True
            return super()._classifier(**kwargs)

    return _SharedMulticlass(seed=0, task=task)


def _tabpvn_best_first_multiclass(task: str):
    """Research candidate: bounded leaf-wise growth on compact numeric multiclass tables."""
    from tabpvn import TabPVN

    class _BestFirstMulticlass(TabPVN):
        def _auto_tune_clf(self, X, y):
            config = super()._auto_tune_clf(X, y)
            if np.unique(y).size > 2 and not getattr(self, "_cat_groups", ()):
                config = dict(config)
                config["depth"] = max(12, int(config.get("depth", 4)))
                config["max_leaves"] = 24
                config["best_first_pair"] = False
                config["adaptive_best_first_pair"] = False
            return config

    return _BestFirstMulticlass(seed=0, task=task)


def _tabpvn_hard_pair_best_first(task: str):
    """Research candidate: leaf-wise capacity only for the currently hardest class pair."""
    from tabpvn import TabPVN

    class _HardPairBestFirst(TabPVN):
        def _auto_tune_clf(self, X, y):
            config = super()._auto_tune_clf(X, y)
            if np.unique(y).size > 2 and not getattr(self, "_cat_groups", ()):
                config = dict(config)
                config["max_leaves"] = 24
                config["best_first_pair"] = True
                config["adaptive_best_first_pair"] = False
            return config

    return _HardPairBestFirst(seed=0, task=task)


def _tabpvn_adaptive_hard_pair(task: str):
    """Compatibility alias for the promoted adaptive hard-pair default."""
    return _tabpvn(task)


def _tabpvn_ungated_adaptive_hard_pair(task: str):
    """Research ablation: retain residual routing but skip proper-score stage selection."""
    from tabpvn import TabPVN

    class _UngatedAdaptiveHardPair(TabPVN):
        def _with_adaptive_multiclass_pair_growth(self, X, y, config):
            selected = super()._with_adaptive_multiclass_pair_growth(X, y, config)
            if selected.get("adaptive_best_first_pair", False):
                selected["verifier_gated_pair_growth"] = False
            return selected

    return _UngatedAdaptiveHardPair(seed=0, task=task)


def _tabpvn_no_adaptive_hard_pair(task: str):
    """Research ablation: disable verifier-triggered multiclass pair capacity."""
    from tabpvn import TabPVN

    class _NoAdaptiveHardPair(TabPVN):
        def _with_adaptive_multiclass_pair_growth(self, X, y, config):
            del X, y
            return dict(config)

    return _NoAdaptiveHardPair(seed=0, task=task)


def _tabpvn_unstratified_multiclass(task: str):
    """Research ablation: restore the pre-fix final multiclass verifier split."""
    from tabpvn import TabPVN

    class _UnstratifiedMulticlass(TabPVN):
        _multiclass_stratified_verifier = False

    return _UnstratifiedMulticlass(seed=0, task=task)


def _tabpvn_depth4_affine_auc(task: str):
    """Research isolation: depth-four affine regions with AUC checkpointing."""
    from tabpvn import TabPVN

    class _DepthFourAffineAUC(TabPVN):
        def _auto_joint_rank_regions(self, X, y, boost):
            challenger = dict(boost)
            challenger.update(lr=0.05, depth=4, linear_leaf=True, validation_metric="auc")
            return challenger

    return _DepthFourAffineAUC(seed=0, task=task)


def _tabpvn_threshold_rules(task: str):
    """Compatibility alias for the promoted bounded threshold-clause default."""
    from tabpvn import TabPVN

    class _ThresholdRules(TabPVN):
        _threshold_predicates = True

    return _ThresholdRules(seed=0, task=task)


def _tabpvn_no_threshold_rules(task: str):
    """Research ablation: disable only bounded numeric threshold clauses."""
    from tabpvn import TabPVN

    class _NoThresholdRules(TabPVN):
        _threshold_predicates = False

    return _NoThresholdRules(seed=0, task=task)


def _tabpvn_no_mdl_symbolic_beam(task: str):
    """Research ablation: retain finite predicates but disable MDL beam expansion."""
    from tabpvn import TabPVN

    class _NoMDLSymbolicBeam(TabPVN):
        _symbolic_mdl_beam = False

    return _NoMDLSymbolicBeam(seed=0, task=task)


def _tabpvn_no_mdl_dnf(task: str):
    """Research ablation: retain signed conjunction search but disable DNF composition."""
    from tabpvn import TabPVN

    class _NoMDLDNF(TabPVN):
        _symbolic_mdl_dnf = False

    return _NoMDLDNF(seed=0, task=task)


def _tabpvn_no_mdl_recursive_dnf(task: str):
    """Research ablation: retain pair DNF but disable the third recursive branch."""
    from tabpvn import TabPVN

    class _NoMDLRecursiveDNF(TabPVN):
        _symbolic_mdl_recursive_dnf = False

    return _NoMDLRecursiveDNF(seed=0, task=task)


def _tabpvn_no_mdl_exception(task: str):
    """Research ablation: retain recursive DNF but disable set-difference programs."""
    from tabpvn import TabPVN

    class _NoMDLException(TabPVN):
        _symbolic_mdl_exception = False

    return _NoMDLException(seed=0, task=task)


def _tabpvn_no_bayesian_expert_router(task: str):
    """Research ablation: retain memory experts but apply one global blend."""
    from tabpvn import TabPVN

    class _NoBayesianExpertRouter(TabPVN):
        _bayesian_expert_routing = False

    return _NoBayesianExpertRouter(seed=0, task=task)


def _tabpvn_no_hierarchical_path_memory(task: str):
    """Research ablation: retain the incumbent local proof-path vote only."""
    from tabpvn import TabPVN

    class _NoHierarchicalPathMemory(TabPVN):
        _hierarchical_proof_path_memory = False

    return _NoHierarchicalPathMemory(seed=0, task=task)


def _tabpvn_no_temporal_context_state(task: str):
    """Research ablation: retain Laplace histories but disable exact depth-two context."""
    from tabpvn import TabPVN

    class _NoTemporalContextState(TabPVN):
        _temporal_context_state = False

    return _NoTemporalContextState(seed=0, task=task)


def _tabpvn_no_temporal_suffix_tree(task: str):
    """Research ablation: retain fixed context but disable variable-order suffix state."""
    from tabpvn import TabPVN

    class _NoTemporalSuffixTree(TabPVN):
        _temporal_context_tree = False

    return _NoTemporalSuffixTree(seed=0, task=task)


def _tabpvn_no_categorical_hypergraph(task: str):
    """Research ablation: retain single/pair posteriors but disable categorical hyperedges."""
    from tabpvn import TabPVN

    class _NoCategoricalHypergraph(TabPVN):
        _categorical_hypergraph_posterior = False

    return _NoCategoricalHypergraph(seed=0, task=task)


def _tabpvn_no_rare_architecture(task: str):
    """Research ablation: disable AP checkpointing and rare symbolic programs."""
    from tabpvn import TabPVN

    class _NoRareArchitecture(TabPVN):
        def _auto_rare_interactions(self, X, y, boost):
            return None

    return _NoRareArchitecture(seed=0, task=task)


def _tabpvn_no_affine_rank(task: str):
    """Research ablation: disable only the OOF-gated global affine rank read."""
    from tabpvn import TabPVN

    class _NoAffineRank(TabPVN):
        _affine_rank_evidence = False

    return _NoAffineRank(seed=0, task=task)


REGISTRY: dict[str, Factory] = {
    "rf": _sklearn_baseline,
    "linear": _logreg_or_ridge,
    "hgb": _hist_gradient_boosting,
    "xgboost": _xgboost,
    "lightgbm": _lightgbm,
    "catboost": _catboost,
    "tabpfn": _tabpfn,
    "tabpvn": _tabpvn,
    "tabpvn_legacy_allocation": _tabpvn_legacy_allocation,
    "tabpvn_independent_linear_gate": _tabpvn_independent_linear_gate,
    "tabpvn_coarse_memory_blends": _tabpvn_coarse_memory_blends,
    "tabpvn_strict_target_encoding": _tabpvn_strict_target_encoding,
    "tabpvn_base": _tabpvn_base,
    "tabpvn_freq": _tabpvn_frequency_only,
    "tabpvn_shared": _tabpvn_shared_multiclass,
    "tabpvn_best_first_multiclass": _tabpvn_best_first_multiclass,
    "tabpvn_hard_pair_best_first": _tabpvn_hard_pair_best_first,
    "tabpvn_adaptive_hard_pair": _tabpvn_adaptive_hard_pair,
    "tabpvn_ungated_adaptive_hard_pair": _tabpvn_ungated_adaptive_hard_pair,
    "tabpvn_no_adaptive_hard_pair": _tabpvn_no_adaptive_hard_pair,
    "tabpvn_unstratified_multiclass": _tabpvn_unstratified_multiclass,
    "tabpvn_depth4_affine_auc": _tabpvn_depth4_affine_auc,
    "tabpvn_threshold_rules": _tabpvn_threshold_rules,
    "tabpvn_no_threshold_rules": _tabpvn_no_threshold_rules,
    "tabpvn_no_mdl_symbolic_beam": _tabpvn_no_mdl_symbolic_beam,
    "tabpvn_no_mdl_dnf": _tabpvn_no_mdl_dnf,
    "tabpvn_no_mdl_recursive_dnf": _tabpvn_no_mdl_recursive_dnf,
    "tabpvn_no_mdl_exception": _tabpvn_no_mdl_exception,
    "tabpvn_no_bayesian_expert_router": _tabpvn_no_bayesian_expert_router,
    "tabpvn_no_hierarchical_path_memory": _tabpvn_no_hierarchical_path_memory,
    "tabpvn_no_temporal_context_state": _tabpvn_no_temporal_context_state,
    "tabpvn_no_temporal_suffix_tree": _tabpvn_no_temporal_suffix_tree,
    "tabpvn_no_categorical_hypergraph": _tabpvn_no_categorical_hypergraph,
    "tabpvn_no_rare_architecture": _tabpvn_no_rare_architecture,
    "tabpvn_no_affine_rank": _tabpvn_no_affine_rank,
}


def build(name: str, task: str):
    """Instantiate a model by registry name; raises ModelUnavailable if missing."""
    if name not in REGISTRY:
        raise KeyError(f"unknown model '{name}'. known: {sorted(REGISTRY)}")
    return REGISTRY[name](task)
