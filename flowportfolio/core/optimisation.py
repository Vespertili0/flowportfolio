from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from skfolio.optimization import BaseOptimization
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

_RISK_DENOM_EPS: float = 1e-12
_RATIO_RISK_FLOOR: float = 1e-4
_RATIO_RISK_ATTRS: dict[str, str] = {
    "CVaR Ratio": "cvar",
    "CDaR Ratio": "cdar",
    "EVaR Ratio": "evar",
    "Sharpe Ratio": "standard_deviation",
    "Sortino Ratio": "semi_deviation",
    "Calmar Ratio": "max_drawdown",
}


def _copy_weights(weights: Any) -> Any:
    """Return a defensive copy of dict or ndarray weights."""
    if isinstance(weights, dict):
        return dict(weights)
    if isinstance(weights, np.ndarray):
        return weights.copy()
    return weights


def _copy_fitted_attrs(src: object, dst: object) -> None:
    """Copy scikit-learn fitted attributes (trailing underscore) from ``src`` to ``dst``."""
    for key, value in src.__dict__.items():
        if key.endswith("_") and not key.startswith("__"):
            setattr(dst, key, value)


def _normalize_previous_weights(previous_weights: Any, X: Any) -> Any:
    """Align 1D array-like previous weights to DataFrame column names when possible."""
    if previous_weights is None:
        return None
    if isinstance(previous_weights, dict):
        return dict(previous_weights)
    if isinstance(X, pd.DataFrame):
        try:
            arr = np.asarray(previous_weights, dtype=float)
        except (TypeError, ValueError):
            return _copy_weights(previous_weights)
        if arr.ndim == 1 and len(arr) == len(X.columns):
            return {str(col): float(w) for col, w in zip(X.columns, arr)}
    return _copy_weights(previous_weights)


class _SafeScorer:
    """Picklable scorer wrapper sanitizing degenerate zero-risk ratio folds."""

    def __init__(self, scoring: Callable[..., Any] | None) -> None:
        self.scoring = scoring

    @staticmethod
    def _read_finite_float(
        obj: Any, attr: str, *, default: float | None = 0.0
    ) -> float | None:
        raw = getattr(obj, attr, None)
        if raw is None:
            return default
        try:
            val = float(raw)
        except (TypeError, ValueError):
            return default
        return val if math.isfinite(val) else default

    @classmethod
    def _risk_is_degenerate(cls, pred: Any, risk_attr: str) -> bool:
        if getattr(pred, risk_attr, None) is None:
            return False
        risk = cls._read_finite_float(pred, risk_attr, default=None)
        return risk is None or risk <= _RISK_DENOM_EPS

    @classmethod
    def _degenerate_fallback(cls, pred: Any) -> float:
        """Score zero-/negative-risk folds in dimensionless ratio units."""
        m = cls._read_finite_float(pred, "mean", default=None)
        if m is None:
            return 0.0
        rfr = cls._read_finite_float(pred, "risk_free_rate", default=0.0) or 0.0
        std = cls._read_finite_float(pred, "standard_deviation", default=0.0) or 0.0
        return (m - rfr) / max(max(std, 0.0), _RATIO_RISK_FLOOR)

    def _eval_raw(self, estimator: Any, X: Any, y: Any) -> float:
        if self.scoring is None:
            return (
                float(estimator.score(X)) if y is None else float(estimator.score(X, y))
            )
        return (
            float(self.scoring(estimator, X))
            if y is None
            else float(self.scoring(estimator, X, y))
        )

    def _eval_ratio_with_fallback(
        self, estimator: Any, X: Any, y: Any, risk_attr: str
    ) -> float:
        pred = estimator.predict(X) if hasattr(estimator, "predict") else None
        if pred is not None and self._risk_is_degenerate(pred, risk_attr):
            return self._degenerate_fallback(pred)
        raw = self._eval_raw(estimator, X, y)
        if math.isfinite(raw):
            return raw
        return self._degenerate_fallback(pred) if pred is not None else 0.0

    def __call__(self, estimator: Any, X: Any, y: Any = None) -> float:
        with np.errstate(divide="ignore", invalid="ignore"):
            if self.scoring is None:
                return self._eval_ratio_with_fallback(
                    estimator, X, y, "standard_deviation"
                )

            score_func = getattr(self.scoring, "_score_func", None)
            score_name = getattr(score_func, "__name__", "")
            risk_attr = _RATIO_RISK_ATTRS.get(score_name)
            if risk_attr is not None and hasattr(estimator, "predict"):
                return self._eval_ratio_with_fallback(estimator, X, y, risk_attr)

            raw = self._eval_raw(estimator, X, y)
            return raw if math.isfinite(raw) else 0.0


def _make_safe_scorer(scoring: Any) -> Any:
    """Wrap callable or default scoring to sanitize non-finite / zero-risk ratio folds."""
    if scoring is None or callable(scoring):
        return _SafeScorer(scoring)
    return scoring


def set_previous_weights_chain(
    opt: Any, weights: Any, _visited: set[int] | None = None
) -> None:
    """Propagate previous_weights across a Pipeline, optimizer, and its entire fallback chain."""
    if opt is None:
        return
    visited = _visited if _visited is not None else set()
    if id(opt) in visited:
        return
    visited.add(id(opt))

    if isinstance(opt, Pipeline):
        for _, step in opt.steps:
            set_previous_weights_chain(step, weights, visited)
        return

    if hasattr(opt, "previous_weights"):
        opt.previous_weights = _copy_weights(weights)

    if isinstance(opt, BaseEstimator):
        est = getattr(opt, "estimator", None)
        if isinstance(est, BaseEstimator):
            set_previous_weights_chain(est, weights, visited)
        best = getattr(opt, "best_estimator_", None)
        if isinstance(best, BaseEstimator):
            set_previous_weights_chain(best, weights, visited)

    fallback = getattr(opt, "fallback", None)
    if fallback is not None and (
        isinstance(fallback, BaseEstimator) or hasattr(fallback, "previous_weights")
    ):
        set_previous_weights_chain(fallback, weights, visited)


def _has_max_turnover_chain(opt: Any, _visited: set[int] | None = None) -> bool:
    """Return True if ``opt``, its Pipeline steps, or any fallback configures ``max_turnover``."""
    if opt is None:
        return False
    visited = _visited if _visited is not None else set()
    if id(opt) in visited:
        return False
    visited.add(id(opt))

    if isinstance(opt, Pipeline):
        return any(
            _has_max_turnover_chain(step, _visited=visited) for _, step in opt.steps
        )
    if getattr(opt, "max_turnover", None) is not None:
        return True
    fallback = getattr(opt, "fallback", None)
    return (
        _has_max_turnover_chain(fallback, _visited=visited)
        if fallback is not None
        else False
    )


def _is_positive_tc_value(value: Any) -> bool:
    """Return True if a transaction_costs candidate value contains any strictly positive cost."""
    if value is None or isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        f_val = float(value)
        return math.isfinite(f_val) and f_val > 0.0
    if isinstance(value, dict):
        return any(_is_positive_tc_value(v) for v in value.values())
    try:
        arr = np.asarray(value, dtype=float)
        return bool(np.any(np.isfinite(arr) & (arr > 0.0)))
    except (TypeError, ValueError):
        return False


def _iter_param_grid_items(param_grid: Any) -> list[tuple[str, list[Any]]]:
    """Flatten a dict or sequence-of-dicts param_grid into ``(key, candidates)`` pairs."""
    if param_grid is None:
        return []
    grids = (
        [param_grid]
        if isinstance(param_grid, dict)
        else list(param_grid)
        if isinstance(param_grid, (list, tuple))
        else []
    )
    items: list[tuple[str, list[Any]]] = []
    for grid in grids:
        if not isinstance(grid, dict):
            continue
        for key, values in grid.items():
            if not isinstance(key, str):
                continue
            if isinstance(values, (list, tuple, set, np.ndarray)):
                cand_list = list(values)
            elif isinstance(values, (dict, str, bytes)):
                cand_list = [values]
            else:
                try:
                    cand_list = list(values)
                except TypeError:
                    cand_list = [values]
            items.append((key, cand_list))
    return items


def _param_grid_sets_param(
    param_grid: Any, param_name: str, *, require_non_none: bool = True
) -> bool:
    """Return True if ``param_grid`` sweeps ``param_name`` (top-level or Pipeline step)."""
    suffix = f"__{param_name}"
    for key, values in _iter_param_grid_items(param_grid):
        if key == param_name or key.endswith(suffix):
            if not require_non_none:
                return True
            if any(v is not None for v in values):
                return True
    return False


def _param_grid_sets_max_turnover(param_grid: Any) -> bool:
    """Return True if ``param_grid`` sweeps any non-None ``max_turnover`` candidate."""
    return _param_grid_sets_param(param_grid, "max_turnover", require_non_none=True)


def _param_grid_sets_positive_tc(param_grid: Any) -> bool:
    """Return True if ``param_grid`` sweeps any strictly positive ``transaction_costs`` candidate."""
    for key, values in _iter_param_grid_items(param_grid):
        if (key == "transaction_costs" or key.endswith("__transaction_costs")) and any(
            _is_positive_tc_value(v) for v in values
        ):
            return True
    return False


def _collect_param_grid_neutral_params(param_grid: Any) -> dict[str, Any]:
    """Collect estimator-relative Fold-0 neutral overrides for friction params swept in ``param_grid``."""
    neutral: dict[str, Any] = {}
    for key, values in _iter_param_grid_items(param_grid):
        if (key == "max_turnover" or key.endswith("__max_turnover")) and any(
            v is not None for v in values
        ):
            neutral[key] = None
        elif (
            key == "transaction_costs" or key.endswith("__transaction_costs")
        ) and any(_is_positive_tc_value(v) for v in values):
            neutral[key] = 0.0
    return neutral


def _neutral_equal_previous_weights(X: Any) -> Any:
    """Build equal-weight baseline matching ``X`` columns to keep constrained inner CV feasible."""
    if isinstance(X, pd.DataFrame) and X.shape[1] > 0:
        w = 1.0 / float(X.shape[1])
        return {str(col): w for col in X.columns}
    if hasattr(X, "shape") and len(X.shape) > 1 and X.shape[1] > 0:
        w = 1.0 / float(X.shape[1])
        return np.full(X.shape[1], w, dtype=float)
    return None


class SkfolioGridSearchCV(BaseOptimization, GridSearchCV):
    """A wrapper for GridSearchCV that conforms to skfolio's BaseOptimization API.

    This ensures that when skfolio's cross_val_predict uses this search object,
    it exposes weights_ and predict correctly as an optimization estimator,
    isolates inner CV folds from future outer-terminal ``previous_weights`` when
    refitting, supports Fold-0 entry neutralisation via ``refit_param_overrides``
    even when ``param_grid`` sweeps friction parameters, and sanitizes
    degenerate zero-risk ratio fold scores.
    """

    def __init__(
        self,
        estimator: Any = None,
        param_grid: Any = None,
        *,
        scoring: Any = None,
        n_jobs: int | None = None,
        refit: bool | str | Callable[..., Any] = True,
        cv: Any = None,
        verbose: int = 0,
        pre_dispatch: int | str = "2*n_jobs",
        error_score: float | str = np.nan,
        return_train_score: bool = False,
        previous_weights: Any = None,
        portfolio_params: dict[str, Any] | None = None,
        isolate_inner_previous_weights: bool = True,
        refit_param_overrides: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        GridSearchCV.__init__(
            self,
            estimator=estimator,
            param_grid=param_grid,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
            **kwargs,
        )
        BaseOptimization.__init__(self, portfolio_params=portfolio_params)
        self.portfolio_params: dict[str, Any] | None = portfolio_params
        self.previous_weights = previous_weights
        self.isolate_inner_previous_weights = isolate_inner_previous_weights
        self.refit_param_overrides = refit_param_overrides

    def _get_target_optimizer(self, est: Any) -> Any:
        """Return the terminal optimiser of a Pipeline, or ``est`` itself."""
        if isinstance(est, Pipeline) and est.steps:
            return est.steps[-1][1]
        return est

    @property
    def needs_previous_weights(self) -> bool:
        """Whether the wrapped optimiser or param_grid requires sequential previous weights."""
        if _param_grid_sets_max_turnover(
            self.param_grid
        ) or _param_grid_sets_positive_tc(self.param_grid):
            return True
        current = self._get_target_optimizer(self.estimator)
        visited: set[int] = set()
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            if bool(getattr(current, "needs_previous_weights", False)):
                return True
            current = getattr(current, "fallback", None)
        return bool(
            self.portfolio_params is not None
            and self.portfolio_params.get("weight_drift", False)
        )

    @staticmethod
    def _as_optional_float(value: Any) -> float | None:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            f_val = float(value)
            if math.isfinite(f_val):
                return f_val
        return None

    @property
    def risk_free_rate(self) -> float:
        """Per-period risk-free rate inherited from fitted or wrapped optimiser.

        Follows ``skfolio`` precedence: explicit ``portfolio_params['risk_free_rate']``
        overrides estimator-level ``risk_free_rate`` attributes.
        """
        candidates: list[Any] = [getattr(self, "_fitted_risk_free_rate_", None)]
        if isinstance(self.portfolio_params, dict):
            candidates.append(self.portfolio_params.get("risk_free_rate"))
        inner = self._get_target_optimizer(self.estimator)
        if inner is not None:
            inner_pp = getattr(inner, "portfolio_params", None)
            if isinstance(inner_pp, dict):
                candidates.append(inner_pp.get("risk_free_rate"))
            candidates.append(getattr(inner, "risk_free_rate", None))
        for candidate in candidates:
            resolved = self._as_optional_float(candidate)
            if resolved is not None:
                return resolved
        return 0.0

    @risk_free_rate.setter
    def risk_free_rate(self, value: float) -> None:
        self._fitted_risk_free_rate_ = float(value)

    @staticmethod
    def _expand_weights_to_input(
        X: Any, best_estimator: Any, inner: Any, raw_weights: Any
    ) -> tuple[Any, Any]:
        """Expand reduced pre-selection weights back to the full input feature space of X."""
        if raw_weights is None:
            return None, getattr(inner, "feature_names_in_", None)

        w_arr = np.asarray(raw_weights, dtype=float)
        n_in = X.shape[1] if hasattr(X, "shape") and len(X.shape) > 1 else None
        full_names = (
            np.asarray(X.columns, dtype=object)
            if hasattr(X, "columns")
            else getattr(best_estimator, "feature_names_in_", None)
        )

        if n_in is None or w_arr.shape[-1] == n_in:
            return w_arr, (
                full_names
                if full_names is not None
                else getattr(inner, "feature_names_in_", None)
            )

        inner_names = getattr(inner, "feature_names_in_", None)
        if full_names is not None and inner_names is not None:
            name_to_idx = {str(col): idx for idx, col in enumerate(full_names)}
            out_shape = (*w_arr.shape[:-1], n_in)
            expanded = np.zeros(out_shape, dtype=float)
            for sub_idx, name in enumerate(inner_names):
                target_idx = name_to_idx.get(str(name))
                if target_idx is not None:
                    expanded[..., target_idx] = w_arr[..., sub_idx]
            return expanded, full_names

        if isinstance(best_estimator, Pipeline):
            mask = np.arange(n_in)
            for _, step in best_estimator.steps[:-1]:
                if hasattr(step, "get_support"):
                    support = np.asarray(step.get_support(), dtype=bool)
                    if len(support) == len(mask):
                        mask = mask[support]
            if len(mask) == w_arr.shape[-1]:
                out_shape = (*w_arr.shape[:-1], n_in)
                expanded = np.zeros(out_shape, dtype=float)
                expanded[..., mask] = w_arr
                return expanded, full_names

        return w_arr, getattr(inner, "feature_names_in_", None)

    def fit(self, X: Any, y: Any = None, **fit_params: Any) -> SkfolioGridSearchCV:
        for stale_attr in (
            "_fitted_risk_free_rate_",
            "_resolved_portfolio_params_",
            "fallback_chain_",
            "error_",
            "weights_",
        ):
            if hasattr(self, stale_attr):
                delattr(self, stale_attr)

        est_clone = clone(self.estimator) if self.estimator is not None else None
        effective_pw = _normalize_previous_weights(self.previous_weights, X)
        safe_scorer = _make_safe_scorer(self.scoring)
        has_refit_overrides = (
            isinstance(self.refit_param_overrides, dict)
            and len(self.refit_param_overrides) > 0
        )

        if self.isolate_inner_previous_weights and est_clone is not None:
            # Inner CV evaluates historical sub-windows using a causal neutral
            # baseline so drifted terminal previous_weights (t_1) never leak into
            # earlier inner folds; effective_pw is restored only on full-window refit.
            needs_feasible_baseline = _has_max_turnover_chain(
                est_clone
            ) or _param_grid_sets_max_turnover(self.param_grid)
            inner_pw = (
                _neutral_equal_previous_weights(X) if needs_feasible_baseline else None
            )
            set_previous_weights_chain(est_clone, inner_pw)
            run_delegate_refit = self.refit if isinstance(self.refit, str) else False
        else:
            if effective_pw is not None and est_clone is not None:
                set_previous_weights_chain(est_clone, effective_pw)
            run_delegate_refit = (
                (self.refit if isinstance(self.refit, str) else False)
                if has_refit_overrides
                else self.refit
            )

        delegate = GridSearchCV(
            estimator=est_clone,
            param_grid=self.param_grid,
            scoring=safe_scorer,
            n_jobs=self.n_jobs,
            refit=run_delegate_refit,
            cv=self.cv,
            verbose=self.verbose,
            pre_dispatch=self.pre_dispatch,
            error_score=self.error_score,
            return_train_score=self.return_train_score,
        )
        delegate.fit(X, y, **fit_params)
        _copy_fitted_attrs(delegate, self)

        if (
            self.isolate_inner_previous_weights or has_refit_overrides
        ) and self.estimator is not None:
            should_refit = False
            if self.refit is True or callable(self.refit):
                best_idx = (
                    int(self.refit(delegate.cv_results_))
                    if callable(self.refit)
                    else int(np.argmin(delegate.cv_results_["rank_test_score"]))
                )
                self.best_index_ = best_idx
                self.best_params_ = dict(delegate.cv_results_["params"][best_idx])
                if "mean_test_score" in delegate.cv_results_:
                    self.best_score_ = float(
                        delegate.cv_results_["mean_test_score"][best_idx]
                    )
                should_refit = True
            elif isinstance(self.refit, str) and hasattr(self, "best_params_"):
                should_refit = True

            if should_refit:
                refit_est = clone(self.estimator).set_params(**self.best_params_)
                if effective_pw is not None:
                    set_previous_weights_chain(refit_est, effective_pw)
                if has_refit_overrides and self.refit_param_overrides is not None:
                    refit_est.set_params(**self.refit_param_overrides)
                refit_start = time.perf_counter()
                refit_est.fit(X, y, **fit_params)
                self.refit_time_ = time.perf_counter() - refit_start
                self.best_estimator_ = refit_est

        if not hasattr(self, "best_estimator_"):
            return self

        inner = self._get_target_optimizer(self.best_estimator_)
        raw_weights = getattr(inner, "weights_", None)
        expanded_weights, resolved_names = self._expand_weights_to_input(
            X, self.best_estimator_, inner, raw_weights
        )
        self.weights_ = expanded_weights

        if resolved_names is not None:
            self.feature_names_in_ = np.asarray(resolved_names, dtype=object)
        elif hasattr(self, "feature_names_in_"):
            delattr(self, "feature_names_in_")

        for attr in ("management_fees", "transaction_costs"):
            if hasattr(inner, attr):
                setattr(self, attr, getattr(inner, attr))

        inner_ptf_params = getattr(inner, "portfolio_params", None)
        if isinstance(inner_ptf_params, dict):
            merged_ptf_params = dict(inner_ptf_params)
            if isinstance(self.portfolio_params, dict):
                merged_ptf_params.update(self.portfolio_params)
            self._resolved_portfolio_params_: dict[str, Any] | None = merged_ptf_params
        else:
            self._resolved_portfolio_params_ = (
                dict(self.portfolio_params)
                if isinstance(self.portfolio_params, dict)
                else self.portfolio_params
            )

        resolved_rfr = self._as_optional_float(
            self._resolved_portfolio_params_.get("risk_free_rate")
            if isinstance(self._resolved_portfolio_params_, dict)
            else None
        )
        if resolved_rfr is None:
            resolved_rfr = self._as_optional_float(
                getattr(inner, "risk_free_rate", None)
            )
        if resolved_rfr is not None:
            self.risk_free_rate = resolved_rfr

        if hasattr(inner, "fallback_chain_"):
            self.fallback_chain_ = inner.fallback_chain_
        if hasattr(inner, "error_"):
            self.error_ = inner.error_

        return self

    def predict(self, X: Any) -> Any:
        resolved_params = getattr(
            self, "_resolved_portfolio_params_", self.portfolio_params
        )
        if resolved_params != self.portfolio_params:
            delegate = SkfolioGridSearchCV(
                estimator=self.estimator,
                param_grid=self.param_grid,
                portfolio_params=resolved_params,
                previous_weights=self.previous_weights,
            )
            _copy_fitted_attrs(self, delegate)
            for attr in ("management_fees", "transaction_costs", "risk_free_rate"):
                if hasattr(self, attr):
                    setattr(delegate, attr, getattr(self, attr))
            return BaseOptimization.predict(delegate, X)
        return BaseOptimization.predict(self, X)
