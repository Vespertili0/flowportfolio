from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from skfolio.optimization import BaseOptimization
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


def set_previous_weights_chain(opt: Any, weights: Any) -> None:
    """Propagate previous_weights across a Pipeline, optimizer, and its entire fallback chain."""
    if isinstance(opt, Pipeline):
        for _, step in opt.steps:
            set_previous_weights_chain(step, weights)
        return

    if not hasattr(opt, "_mock_name"):
        if hasattr(opt, "estimator") and opt.estimator is not None:
            set_previous_weights_chain(opt.estimator, weights)
        if hasattr(opt, "best_estimator_") and opt.best_estimator_ is not None:
            set_previous_weights_chain(opt.best_estimator_, weights)

    current = opt
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if hasattr(current, "previous_weights"):
            current.previous_weights = (
                dict(weights) if isinstance(weights, dict) else weights
            )
        current = getattr(current, "fallback", None)


class SkfolioGridSearchCV(BaseOptimization, GridSearchCV):
    """A wrapper for GridSearchCV that conforms to skfolio's BaseOptimization API.

    This ensures that when skfolio's cross_val_predict uses this search object,
    it exposes weights_ and predict correctly as an optimization estimator.
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

    def _get_target_optimizer(self, est: Any) -> Any:
        """Return the terminal optimiser of a Pipeline, or ``est`` itself."""
        if isinstance(est, Pipeline) and est.steps:
            return est.steps[-1][1]
        return est

    @property
    def needs_previous_weights(self) -> bool:
        """Whether the wrapped optimiser requires sequential previous weights."""
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

    def fit(self, X: Any, y: Any = None, **fit_params: Any) -> "SkfolioGridSearchCV":
        target = (
            self._get_target_optimizer(self.estimator)
            if self.estimator is not None
            else None
        )
        if self.previous_weights is not None and target is not None:
            if (
                isinstance(X, pd.DataFrame)
                and len(X.columns) > 0
                and getattr(self, "refit", True)
            ):
                eq_w = {col: 1.0 / len(X.columns) for col in X.columns}
                set_previous_weights_chain(self.estimator, eq_w)
            else:
                set_previous_weights_chain(self.estimator, self.previous_weights)

        GridSearchCV.fit(self, X, y, **fit_params)

        if (
            self.previous_weights is not None
            and getattr(self, "refit", True)
            and hasattr(self, "best_estimator_")
        ):
            set_previous_weights_chain(self.best_estimator_, self.previous_weights)
            self.best_estimator_.fit(X, y, **fit_params)

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

        for attr in ("management_fees", "transaction_costs", "risk_free_rate"):
            if hasattr(inner, attr):
                setattr(self, attr, getattr(inner, attr))

        inner_ptf_params = getattr(inner, "portfolio_params", None)
        if isinstance(inner_ptf_params, dict):
            merged_ptf_params = dict(inner_ptf_params)
            if isinstance(self.portfolio_params, dict):
                merged_ptf_params.update(self.portfolio_params)
            self.portfolio_params = merged_ptf_params

        if hasattr(inner, "fallback_chain_"):
            self.fallback_chain_ = inner.fallback_chain_
        if hasattr(inner, "error_"):
            self.error_ = inner.error_

        return self
