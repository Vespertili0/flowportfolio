from collections.abc import Callable
from typing import Any

import numpy as np
from skfolio.optimization import BaseOptimization
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


def set_previous_weights_chain(opt: Any, weights: Any) -> None:
    """Propagate previous_weights across an optimizer and its entire fallback chain."""
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
        if self.portfolio_params is not None and self.portfolio_params.get(
            "weight_drift", False
        ):
            return True
        return False

    def fit(self, X: Any, y: Any = None, **fit_params: Any) -> "SkfolioGridSearchCV":
        target = (
            self._get_target_optimizer(self.estimator)
            if self.estimator is not None
            else None
        )
        if self.previous_weights is not None and target is not None:
            # Use neutral 1/N weights during inner CV splits so outer boundary weights
            # from the end of X do not leak backward into earlier inner folds.
            if hasattr(X, "columns") and len(X.columns) > 0:
                neutral_weights = {col: 1.0 / len(X.columns) for col in X.columns}
                set_previous_weights_chain(target, neutral_weights)
            else:
                set_previous_weights_chain(target, self.previous_weights)

        GridSearchCV.fit(self, X, y, **fit_params)

        if self.previous_weights is not None and target is not None:
            set_previous_weights_chain(target, self.previous_weights)
            if self.refit and hasattr(self, "best_estimator_"):
                best_target = self._get_target_optimizer(self.best_estimator_)
                set_previous_weights_chain(best_target, self.previous_weights)
                self.best_estimator_.fit(X, y, **fit_params)

        inner = self._get_target_optimizer(self.best_estimator_)
        self.weights_ = getattr(inner, "weights_", None)
        if hasattr(inner, "feature_names_in_"):
            self.feature_names_in_ = inner.feature_names_in_
        if hasattr(inner, "fallback_chain_"):
            self.fallback_chain_ = inner.fallback_chain_
        if hasattr(inner, "error_"):
            self.error_ = inner.error_

        return self
