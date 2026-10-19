from collections.abc import Callable
from typing import Any

import numpy as np
from skfolio.optimization import BaseOptimization
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


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
        target = self._get_target_optimizer(self.estimator)
        if bool(getattr(target, "needs_previous_weights", False)):
            return True
        if self.portfolio_params is not None and self.portfolio_params.get(
            "weight_drift", False
        ):
            return True
        return False

    def fit(self, X: Any, y: Any = None, **fit_params: Any) -> "SkfolioGridSearchCV":
        if self.previous_weights is not None and self.estimator is not None:
            target = self._get_target_optimizer(self.estimator)
            if hasattr(target, "previous_weights"):
                target.previous_weights = self.previous_weights
                if hasattr(target, "fallback") and target.fallback is not None:
                    if hasattr(target.fallback, "previous_weights"):
                        target.fallback.previous_weights = self.previous_weights

        GridSearchCV.fit(self, X, y, **fit_params)

        inner = self._get_target_optimizer(self.best_estimator_)
        self.weights_ = getattr(inner, "weights_", None)
        if hasattr(inner, "feature_names_in_"):
            self.feature_names_in_ = inner.feature_names_in_
        if hasattr(inner, "fallback_chain_"):
            self.fallback_chain_ = inner.fallback_chain_
        if hasattr(inner, "error_"):
            self.error_ = inner.error_

        return self
