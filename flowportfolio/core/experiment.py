"""Benchmarking harness module.

This module provides the :class:`PortfolioExperimentEngine`, which automates
hyperparameter tuning and walk-forward cross-validation simulation across
multiple investment strategies using ``skfolio``.
"""

from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from skfolio import Population, RatioMeasure
from skfolio.metrics import make_scorer
from skfolio.model_selection import (
    cross_val_predict,
    WalkForward,
    CombinatorialPurgedCV,
    MultipleRandomizedCV,
    OnlineGridSearch,
    online_predict,
)
from skfolio.moments import EWMu, EWCovariance
from flowportfolio.core.universe import Universe


class PortfolioExperimentEngine:
    """Automates hyperparameter tuning and cross-validation pipelines.

    Maintains a registry of named strategies and evaluates them uniformly
    against a common asset universe and cross-validation protocol.

    Parameters
    ----------
    universe : Universe
        The asset universe providing historical returns and fees.
    constraints : list[str]
        A list of ``skfolio``-compatible linear constraint strings. Stored
        on the engine (as ``self._constraints``) so that callers can
        inject them into estimators before registering them via
        ``add_strategy``. The engine does not automatically inject these
        constraints into the estimators.
    n_jobs : int, default -1
        The number of parallel jobs to run during cross-validation.
        ``-1`` means using all available processors.

    Raises
    ------
    TypeError
        If ``universe`` is not a :class:`Universe` instance, or if
        ``constraints`` is not a list.
    """

    def __init__(
        self,
        universe: Universe,
        constraints: list[str],
        n_jobs: int = -1,
    ) -> None:
        if not isinstance(universe, Universe):
            raise TypeError("universe must be a Universe instance.")
        if not isinstance(constraints, list):
            raise TypeError("constraints must be a list of strings.")

        self._universe = universe
        self._constraints = list(constraints)
        self._n_jobs = n_jobs
        self._strategies: dict[str, dict] = {}

    def add_strategy(
        self,
        name: str,
        estimator: BaseEstimator,
        grid: dict,
    ) -> None:
        """Register a new strategy for benchmarking.

        Parameters
        ----------
        name : str
            A unique name for the strategy.
        estimator : BaseEstimator
            The scikit-learn compatible estimator or pipeline to be tuned.
        grid : dict
            The hyperparameter search grid for the estimator.

        Raises
        ------
        TypeError
            If ``name`` is not a string, ``estimator`` is not a ``BaseEstimator``,
            or ``grid`` is not a dictionary.
        ValueError
            If ``name`` is empty or if the strategy name is already registered.
        """
        if not isinstance(name, str):
            raise TypeError("Strategy name must be a string.")
        if not name.strip():
            raise ValueError("Strategy name cannot be empty.")
        if not isinstance(estimator, BaseEstimator):
            raise TypeError("estimator must be a scikit-learn BaseEstimator.")
        if not isinstance(grid, dict):
            raise TypeError("grid must be a dictionary.")
        if name in self._strategies:
            raise ValueError(f"Strategy '{name}' is already registered.")

        self._strategies[name] = {
            "estimator": estimator,
            "grid": grid,
        }

    def _resolve_cv_splitter(self, cv_type: str, **cv_kwargs):
        """Resolve a cv_type string to a skfolio CV splitter instance.

        Parameters
        ----------
        cv_type : str
            One of "walk_forward", "combinatorial", "randomised".
        **cv_kwargs
            Constructor arguments forwarded to the splitter.

        Returns
        -------
        A configured skfolio CV splitter instance.

        Raises
        ------
        ValueError
            If cv_type is not one of the three permitted values.
        """
        _map = {
            "walk_forward": WalkForward,
            "combinatorial": CombinatorialPurgedCV,
            "randomised": MultipleRandomizedCV,
        }
        if cv_type not in _map:
            raise ValueError(
                f"Unknown cv_type: '{cv_type}'. Must be one of "
                "'walk_forward', 'combinatorial', or 'randomised'."
            )
        return _map[cv_type](**cv_kwargs)

    def run_robustness_test(
        self,
        cv_type: str = "walk_forward",
        **cv_kwargs,
    ) -> Population:
        """Execute the full benchmarking pipeline for all registered strategies.

        Performs a two-step process for each strategy:
        1. Tunes hyperparameters via ``GridSearchCV`` over the full dataset,
           refitting the best estimator.
        2. Simulates an out-of-sample rebalancing journey using
           ``cross_val_predict`` with the best tuned estimator.

        Parameters
        ----------
        cv_type : {"walk_forward", "combinatorial", "randomised"}, default "walk_forward"
            The cross-validation protocol to use.
        **cv_kwargs
            Additional keyword arguments forwarded to the underlying
            cross-validator constructor (e.g., ``train_size``, ``test_size``).

        Returns
        -------
        Population
            A combined population containing the out-of-sample portfolio
            trajectories for all strategies. Each portfolio is tagged with
            its generating strategy name.

        Raises
        ------
        ValueError
            If ``cv_type`` is not one of the supported strings, or if
            ``universe.returns`` is unavailable.
        """
        # 1. Resolve CV protocol
        cv = self._resolve_cv_splitter(cv_type, **cv_kwargs)

        # 2. Extract returns (will raise ValueError if not fetched)
        returns = self._universe.returns

        if not self._strategies:
            raise RuntimeError("No strategies registered. Call add_strategy() first.")

        # 3. Process each strategy
        collected = []
        for name, config in self._strategies.items():
            # a. Hyperparameter search
            search = GridSearchCV(
                estimator=config["estimator"],
                param_grid=config["grid"],
                cv=cv,
                scoring=make_scorer(RatioMeasure.CVAR_RATIO),
                refit=True,
                n_jobs=self._n_jobs,
            )
            search.fit(returns)
            best_model = search.best_estimator_

            # b. Out-of-sample journey simulation
            portfolio = cross_val_predict(
                best_model,
                returns,
                cv=cv,
                n_jobs=self._n_jobs,
                portfolio_params={"tag": name},
            )
            
            collected.append(portfolio)

        return Population(collected)

    def run_online_evaluation(
        self,
        cv_type: str = "walk_forward",
        **cv_kwargs,
    ) -> Population:
        """Run online (streaming) evaluation of all registered strategies.

        Mirrors run_robustness_test() but uses OnlineGridSearch and
        online_predict to support incremental model updates. This is
        appropriate for strategies using exponentially weighted estimators
        (EWMu, EWCovariance) that support partial_fit().

        Prefect Compatibility: This method is a pure function of self.strategies
        and self.universe.returns. It holds no stateful connections between calls.

        Parameters
        ----------
        cv_type : str, optional
            Cross-validation strategy. One of: "walk_forward",
            "combinatorial", "randomised". Default is "walk_forward".
        **cv_kwargs
            Additional keyword arguments forwarded to the chosen CV splitter
            constructor (e.g., train_size=252, test_size=63).

        Returns
        -------
        skfolio.Population
            A Population object containing one MultiPeriodPortfolio per
            registered strategy, each tagged with the strategy name.

        Raises
        ------
        ValueError
            If cv_type is not one of the three permitted string values.
        RuntimeError
            If no strategies have been registered via add_strategy().
        """
        if not self._strategies:
            raise RuntimeError("No strategies registered. Call add_strategy() first.")

        cv = self._resolve_cv_splitter(cv_type, **cv_kwargs)
        returns = self._universe.returns

        collected = []
        for name, config in self._strategies.items():
            search = OnlineGridSearch(
                estimator=config["estimator"],
                param_grid=config["grid"],
                cv=cv,
                scoring=make_scorer(RatioMeasure.CVAR_RATIO),
                refit=True,
                n_jobs=self._n_jobs,
            )
            search.fit(returns)
            best_model = search.best_estimator_

            portfolio = online_predict(
                best_model,
                returns,
                cv=cv,
                n_jobs=self._n_jobs,
                portfolio_params={"tag": name},
            )
            
            collected.append(portfolio)

        return Population(collected)
