"""Benchmarking harness module.

This module provides the :class:`PortfolioExperimentEngine`, which automates
hyperparameter tuning and walk-forward cross-validation simulation across
multiple investment strategies using ``skfolio``.
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from skfolio import Population
from skfolio.measures import RatioMeasure
from skfolio.metrics import make_scorer
from skfolio.model_selection import (
    CombinatorialPurgedCV,
    MultipleRandomizedCV,
    OnlineGridSearch,
    WalkForward,
    cross_val_predict,
    online_predict,
)
from sklearn.base import BaseEstimator, clone

from flowportfolio.core.optimisation import SkfolioGridSearchCV as GridSearchCV
from flowportfolio.core.optimisation import set_previous_weights_chain
from flowportfolio.core.protocols import UniverseProtocol

logger = logging.getLogger(__name__)


class StrategyConfig(TypedDict):
    estimator: BaseEstimator
    grid: dict


class PortfolioExperimentEngine:
    """Automates hyperparameter tuning via nested cross-validation.

    Maintains a registry of named strategies and evaluates them uniformly
    against a common asset universe and cross-validation protocol.

    Parameters
    ----------
    universe : Universe
        The asset universe providing historical returns and fees.
    n_jobs : int, default -1
        The number of parallel jobs to run during cross-validation.
        ``-1`` means using all available processors.

    Raises
    ------
    TypeError
        If ``universe`` is not a :class:`Universe` instance.
    """

    def __init__(
        self,
        universe: UniverseProtocol,
        n_jobs: int = -1,
    ) -> None:
        from flowportfolio.core.protocols import is_universe_protocol

        if not is_universe_protocol(universe):
            raise TypeError("universe must implement UniverseProtocol.")
        self._universe = universe
        self._n_jobs = n_jobs
        self._strategies: dict[str, StrategyConfig] = {}
        self._latest_weights: dict[str, dict[str, float]] = {}

    @property
    def latest_weights(self) -> dict[str, dict[str, float]]:
        """Live target weights fitted on the trailing window up to the final observation."""
        return {k: dict(v) for k, v in self._latest_weights.items()}

    def add_strategy(
        self,
        name: str,
        estimator: BaseEstimator,
        grid: dict[str, Any],
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
            "estimator": clone(estimator),
            "grid": dict(grid),
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

    @staticmethod
    def _derive_inner_walk_forward_sizes(
        outer_train: int, purged_size: int = 0
    ) -> tuple[int, int]:
        """Derive inner walk-forward train and test sizes constrained by outer train window.

        Parameters
        ----------
        outer_train : int
            The number of observations in the outer training window.
        purged_size : int, default 0
            The purged/embargo buffer between folds.

        Returns
        -------
        tuple[int, int]
            (inner_train, inner_test) sizes.

        Raises
        ------
        ValueError
            If the available training window is <= 2 observations.
        """
        available = outer_train - max(0, int(purged_size))
        if available <= 2:
            raise ValueError(
                f"Outer train_size={outer_train} (with purged_size={purged_size}) "
                "is too small to derive inner walk-forward splits."
            )
        inner_test = max(1, int(available * 0.2))
        inner_train = max(1, int(available * 0.6))
        if inner_train + inner_test > available:
            inner_train = max(1, available - inner_test)
        return inner_train, inner_test

    @staticmethod
    def _resolve_outer_train_size(
        cv_type: str,
        outer_cv: Any,
        outer_kwargs: dict[str, Any],
        n_samples: int,
        returns: Any,
    ) -> int:
        """Determine the (minimum) number of observations in an outer training fold."""
        if cv_type == "walk_forward":
            return int(outer_kwargs.get("train_size", 252))

        if cv_type == "combinatorial":
            try:
                return max(
                    1, min(len(train_idx) for train_idx, _ in outer_cv.split(returns))
                )
            except Exception:  # noqa: BLE001
                n_folds = outer_kwargs.get("n_folds", 10)
                n_test_folds = outer_kwargs.get("n_test_folds", 8)
                return max(1, int(n_samples * (n_folds - n_test_folds) / n_folds))

        base_cv = outer_kwargs.get("walk_forward")
        train_size = getattr(base_cv, "train_size", None)
        if isinstance(train_size, int):
            return min(train_size, n_samples)
        return n_samples

    def run_robustness_test(
        self,
        cv_type: str = "walk_forward",
        inner_cv_kwargs: dict | None = None,
        **cv_kwargs,
    ) -> Population:
        """Execute the full benchmarking pipeline for all registered strategies.

        Performs a two-step process for each strategy:
        1. Tunes hyperparameters via ``GridSearchCV`` using an inner cross-validator
           derived to fit within the outer training window.
        2. Simulates an out-of-sample rebalancing journey using
           ``cross_val_predict`` with the outer cross-validator.

        Parameters
        ----------
        cv_type : {"walk_forward", "combinatorial", "randomised"}, default "walk_forward"
            The cross-validation protocol to use.
        inner_cv_kwargs : dict or None, optional
            Explicit constructor arguments for the inner CV splitter used in
            hyperparameter search. If None, derived dynamically from the outer
            training parameters.
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
        # 1. Resolve CV protocols
        resolved_outer_kwargs = dict(cv_kwargs)
        if cv_type == "walk_forward":
            resolved_outer_kwargs.setdefault("train_size", 252)
            resolved_outer_kwargs.setdefault("test_size", 63)
        outer_cv = self._resolve_cv_splitter(cv_type, **resolved_outer_kwargs)

        # 2. Extract returns (will raise ValueError if not fetched)
        returns = self._universe.returns

        # Determine the size of the outer training window, which bounds the data
        # available to the inner hyperparameter search.
        outer_train = self._resolve_outer_train_size(
            cv_type, outer_cv, resolved_outer_kwargs, len(returns), returns
        )

        if inner_cv_kwargs is not None:
            resolved_explicit_inner = dict(inner_cv_kwargs)
            inner_type = resolved_explicit_inner.pop("cv_type", "walk_forward")
            if inner_type in ("combinatorial", "randomised"):
                raise ValueError(
                    f"Inner cross-validation splitter cannot be '{inner_type}'. "
                    "Hyperparameter tuning via GridSearchCV is restricted to single-test-fold splitters such as 'walk_forward'."
                )
            if inner_type == "walk_forward" and (
                "train_size" not in resolved_explicit_inner
                or "test_size" not in resolved_explicit_inner
            ):
                inner_train, inner_test = self._derive_inner_walk_forward_sizes(
                    outer_train, resolved_explicit_inner.get("purged_size", 0)
                )
                resolved_explicit_inner.setdefault("train_size", inner_train)
                resolved_explicit_inner.setdefault("test_size", inner_test)
            inner_cv = self._resolve_cv_splitter(inner_type, **resolved_explicit_inner)
        else:
            if cv_type == "walk_forward":
                resolved_inner_kwargs = dict(cv_kwargs)
                inner_train, inner_test = self._derive_inner_walk_forward_sizes(
                    outer_train, cv_kwargs.get("purged_size", 0)
                )
                resolved_inner_kwargs["train_size"] = inner_train
                resolved_inner_kwargs["test_size"] = inner_test

                inner_cv = self._resolve_cv_splitter(
                    "walk_forward", **resolved_inner_kwargs
                )
            else:
                # CombinatorialPurgedCV and MultipleRandomizedCV are multi-path splitters
                # incompatible with GridSearchCV. Restrict inner tuning to standard single-test-fold WalkForward.
                inner_train, inner_test = self._derive_inner_walk_forward_sizes(
                    outer_train, 0
                )
                inner_cv = self._resolve_cv_splitter(
                    "walk_forward", train_size=inner_train, test_size=inner_test
                )

        if not self._strategies:
            raise RuntimeError("No strategies registered. Call add_strategy() first.")

        # 3. Process each strategy
        collected = []
        equal_weights = {col: 1.0 / len(returns.columns) for col in returns.columns}
        for name, config in self._strategies.items():
            est_clone = clone(config["estimator"])
            target_opt = (
                est_clone.steps[-1][1]
                if hasattr(est_clone, "steps") and est_clone.steps
                else est_clone
            )
            # Prevent present-day physical weights from contaminating historical fold 0
            # while keeping turnover constraints active across sequential walk-forward folds.
            if (
                getattr(target_opt, "max_turnover", None) is not None
                and getattr(target_opt, "previous_weights", None) is not None
            ):
                set_previous_weights_chain(target_opt, equal_weights)

            # a. Hyperparameter search
            search = GridSearchCV(
                estimator=est_clone,
                param_grid=config["grid"],
                cv=inner_cv,
                scoring=make_scorer(RatioMeasure.CVAR_RATIO),
                refit=True,
                n_jobs=1,
            )

            # b. Out-of-sample journey simulation
            portfolio = cross_val_predict(
                search,
                returns,
                cv=outer_cv,
                n_jobs=self._n_jobs,
                portfolio_params={"tag": name},
            )

            # c. Live terminal refit on trailing outer_train observations
            try:
                live_search = clone(search)
                live_window = returns.iloc[-min(outer_train, len(returns)) :]
                # Anchor live refit turnover to final OOS portfolio weights if present
                last_oos_weights = None
                ports_list = (
                    list(portfolio)
                    if type(portfolio).__name__ == "Population"
                    or isinstance(portfolio, list)
                    else [portfolio]
                )
                if ports_list:
                    last_p = ports_list[-1]
                    if hasattr(last_p, "portfolios") and last_p.portfolios:
                        last_oos_weights = getattr(
                            last_p.portfolios[-1], "weights_dict", None
                        )
                    else:
                        last_oos_weights = getattr(last_p, "weights_dict", None)
                if (
                    last_oos_weights
                    and getattr(target_opt, "max_turnover", None) is not None
                ):
                    set_previous_weights_chain(live_search, last_oos_weights)

                live_search.fit(live_window)
                if getattr(live_search, "weights_", None) is not None:
                    feature_names = getattr(
                        live_search, "feature_names_in_", live_window.columns
                    )
                    if len(feature_names) == len(live_search.weights_):
                        live_w = {
                            str(col): float(w)
                            for col, w in zip(feature_names, live_search.weights_)
                        }
                    else:
                        live_w = {
                            str(col): float(w)
                            for col, w in zip(live_window.columns, live_search.weights_)
                        }
                    self._latest_weights[name] = live_w
                    for p_obj in ports_list:
                        try:
                            p_obj.latest_weights_dict = dict(live_w)
                        except AttributeError:
                            pass
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Live terminal refit failed for strategy '%s'", name, exc_info=True
                )

            if type(portfolio).__name__ == "Population" or isinstance(portfolio, list):
                collected.extend(portfolio)
            else:
                collected.append(portfolio)

        return Population(collected)

    @staticmethod
    def _validate_online_estimator(name: str, est: BaseEstimator) -> None:
        """Validate that an estimator and its prior hierarchy support online estimation."""
        if not hasattr(est, "partial_fit"):
            raise TypeError(
                f"Strategy '{name}' estimator '{type(est).__name__}' does not implement "
                "partial_fit() and cannot be evaluated online."
            )
        prior_est = getattr(est, "prior_estimator", None)
        if prior_est is None:
            raise TypeError(
                f"Strategy '{name}' has prior_estimator=None, which defaults to batch "
                "EmpiricalPrior(EmpiricalMu(), EmpiricalCovariance()). Provide an online "
                "prior (e.g., EmpiricalPrior(mu_estimator=EWMu(), covariance_estimator=EWCovariance()))."
            )
        if not hasattr(prior_est, "partial_fit"):
            raise TypeError(
                f"Strategy '{name}' uses batch prior '{type(prior_est).__name__}', "
                "which does not implement partial_fit(). Use an online prior "
                "(e.g., EmpiricalPrior(mu_estimator=EWMu(), covariance_estimator=EWCovariance())) "
                "or use run_robustness_test() for batch cross-validation."
            )
        for sub_attr in ("mu_estimator", "covariance_estimator"):
            sub_est = getattr(prior_est, sub_attr, None)
            if sub_est is not None and not hasattr(sub_est, "partial_fit"):
                raise TypeError(
                    f"Strategy '{name}' prior '{type(prior_est).__name__}' uses batch "
                    f"{sub_attr} '{type(sub_est).__name__}', which does not implement partial_fit()."
                )

    def run_online_evaluation(
        self,
        cv_type: str = "walk_forward",
        burn_in_size: int | None = None,
        **cv_kwargs,
    ) -> Population:
        """Run online (streaming) evaluation of all registered strategies.

        Mirrors run_robustness_test() but fits hyperparameter search on a
        dedicated historical burn-in period prior to executing online prediction,
        preventing look-ahead bias.

        Parameters
        ----------
        cv_type : str, optional
            Cross-validation strategy. One of: "walk_forward",
            "combinatorial", "randomised". Default is "walk_forward".
        burn_in_size : int or None, optional
            Number of initial observations allocated for hyperparameter tuning.
            If None, defaults to the train_size or warmup_size specified in
            cv_kwargs, or 252.
        **cv_kwargs
            Additional keyword arguments forwarded to the chosen CV splitter
            constructor or online prediction parameters.

        Returns
        -------
        skfolio.Population
            A Population object containing one MultiPeriodPortfolio per
            registered strategy, each tagged with the strategy name.

        Raises
        ------
        ValueError
            If cv_type is not one of the three permitted string values, if
            returns is shorter than burn_in_size, or if a registered strategy
            is a multi-step Pipeline (pre-selection steps are unsupported by
            online learning).
        TypeError
            If a registered strategy uses a batch prior that does not
            implement ``partial_fit``.
        RuntimeError
            If no strategies have been registered via add_strategy().
        """
        if not self._strategies:
            raise RuntimeError("No strategies registered. Call add_strategy() first.")

        returns = self._universe.returns

        if burn_in_size is None:
            burn_in_size = cv_kwargs.get(
                "warmup_size", cv_kwargs.get("train_size", 252)
            )

        if burn_in_size >= len(returns):
            raise ValueError(
                f"burn_in_size={burn_in_size} must be strictly less than total "
                f"observations ({len(returns)}) to allow out-of-sample evaluation."
            )

        burn_in_returns = returns.iloc[:burn_in_size]

        resolved_cv_kwargs = {
            k: v
            for k, v in cv_kwargs.items()
            if k not in {"warmup_size", "previous", "params"}
        }
        if cv_type == "walk_forward":
            purged = resolved_cv_kwargs.get("purged_size", 0)
            if (
                "train_size" not in resolved_cv_kwargs
                or "test_size" not in resolved_cv_kwargs
                or int(resolved_cv_kwargs["train_size"])
                + int(resolved_cv_kwargs["test_size"])
                + int(purged)
                > burn_in_size
            ):
                inner_train, inner_test = self._derive_inner_walk_forward_sizes(
                    burn_in_size, purged
                )
                resolved_cv_kwargs["train_size"] = inner_train
                resolved_cv_kwargs["test_size"] = inner_test

        cv = self._resolve_cv_splitter(cv_type, **resolved_cv_kwargs)

        online_predict_keys = {
            "test_size",
            "freq",
            "freq_offset",
            "previous",
            "purged_size",
            "params",
        }
        predict_kwargs = {
            k: v for k, v in cv_kwargs.items() if k in online_predict_keys
        }

        collected = []
        equal_weights = {col: 1.0 / len(returns.columns) for col in returns.columns}
        for name, config in self._strategies.items():
            est = clone(config["estimator"])
            grid = dict(config["grid"])

            if hasattr(est, "steps"):
                if len(est.steps) > 1:
                    raise ValueError(
                        f"Strategy '{name}' contains pipeline pre-selection steps, which "
                        "are not supported by skfolio online_predict/OnlineGridSearch."
                    )
                est = est.steps[-1][1]
                grid = {
                    (k.split("__", 1)[1] if k.startswith("optimizer__") else k): v
                    for k, v in grid.items()
                }

            self._validate_online_estimator(name, est)

            if (
                getattr(est, "max_turnover", None) is not None
                and getattr(est, "previous_weights", None) is not None
            ):
                set_previous_weights_chain(est, equal_weights)

            search = OnlineGridSearch(
                estimator=est,
                param_grid=grid,
                cv=cv,
                scoring=make_scorer(RatioMeasure.CVAR_RATIO),
                refit=True,
                n_jobs=1,
            )
            search.fit(burn_in_returns)
            best_model = search.best_estimator_

            portfolio = online_predict(
                best_model,
                returns,
                warmup_size=burn_in_size,
                portfolio_params={"tag": name},
                **predict_kwargs,
            )

            if type(portfolio).__name__ == "Population" or isinstance(portfolio, list):
                collected.extend(portfolio)
            else:
                collected.append(portfolio)

        return Population(collected)
