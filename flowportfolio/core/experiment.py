"""Benchmarking harness module.

This module provides the :class:`PortfolioExperimentEngine`, which automates
hyperparameter tuning and walk-forward cross-validation simulation across
multiple investment strategies using ``skfolio``.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any, Literal, TypedDict, overload

import numpy as np
import pandas as pd
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


@dataclass(frozen=True)
class ExperimentResult:
    """Immutable container pairing an OOS Population with live terminal weights.

    Unlike dynamic attributes attached onto ``skfolio`` ``Population`` or
    ``__slots__``-based ``MultiPeriodPortfolio`` instances, this explicit
    result object preserves live terminal weights across downstream slicing,
    filtering, or ``Population(list(pop))`` reconstruction.
    """

    population: Population
    live_weights: dict[str, dict[str, float]]


class _GapAwareWalkForward(WalkForward):
    """WalkForward splitter that never spans purged discontinuities in CPCV folds.

    When an outer ``CombinatorialPurgedCV`` training slice omits purged/test
    blocks from ``reference_index``, this splitter partitions the slice into
    maximal contiguous runs (adjacent positions in ``reference_index``) and
    yields walk-forward ``(train, test)`` pairs strictly inside each contiguous
    segment.
    """

    def __init__(
        self,
        train_size: int = 252,
        test_size: int = 63,
        *,
        purged_size: int = 0,
        reference_index: pd.Index | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            train_size=train_size,
            test_size=test_size,
            purged_size=purged_size,
            **kwargs,
        )
        self.reference_index = reference_index

    def _contiguous_segments(self, X: Any) -> list[np.ndarray] | None:
        if self.reference_index is None or not hasattr(X, "index"):
            return None
        orig_pos = self.reference_index.get_indexer(X.index)
        if len(orig_pos) == 0 or bool(np.any(orig_pos < 0)):
            return None
        break_pts = np.where(np.diff(orig_pos) != 1)[0] + 1
        if len(break_pts) == 0:
            return None
        return [
            seg for seg in np.split(np.arange(len(orig_pos)), break_pts) if len(seg) > 0
        ]

    @staticmethod
    def _yield_segment_splits(
        seg: np.ndarray, train_size: int, test_size: int, purged_size: int
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        splits: list[tuple[np.ndarray, np.ndarray]] = []
        n = len(seg)
        offset = 0
        while offset + train_size + purged_size + test_size <= n:
            tr = seg[offset : offset + train_size]
            te_start = offset + train_size + purged_size
            te = seg[te_start : te_start + test_size]
            splits.append((tr, te))
            offset += test_size
        return splits

    def split(self, X: Any, y: Any = None, groups: Any = None):
        segments = self._contiguous_segments(X)
        if segments is None:
            yield from super().split(X, y=y, groups=groups)
            return

        purge = max(0, int(self.purged_size))
        tr_sz = int(self.train_size)
        te_sz = int(self.test_size)

        collected: list[tuple[np.ndarray, np.ndarray]] = []
        for seg in segments:
            collected.extend(self._yield_segment_splits(seg, tr_sz, te_sz, purge))

        if not collected:
            # Segments are shorter than requested train+test; adapt within each
            # contiguous segment while strictly respecting segment boundaries.
            for seg in segments:
                n = len(seg)
                if n < 2:
                    continue
                seg_purge = purge if n - purge >= 2 else 0
                avail = n - seg_purge
                seg_te = max(1, int(avail * 0.2))
                seg_tr = max(1, min(int(avail * 0.6), avail - seg_te))
                collected.extend(
                    self._yield_segment_splits(seg, seg_tr, seg_te, seg_purge)
                )

        if not collected:
            raise ValueError(
                "No contiguous segment inside the CombinatorialPurgedCV training "
                "fold has at least 2 observations to form an inner walk-forward split."
            )

        yield from collected

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        if X is not None and self._contiguous_segments(X) is not None:
            return sum(1 for _ in self.split(X, y=y, groups=groups))
        return super().get_n_splits(X=X, y=y, groups=groups)


def _has_transaction_costs(opt: Any) -> bool:
    """Return True if ``opt`` configures any strictly positive transaction cost."""
    tc = getattr(opt, "transaction_costs", None)
    if tc is None:
        return False
    if isinstance(tc, (int, float)) and not isinstance(tc, bool):
        return float(tc) > 0.0
    if isinstance(tc, dict):
        return any(float(v) > 0.0 for v in tc.values())
    try:
        return bool(np.any(np.asarray(tc, dtype=float) > 0.0))
    except (TypeError, ValueError):
        return False


def _has_rebalancing_friction(opt: Any) -> bool:
    """Return True if ``opt`` or any fallback has max_turnover or transaction_costs."""
    current = opt
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if getattr(current, "max_turnover", None) is not None:
            return True
        if _has_transaction_costs(current):
            return True
        current = getattr(current, "fallback", None)
    return False


def _collect_chain_neutral_params(
    opt: Any, prefix: str, params: dict[str, Any]
) -> None:
    """Populate Fold-0 neutral entry params across an optimizer and its fallback chain."""
    current = opt
    cur_prefix = prefix
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if getattr(current, "max_turnover", None) is not None:
            params[f"{cur_prefix}max_turnover"] = None
        if _has_transaction_costs(current):
            params[f"{cur_prefix}transaction_costs"] = 0.0
        current = getattr(current, "fallback", None)
        cur_prefix = f"{cur_prefix}fallback__"


def _build_entry_rebalancing_params(
    est: Any, *, prefix: str = "estimator__"
) -> dict[str, Any] | None:
    """Build ``skfolio`` ``entry_rebalancing_params`` neutralizing Fold-0 entry friction."""
    params: dict[str, Any] = {}
    if hasattr(est, "steps") and est.steps:
        for step_name, step in est.steps:
            _collect_chain_neutral_params(step, f"{prefix}{step_name}__", params)
    else:
        _collect_chain_neutral_params(est, prefix, params)
    return params or None


def _extract_weights_dict(
    model: Any, fallback_columns: pd.Index
) -> dict[str, float] | None:
    """Extract a ``{ticker: float}`` mapping from a fitted estimator's ``weights_``."""
    weights = getattr(model, "weights_", None)
    if weights is None:
        return None
    feature_names = getattr(model, "feature_names_in_", fallback_columns)
    cols = feature_names if len(feature_names) == len(weights) else fallback_columns
    return {str(col): float(w) for col, w in zip(cols, weights)}


def _extract_terminal_oos_weights(portfolio: Any) -> dict[str, float] | None:
    """Extract the terminal OOS weights_dict from a Portfolio, MultiPeriodPortfolio, or Population."""
    ports_list = (
        list(portfolio)
        if type(portfolio).__name__ == "Population" or isinstance(portfolio, list)
        else [portfolio]
    )
    if not ports_list:
        return None
    last_p = ports_list[-1]
    raw = (
        getattr(last_p.portfolios[-1], "weights_dict", None)
        if hasattr(last_p, "portfolios") and last_p.portfolios
        else getattr(last_p, "weights_dict", None)
    )
    if isinstance(raw, dict):
        return {str(k): float(v) for k, v in raw.items()}
    return None


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
        self._last_result: ExperimentResult | None = None

    @property
    def latest_weights(self) -> dict[str, dict[str, float]]:
        """Live target weights fitted on the trailing window up to the final observation."""
        return {k: dict(v) for k, v in self._latest_weights.items()}

    @property
    def last_result(self) -> ExperimentResult | None:
        """Immutable :class:`ExperimentResult` from the most recent evaluation run."""
        return self._last_result

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
            if outer_kwargs.get("freq") is not None and hasattr(outer_cv, "split"):
                try:
                    splits = list(outer_cv.split(returns))
                    if splits:
                        return max(1, min(len(train_idx) for train_idx, _ in splits))
                except Exception:  # noqa: BLE001, S110
                    pass
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

    @overload
    def run_robustness_test(
        self,
        cv_type: str = "walk_forward",
        inner_cv_kwargs: dict | None = None,
        *,
        return_result: Literal[False] = False,
        **cv_kwargs: Any,
    ) -> Population: ...

    @overload
    def run_robustness_test(
        self,
        cv_type: str = "walk_forward",
        inner_cv_kwargs: dict | None = None,
        *,
        return_result: Literal[True],
        **cv_kwargs: Any,
    ) -> ExperimentResult: ...

    def run_robustness_test(
        self,
        cv_type: str = "walk_forward",
        inner_cv_kwargs: dict | None = None,
        *,
        return_result: bool = False,
        **cv_kwargs: Any,
    ) -> Population | ExperimentResult:
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
        return_result : bool, default False
            If ``True``, return an immutable :class:`ExperimentResult` containing
            both the OOS ``Population`` and ``live_weights``. If ``False``,
            return ``Population`` directly (while also recording the result on
            :attr:`last_result`).
        **cv_kwargs
            Additional keyword arguments forwarded to the underlying
            cross-validator constructor (e.g., ``train_size``, ``test_size``).

        Returns
        -------
        Population or ExperimentResult
            The combined out-of-sample portfolio population (or immutable
            :class:`ExperimentResult` when ``return_result=True``).

        Raises
        ------
        ValueError
            If ``cv_type`` is not one of the supported strings, or if
            ``universe.returns`` is unavailable.
        """
        self._latest_weights = {}
        self._last_result = None

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
            if cv_type == "combinatorial" and inner_type == "walk_forward":
                inner_cv: Any = _GapAwareWalkForward(
                    reference_index=returns.index, **resolved_explicit_inner
                )
            else:
                inner_cv = self._resolve_cv_splitter(
                    inner_type, **resolved_explicit_inner
                )
        else:
            if cv_type == "walk_forward":
                resolved_inner_kwargs = dict(cv_kwargs)
                inner_train, inner_test = self._derive_inner_walk_forward_sizes(
                    outer_train, cv_kwargs.get("purged_size", 0)
                )
                resolved_inner_kwargs["train_size"] = inner_train
                resolved_inner_kwargs["test_size"] = inner_test
                if cv_kwargs.get("freq") is not None:
                    # inner_train and inner_test are observation row counts, not calendar periods
                    resolved_inner_kwargs.pop("freq", None)

                inner_cv = self._resolve_cv_splitter(
                    "walk_forward", **resolved_inner_kwargs
                )
            else:
                # CombinatorialPurgedCV and MultipleRandomizedCV are multi-path splitters
                # incompatible with GridSearchCV. Restrict inner tuning to standard single-test-fold WalkForward.
                if outer_train <= 2:
                    inner_train, inner_test = 1, 1
                else:
                    inner_train, inner_test = self._derive_inner_walk_forward_sizes(
                        outer_train, 0
                    )
                if cv_type == "combinatorial":
                    inner_cv = _GapAwareWalkForward(
                        train_size=inner_train,
                        test_size=inner_test,
                        reference_index=returns.index,
                    )
                else:
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
            user_previous_weights = copy.deepcopy(
                getattr(target_opt, "previous_weights", None)
            )
            has_friction = _has_rebalancing_friction(target_opt)

            # Prevent present-day physical weights from contaminating historical
            # Fold 0 under BOTH max_turnover and transaction_costs.
            entry_params: dict[str, Any] | None = None
            if has_friction or user_previous_weights is not None:
                if cv_type == "walk_forward":
                    set_previous_weights_chain(est_clone, None)
                    entry_params = _build_entry_rebalancing_params(
                        est_clone, prefix="estimator__"
                    )
                else:
                    baseline_pw = (
                        equal_weights
                        if getattr(target_opt, "max_turnover", None) is not None
                        else None
                    )
                    set_previous_weights_chain(est_clone, baseline_pw)

            if (
                cv_type in ("combinatorial", "randomised")
                and getattr(target_opt, "max_turnover", None) is not None
            ):
                logger.warning(
                    "Strategy '%s' configured with max_turnover under non-sequential cv_type='%s'. "
                    "Historical cross-validation evaluates turnover against equal baseline weights; "
                    "live terminal refit enforces max_turnover against current physical weights.",
                    name,
                    cv_type,
                )

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
            cv_predict_kwargs: dict[str, Any] = {
                "cv": outer_cv,
                "n_jobs": self._n_jobs,
                "portfolio_params": {"tag": name},
            }
            if entry_params is not None:
                cv_predict_kwargs["entry_rebalancing_params"] = entry_params

            portfolio = cross_val_predict(
                search,
                returns,
                **cv_predict_kwargs,
            )

            # c. Live terminal refit on trailing outer_train observations
            try:
                live_search = clone(search)
                live_window = returns.iloc[-min(outer_train, len(returns)) :]
                last_oos_weights = _extract_terminal_oos_weights(portfolio)

                if has_friction:
                    if user_previous_weights is not None:
                        set_previous_weights_chain(live_search, user_previous_weights)
                    elif last_oos_weights:
                        set_previous_weights_chain(live_search, last_oos_weights)

                live_search.fit(live_window)
                live_w = _extract_weights_dict(live_search, live_window.columns)
                if live_w is not None:
                    self._latest_weights[name] = live_w
            except Exception:
                logger.warning(
                    "Live terminal refit failed for strategy '%s'", name, exc_info=True
                )

            if type(portfolio).__name__ == "Population" or isinstance(portfolio, list):
                collected.extend(portfolio)
            else:
                collected.append(portfolio)

        return self._finalize_run(collected, return_result=return_result)

    def _finalize_run(
        self, collected: list[Any], *, return_result: bool
    ) -> Population | ExperimentResult:
        """Assemble Population, record immutable ExperimentResult, and return requested view."""
        pop = Population(collected)
        try:
            pop.latest_weights_by_tag = self.latest_weights
        except AttributeError:
            pass
        result = ExperimentResult(population=pop, live_weights=self.latest_weights)
        self._last_result = result
        return result if return_result else pop

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

    @overload
    def run_online_evaluation(
        self,
        cv_type: str = "walk_forward",
        burn_in_size: int | None = None,
        *,
        return_result: Literal[False] = False,
        **cv_kwargs: Any,
    ) -> Population: ...

    @overload
    def run_online_evaluation(
        self,
        cv_type: str = "walk_forward",
        burn_in_size: int | None = None,
        *,
        return_result: Literal[True],
        **cv_kwargs: Any,
    ) -> ExperimentResult: ...

    def run_online_evaluation(
        self,
        cv_type: str = "walk_forward",
        burn_in_size: int | None = None,
        *,
        return_result: bool = False,
        **cv_kwargs: Any,
    ) -> Population | ExperimentResult:
        """Run online (streaming) evaluation of all registered strategies.

        Mirrors run_robustness_test() but fits hyperparameter search on a
        dedicated historical burn-in period prior to executing online prediction,
        preventing look-ahead bias. Live terminal weights are obtained by
        continuing streaming ``partial_fit`` across the post-burn-in observations
        rather than truncating history via batch ``fit()``.

        Parameters
        ----------
        cv_type : str, optional
            Cross-validation strategy. One of: "walk_forward",
            "combinatorial", "randomised". Default is "walk_forward".
        burn_in_size : int or None, optional
            Number of initial observations allocated for hyperparameter tuning.
            If None, defaults to the train_size or warmup_size specified in
            cv_kwargs, or 252.
        return_result : bool, default False
            If ``True``, return an immutable :class:`ExperimentResult` pairing
            the OOS ``Population`` with ``live_weights``.
        **cv_kwargs
            Additional keyword arguments forwarded to online hyperparameter
            search or online prediction parameters.

        Returns
        -------
        skfolio.Population or ExperimentResult
            A Population object (or :class:`ExperimentResult` when
            ``return_result=True``) containing one MultiPeriodPortfolio per
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
        self._latest_weights = {}
        self._last_result = None

        if not self._strategies:
            raise RuntimeError("No strategies registered. Call add_strategy() first.")

        if cv_type not in {"walk_forward", "combinatorial", "randomised"}:
            raise ValueError(
                f"Unknown cv_type: '{cv_type}'. Must be one of "
                "'walk_forward', 'combinatorial', or 'randomised'."
            )

        if cv_type != "walk_forward":
            logger.warning(
                "run_online_evaluation only supports sequential walk-forward evaluation; ignoring cv_type='%s'.",
                cv_type,
            )

        returns = self._universe.returns

        if burn_in_size is None:
            burn_in_size = int(
                cv_kwargs.get("warmup_size", cv_kwargs.get("train_size", 252))
            )

        if burn_in_size >= len(returns):
            raise ValueError(
                f"burn_in_size={burn_in_size} must be strictly less than total "
                f"observations ({len(returns)}) to allow out-of-sample evaluation."
            )

        burn_in_returns = returns.iloc[:burn_in_size]
        post_burn_in_returns = returns.iloc[burn_in_size:]

        purged = int(cv_kwargs.get("purged_size", 0))
        if (
            "train_size" in cv_kwargs
            and "test_size" in cv_kwargs
            and int(cv_kwargs["train_size"]) + int(cv_kwargs["test_size"]) + purged
            <= burn_in_size
        ):
            inner_warmup = int(cv_kwargs["train_size"])
            inner_test = int(cv_kwargs["test_size"])
        elif burn_in_size - max(0, purged) <= 2:
            inner_warmup, inner_test = 1, 1
        else:
            inner_warmup, inner_test = self._derive_inner_walk_forward_sizes(
                burn_in_size, purged
            )

        online_search_keys = {
            "freq",
            "freq_offset",
            "previous",
            "reduce_test",
        }
        search_extra_kwargs = {
            k: v for k, v in cv_kwargs.items() if k in online_search_keys
        }

        online_predict_keys = {
            "test_size",
            "freq",
            "freq_offset",
            "previous",
            "purged_size",
            "reduce_test",
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

            user_previous_weights = copy.deepcopy(
                getattr(est, "previous_weights", None)
            )
            has_friction = _has_rebalancing_friction(est)
            entry_params: dict[str, Any] | None = None

            if has_friction or user_previous_weights is not None:
                baseline_pw = (
                    equal_weights
                    if getattr(est, "max_turnover", None) is not None
                    else None
                )
                set_previous_weights_chain(est, baseline_pw)
                entry_params = _build_entry_rebalancing_params(est, prefix="")

            search = OnlineGridSearch(
                estimator=est,
                param_grid=grid,
                scoring=RatioMeasure.CVAR_RATIO,
                warmup_size=inner_warmup,
                test_size=inner_test,
                purged_size=purged,
                refit=True,
                n_jobs=1,
                entry_rebalancing_params=entry_params,
                **search_extra_kwargs,
            )
            search.fit(burn_in_returns)
            best_model = search.best_estimator_

            portfolio = online_predict(
                best_model,
                returns,
                warmup_size=burn_in_size,
                portfolio_params={"tag": name},
                entry_rebalancing_params=entry_params,
                **predict_kwargs,
            )

            term_w = _extract_terminal_oos_weights(portfolio)

            try:
                live_model = copy.deepcopy(best_model)
                if has_friction:
                    if user_previous_weights is not None:
                        set_previous_weights_chain(live_model, user_previous_weights)
                    elif term_w is not None:
                        set_previous_weights_chain(live_model, term_w)

                if len(post_burn_in_returns) > 0:
                    live_model.partial_fit(post_burn_in_returns)
                live_w = _extract_weights_dict(live_model, returns.columns)
                if live_w is not None:
                    self._latest_weights[name] = live_w
                elif term_w is not None:
                    self._latest_weights[name] = dict(term_w)
            except Exception:
                logger.warning(
                    "Streaming live partial_fit failed for strategy '%s'",
                    name,
                    exc_info=True,
                )
                if term_w is not None:
                    self._latest_weights[name] = dict(term_w)

            if type(portfolio).__name__ == "Population" or isinstance(portfolio, list):
                collected.extend(portfolio)
            else:
                collected.append(portfolio)

        return self._finalize_run(collected, return_result=return_result)
