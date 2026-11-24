"""Benchmarking harness module.

This module provides the :class:`PortfolioExperimentEngine`, which automates
hyperparameter tuning and walk-forward cross-validation simulation across
multiple investment strategies using ``skfolio``.
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass
from typing import Any, Literal, TypedDict, overload

import numpy as np
import pandas as pd
from skfolio import Population
from skfolio.measures import PerfMeasure, RatioMeasure, RiskMeasure
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

from flowportfolio.core.constraints import (
    _resolve_periods_per_year,
    _validate_holding_period,
)
from flowportfolio.core.optimisation import (
    SkfolioGridSearchCV as GridSearchCV,
)
from flowportfolio.core.optimisation import (
    _collect_param_grid_neutral_params,
    _has_max_turnover_chain,
    _param_grid_sets_max_turnover,
    _param_grid_sets_positive_tc,
    set_previous_weights_chain,
)
from flowportfolio.core.protocols import UniverseProtocol

logger = logging.getLogger(__name__)

_ONLINE_RISK_DENOM_EPS: float = 1e-12
_ONLINE_RATIO_RISK_FLOOR: float = 1e-4
_CALENDAR_FREQ_KWARGS: tuple[str, ...] = (
    "freq",
    "freq_offset",
    "previous",
    "reduce_test",
)
_ONLINE_SAFE_SCORING: dict[str, Any] = {
    "cvar_ratio": RatioMeasure.CVAR_RATIO,
    "sharpe_ratio": RatioMeasure.SHARPE_RATIO,
    "mean": PerfMeasure.MEAN,
    "std": RiskMeasure.STANDARD_DEVIATION,
    "cvar": RiskMeasure.CVAR,
}


def _online_safe_refit(cv_results: dict[str, Any]) -> int:
    """Select the best OnlineGridSearch candidate with ratio-unit fallback for CVaR <= 0."""
    cvar_ratios = np.asarray(cv_results["mean_score_cvar_ratio"], dtype=float)
    sharpe_ratios = np.asarray(cv_results["mean_score_sharpe_ratio"], dtype=float)
    neg_cvars = np.asarray(cv_results["mean_score_cvar"], dtype=float)
    means = np.asarray(cv_results["mean_score_mean"], dtype=float)
    neg_stds = np.asarray(cv_results["mean_score_std"], dtype=float)

    scores = np.full(len(cvar_ratios), -np.inf, dtype=float)
    for idx in range(len(cvar_ratios)):
        raw_cvar = (
            -float(neg_cvars[idx]) if math.isfinite(float(neg_cvars[idx])) else 0.0
        )
        raw_ratio = float(cvar_ratios[idx])
        if raw_cvar > _ONLINE_RISK_DENOM_EPS and math.isfinite(raw_ratio):
            scores[idx] = raw_ratio
        else:
            raw_mean = float(means[idx]) if math.isfinite(float(means[idx])) else None
            raw_std = (
                max(-float(neg_stds[idx]), 0.0)
                if math.isfinite(float(neg_stds[idx]))
                else 0.0
            )
            raw_sharpe = float(sharpe_ratios[idx])
            if raw_mean is not None:
                excess_mean = (
                    raw_sharpe * raw_std
                    if (raw_std > _ONLINE_RISK_DENOM_EPS and math.isfinite(raw_sharpe))
                    else raw_mean
                )
                scores[idx] = excess_mean / max(raw_std, _ONLINE_RATIO_RISK_FLOOR)
    return int(np.argmax(scores))


class StrategyConfig(TypedDict):
    estimator: BaseEstimator
    grid: dict[str, Any]
    holding_period: float | None
    tc_from_spec: bool


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


def _has_chain_transaction_costs(opt: Any) -> bool:
    """Return True if ``opt`` or any fallback configures strictly positive transaction costs."""
    current = opt
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
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
    est: Any,
    *,
    prefix: str = "estimator__",
    param_grid: Any = None,
    include_refit_overrides: bool = False,
) -> dict[str, Any] | None:
    """Build ``skfolio`` ``entry_rebalancing_params`` neutralizing Fold-0 entry friction.

    Inspects both the base estimator chain and any friction parameters
    (``max_turnover``, ``transaction_costs``) swept in ``param_grid``. When
    ``include_refit_overrides=True``, also populates ``refit_param_overrides``
    so :class:`SkfolioGridSearchCV` preserves Fold-0 neutralisation after
    applying ``best_params_``.
    """
    est_neutral: dict[str, Any] = {}
    if hasattr(est, "steps") and est.steps:
        for step_name, step in est.steps:
            _collect_chain_neutral_params(step, f"{step_name}__", est_neutral)
    else:
        _collect_chain_neutral_params(est, "", est_neutral)

    grid_neutral = _collect_param_grid_neutral_params(param_grid)
    if grid_neutral:
        valid_params = (
            set(est.get_params(deep=True).keys())
            if hasattr(est, "get_params")
            else None
        )
        for key, value in grid_neutral.items():
            if valid_params is None or key in valid_params:
                est_neutral[key] = value

    if not est_neutral:
        return None

    params: dict[str, Any] = {f"{prefix}{k}": v for k, v in est_neutral.items()}
    if include_refit_overrides:
        params["refit_param_overrides"] = dict(est_neutral)
    return params


def _scale_transaction_costs_chain(
    opt: Any, divisor: float, _visited: set[int] | None = None
) -> None:
    """Divide ``transaction_costs`` by ``divisor`` across a Pipeline, optimizer, and fallback chain."""
    if opt is None or divisor <= 0.0:
        return
    visited = _visited if _visited is not None else set()
    if id(opt) in visited:
        return
    visited.add(id(opt))

    if hasattr(opt, "steps") and opt.steps:
        for _step_name, step in opt.steps:
            _scale_transaction_costs_chain(step, divisor, _visited=visited)
        return

    tc = getattr(opt, "transaction_costs", None)
    if tc is not None:
        if isinstance(tc, (int, float)) and not isinstance(tc, bool):
            opt.transaction_costs = float(tc) / divisor
        elif isinstance(tc, dict):
            opt.transaction_costs = {k: float(v) / divisor for k, v in tc.items()}
        else:
            try:
                opt.transaction_costs = np.asarray(tc, dtype=float) / divisor
            except (TypeError, ValueError):
                pass

    fb = getattr(opt, "fallback", None)
    if fb is not None:
        _scale_transaction_costs_chain(fb, divisor, _visited=visited)


def _reconcile_transaction_costs_horizon(
    name: str,
    est_clone: Any,
    target_opt: Any,
    *,
    holding_period: float | None,
    tc_from_spec: bool,
    outer_test_size: float,
    inner_test_size: float | None = None,
) -> None:
    """Reconcile per-observation skfolio transaction costs with outer and inner CV fold horizons."""
    if not _has_chain_transaction_costs(target_opt):
        return

    effective_hp: float | None = None
    if outer_test_size > 1.0:
        if holding_period is None and tc_from_spec:
            _scale_transaction_costs_chain(est_clone, float(outer_test_size))
            effective_hp = float(outer_test_size)
            logger.info(
                "Strategy '%s': amortized unamortized ConstraintBuilder transaction_costs "
                "over outer evaluation horizon test_size=%g so cumulative fold cost equals "
                "cost * turnover.",
                name,
                outer_test_size,
            )
        elif holding_period is not None:
            effective_hp = float(holding_period)
            rel_diff = abs(holding_period - outer_test_size) / max(
                holding_period, outer_test_size
            )
            if rel_diff > 0.25:
                logger.warning(
                    "Strategy '%s': configured transaction_costs holding_period=%g differs "
                    "from outer evaluation test_size=%g; skfolio charges transaction_costs "
                    "per observation, scaling cumulative fold cost by %.2fx.",
                    name,
                    holding_period,
                    outer_test_size,
                    outer_test_size / holding_period,
                )
        else:
            logger.warning(
                "Strategy '%s': raw estimator transaction_costs configured without an "
                "explicit holding_period while outer evaluation test_size=%g; skfolio "
                "charges transaction_costs per observation. Pass holding_period via "
                "ConstraintBuilder or add_strategy() to avoid over-charging.",
                name,
                outer_test_size,
            )
    elif holding_period is not None:
        effective_hp = float(holding_period)
    elif tc_from_spec:
        effective_hp = 1.0

    if (
        inner_test_size is not None
        and inner_test_size >= 1.0
        and effective_hp is not None
        and effective_hp > 0.0
    ):
        inner_rel_diff = abs(inner_test_size - effective_hp) / max(
            inner_test_size, effective_hp
        )
        if inner_rel_diff > 0.25:
            logger.warning(
                "Strategy '%s': inner CV test_size=%g diverges from effective "
                "transaction_costs holding horizon=%g; inner hyperparameter tuning "
                "scales cumulative fold cost by %.2fx relative to outer evaluation.",
                name,
                inner_test_size,
                effective_hp,
                inner_test_size / effective_hp,
            )


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
    """Extract drifted terminal OOS weights (``ending_weights_dict``) from a Portfolio or Population."""
    ports_list = (
        list(portfolio)
        if type(portfolio).__name__ == "Population" or isinstance(portfolio, list)
        else [portfolio]
    )
    if not ports_list:
        return None
    last_p = ports_list[-1]
    terminal = (
        last_p.portfolios[-1]
        if hasattr(last_p, "portfolios") and last_p.portfolios
        else last_p
    )
    raw = getattr(terminal, "ending_weights_dict", None)
    if not isinstance(raw, dict):
        raw = getattr(terminal, "weights_dict", None)
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
        *,
        holding_period: float | None = None,
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
        holding_period : float | None, optional
            Explicit holding horizon (in return periods) already used to
            amortize ``transaction_costs`` on ``estimator``. When omitted,
            inherits metadata from ``StrategyBuilder`` / ``ConstraintSpec`` if
            present.

        Raises
        ------
        TypeError
            If ``name`` is not a string, ``estimator`` is not a ``BaseEstimator``,
            ``grid`` is not a dictionary, or ``holding_period`` is not numeric.
        ValueError
            If ``name`` is empty, if the strategy name is already registered,
            or if ``holding_period`` is not strictly positive and finite.
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

        resolved_hp: float | None
        if holding_period is not None:
            resolved_hp = _validate_holding_period(holding_period)
        else:
            raw_hp = getattr(estimator, "_flowportfolio_holding_period", None)
            resolved_hp = (
                _validate_holding_period(raw_hp) if raw_hp is not None else None
            )
        tc_from_spec = bool(getattr(estimator, "_flowportfolio_tc_from_spec", False))

        self._strategies[name] = {
            "estimator": clone(estimator),
            "grid": dict(grid),
            "holding_period": resolved_hp,
            "tc_from_spec": tc_from_spec,
        }

    @staticmethod
    def _resolve_outer_test_size(
        cv_type: str,
        outer_cv: Any,
        resolved_outer_kwargs: dict[str, Any],
        returns: pd.DataFrame,
    ) -> float:
        """Resolve effective outer test fold length (in return observations)."""
        if (
            cv_type == "walk_forward"
            and resolved_outer_kwargs.get("freq") is None
            and "test_size" in resolved_outer_kwargs
        ):
            ts = float(resolved_outer_kwargs["test_size"])
            if math.isfinite(ts) and ts >= 1.0:
                return ts
        try:
            test_lens: list[int] = []
            for _train_idx, test_idx in outer_cv.split(returns):
                if isinstance(test_idx, list):
                    test_lens.extend(len(t) for t in test_idx if len(t) > 0)
                elif len(test_idx) > 0:
                    test_lens.append(len(test_idx))
            if test_lens:
                return float(np.median(test_lens))
        except (ValueError, TypeError, AttributeError) as exc:
            logger.debug("Could not infer outer test size from splitter: %s", exc)
        return 1.0

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
        if hasattr(outer_cv, "split"):
            try:
                splits = list(outer_cv.split(returns))
                if splits:
                    return max(1, min(len(split[0]) for split in splits))
            except Exception:  # noqa: BLE001, S110
                pass
        train_size = getattr(base_cv, "train_size", None)
        if isinstance(train_size, int) and getattr(base_cv, "freq", None) is None:
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
        # 1. Resolve CV protocols
        resolved_outer_kwargs = dict(cv_kwargs)
        if cv_type == "walk_forward":
            resolved_outer_kwargs.setdefault("train_size", 252)
            resolved_outer_kwargs.setdefault("test_size", 63)
        outer_cv = self._resolve_cv_splitter(cv_type, **resolved_outer_kwargs)

        # 2. Extract returns (will raise ValueError if not fetched)
        returns = self._universe.returns
        annualization_factor = _resolve_periods_per_year(self._universe)

        # Determine the size of the outer training window, which bounds the data
        # available to the inner hyperparameter search.
        outer_train = self._resolve_outer_train_size(
            cv_type, outer_cv, resolved_outer_kwargs, len(returns), returns
        )
        outer_test_size = self._resolve_outer_test_size(
            cv_type, outer_cv, resolved_outer_kwargs, returns
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
                # Derived inner sizes are observation counts, not calendar periods
                for freq_key in _CALENDAR_FREQ_KWARGS:
                    resolved_explicit_inner.pop(freq_key, None)
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
                    for freq_key in _CALENDAR_FREQ_KWARGS:
                        resolved_inner_kwargs.pop(freq_key, None)

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

        inner_test_size: float | None = None
        raw_inner_ts = getattr(inner_cv, "test_size", None)
        if (
            isinstance(raw_inner_ts, (int, float))
            and not isinstance(raw_inner_ts, bool)
            and getattr(inner_cv, "freq", None) is None
        ):
            inner_test_size = float(raw_inner_ts)
        else:
            inner_test_size = self._resolve_outer_test_size(
                "walk_forward", inner_cv, {}, returns
            )

        # 3. Process each strategy using run-local accumulators (concurrency safe)
        if returns.empty or returns.shape[1] == 0:
            raise ValueError(
                "Universe returns must be non-empty with at least one asset."
            )
        collected: list[Any] = []
        run_latest_weights: dict[str, dict[str, float]] = {}
        equal_weights = {col: 1.0 / len(returns.columns) for col in returns.columns}
        for name, config in self._strategies.items():
            est_clone = clone(config["estimator"])
            target_opt = (
                est_clone.steps[-1][1]
                if hasattr(est_clone, "steps") and est_clone.steps
                else est_clone
            )
            _reconcile_transaction_costs_horizon(
                name,
                est_clone,
                target_opt,
                holding_period=config["holding_period"],
                tc_from_spec=config["tc_from_spec"],
                outer_test_size=outer_test_size,
                inner_test_size=inner_test_size,
            )

            user_previous_weights = copy.deepcopy(
                getattr(target_opt, "previous_weights", None)
            )
            has_max_turnover = _has_max_turnover_chain(
                est_clone
            ) or _param_grid_sets_max_turnover(config["grid"])
            has_tc = _has_chain_transaction_costs(
                target_opt
            ) or _param_grid_sets_positive_tc(config["grid"])
            has_friction = has_max_turnover or has_tc

            # Prevent present-day physical weights from contaminating historical
            # Fold 0 under BOTH max_turnover and transaction_costs (including grid-swept friction).
            entry_params: dict[str, Any] | None = None
            if has_friction or user_previous_weights is not None:
                if cv_type in ("walk_forward", "randomised"):
                    set_previous_weights_chain(est_clone, None)
                    entry_params = _build_entry_rebalancing_params(
                        est_clone,
                        prefix="estimator__",
                        param_grid=config["grid"],
                        include_refit_overrides=True,
                    )
                else:
                    baseline_pw = equal_weights if has_max_turnover else None
                    set_previous_weights_chain(est_clone, baseline_pw)

            if cv_type == "combinatorial":
                if has_max_turnover:
                    logger.warning(
                        "Strategy '%s' configured with max_turnover under non-sequential cv_type='combinatorial'. "
                        "Historical cross-validation evaluates turnover against equal baseline weights; "
                        "live terminal refit enforces max_turnover against current physical weights.",
                        name,
                    )
                if has_tc:
                    logger.warning(
                        "Strategy '%s' configured with transaction_costs under non-sequential "
                        "cv_type='combinatorial': independent CPCV folds enter from zero weights "
                        "and pay full entry turnover cost on every fold. Use cv_type='walk_forward' "
                        "for sequential cost simulation.",
                        name,
                    )
            elif cv_type == "randomised" and has_max_turnover:
                logger.warning(
                    "Strategy '%s' configured with max_turnover under cv_type='randomised'. "
                    "Historical subsampled paths start with neutralized Fold-0 entry rebalancing; "
                    "live terminal refit enforces max_turnover against current physical weights.",
                    name,
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
            # skfolio disables parallel execution with a UserWarning whenever
            # previous_weights / friction / entry_rebalancing_params are active.
            effective_n_jobs = (
                None if (has_friction or entry_params is not None) else self._n_jobs
            )
            cv_predict_kwargs: dict[str, Any] = {
                "cv": outer_cv,
                "n_jobs": effective_n_jobs,
                "portfolio_params": {
                    "tag": name,
                    "annualization_factor": annualization_factor,
                },
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
                    run_latest_weights[name] = live_w
            except Exception:
                logger.warning(
                    "Live terminal refit failed for strategy '%s'", name, exc_info=True
                )

            if type(portfolio).__name__ == "Population" or isinstance(portfolio, list):
                collected.extend(portfolio)
            else:
                collected.append(portfolio)

        return self._finalize_run(
            collected, run_latest_weights, return_result=return_result
        )

    def _finalize_run(
        self,
        collected: list[Any],
        run_latest_weights: dict[str, dict[str, float]],
        *,
        return_result: bool,
    ) -> Population | ExperimentResult:
        """Assemble Population, record immutable ExperimentResult, and return requested view."""
        live_copy = {k: dict(v) for k, v in run_latest_weights.items()}
        pop = Population(collected)
        try:
            pop.latest_weights_by_tag = {k: dict(v) for k, v in live_copy.items()}
        except AttributeError:
            pass
        result = ExperimentResult(population=pop, live_weights=live_copy)
        self._latest_weights = live_copy
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

    @staticmethod
    def _resolve_online_schedule(
        returns: pd.DataFrame,
        burn_in_size: int | None,
        cv_kwargs: dict[str, Any],
        predict_kwargs: dict[str, Any],
    ) -> tuple[int, int, float]:
        """Resolve ``(burn_in_rows, predict_warmup, online_test_size)`` across row/freq modes."""
        if burn_in_size is not None and (
            isinstance(burn_in_size, bool)
            or not isinstance(burn_in_size, int)
            or burn_in_size < 1
        ):
            raise ValueError(
                f"burn_in_size must be a positive integer, got {burn_in_size!r}."
            )
        freq = predict_kwargs.get("freq")
        if freq is None:
            resolved_burn_in = (
                int(burn_in_size)
                if burn_in_size is not None
                else int(cv_kwargs.get("warmup_size", cv_kwargs.get("train_size", 252)))
            )
            if resolved_burn_in < 1:
                raise ValueError(
                    f"burn_in_size must be a positive integer, got {resolved_burn_in}."
                )
            if resolved_burn_in >= len(returns):
                raise ValueError(
                    f"burn_in_size={resolved_burn_in} must be strictly less than total "
                    f"observations ({len(returns)}) to allow out-of-sample evaluation."
                )
            return (
                resolved_burn_in,
                resolved_burn_in,
                float(predict_kwargs.get("test_size", 1)),
            )

        wf_args = {
            k: predict_kwargs[k]
            for k in ("freq", "freq_offset", "previous", "purged_size", "reduce_test")
            if k in predict_kwargs
        }
        pred_test = int(predict_kwargs.get("test_size", 1))

        if burn_in_size is None and "warmup_size" in cv_kwargs:
            predict_warmup = int(cv_kwargs["warmup_size"])
            splits = list(
                WalkForward(
                    train_size=predict_warmup, test_size=pred_test, **wf_args
                ).split(returns)
            )
            if not splits:
                raise ValueError(
                    f"warmup_size={predict_warmup} with freq={freq!r} produced no test splits."
                )
            resolved_burn_in = int(splits[0][0][-1]) + 1
        else:
            resolved_burn_in = (
                int(burn_in_size)
                if burn_in_size is not None
                else int(cv_kwargs.get("train_size", 252))
            )
            if resolved_burn_in < 1:
                raise ValueError(
                    f"burn_in_size must be a positive integer, got {resolved_burn_in}."
                )
            if resolved_burn_in >= len(returns):
                raise ValueError(
                    f"burn_in_size={resolved_burn_in} must be strictly less than total "
                    f"observations ({len(returns)}) to allow out-of-sample evaluation."
                )
            step_splits = list(
                WalkForward(
                    train_size=1,
                    test_size=1,
                    freq=wf_args["freq"],
                    freq_offset=wf_args.get("freq_offset"),
                    previous=bool(wf_args.get("previous", False)),
                ).split(returns)
            )
            max_periods = max(1, len(step_splits))
            lo, hi = 1, max_periods
            best_k: int | None = None
            splits = []
            while lo <= hi:
                mid = (lo + hi) // 2
                try:
                    cand_splits = list(
                        WalkForward(
                            train_size=mid, test_size=pred_test, **wf_args
                        ).split(returns)
                    )
                except ValueError:
                    cand_splits = []
                if not cand_splits:
                    hi = mid - 1
                    continue
                if int(cand_splits[0][1][0]) >= resolved_burn_in:
                    best_k = mid
                    splits = cand_splits
                    hi = mid - 1
                else:
                    lo = mid + 1
            if best_k is None or not splits:
                raise ValueError(
                    f"burn_in_size={resolved_burn_in} leaves no calendar '{freq}' "
                    "test periods in returns for online evaluation."
                )
            predict_warmup = best_k

        test_lens = [len(te) for _tr, te in splits if len(te) > 0]
        online_test_size = float(np.median(test_lens)) if test_lens else 1.0
        return resolved_burn_in, predict_warmup, online_test_size

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
        if returns.empty or returns.shape[1] == 0:
            raise ValueError(
                "Universe returns must be non-empty with at least one asset."
            )
        annualization_factor = _resolve_periods_per_year(self._universe)

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

        burn_in_rows, predict_warmup, online_test_size = self._resolve_online_schedule(
            returns, burn_in_size, cv_kwargs, predict_kwargs
        )

        burn_in_returns = returns.iloc[:burn_in_rows]
        post_burn_in_returns = returns.iloc[burn_in_rows:]

        purged = int(cv_kwargs.get("purged_size", 0))
        if (
            predict_kwargs.get("freq") is None
            and "train_size" in cv_kwargs
            and "test_size" in cv_kwargs
            and int(cv_kwargs["train_size"]) + int(cv_kwargs["test_size"]) + purged
            <= burn_in_rows
        ):
            inner_warmup = int(cv_kwargs["train_size"])
            inner_test = int(cv_kwargs["test_size"])
        elif burn_in_rows - max(0, purged) <= 2:
            inner_warmup, inner_test = 1, 1
        else:
            inner_warmup, inner_test = self._derive_inner_walk_forward_sizes(
                burn_in_rows, purged
            )

        collected: list[Any] = []
        run_latest_weights: dict[str, dict[str, float]] = {}
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
            _reconcile_transaction_costs_horizon(
                name,
                est,
                est,
                holding_period=config["holding_period"],
                tc_from_spec=config["tc_from_spec"],
                outer_test_size=online_test_size,
                inner_test_size=float(inner_test),
            )

            user_previous_weights = copy.deepcopy(
                getattr(est, "previous_weights", None)
            )
            has_max_turnover = _has_max_turnover_chain(
                est
            ) or _param_grid_sets_max_turnover(grid)
            has_tc = _has_chain_transaction_costs(est) or _param_grid_sets_positive_tc(
                grid
            )
            has_friction = has_max_turnover or has_tc
            entry_params: dict[str, Any] | None = None

            if has_friction or user_previous_weights is not None:
                baseline_pw = equal_weights if has_max_turnover else None
                set_previous_weights_chain(est, baseline_pw)
                entry_params = _build_entry_rebalancing_params(
                    est, prefix="", param_grid=grid
                )

            # inner_warmup and inner_test are observation row counts from burn_in_returns,
            # so calendar freq/freq_offset/previous/reduce_test must NOT leak into OnlineGridSearch.
            search = OnlineGridSearch(
                estimator=est,
                param_grid=grid,
                scoring=_ONLINE_SAFE_SCORING,
                warmup_size=inner_warmup,
                test_size=inner_test,
                purged_size=purged,
                refit=_online_safe_refit,
                n_jobs=1,
                entry_rebalancing_params=entry_params,
            )
            search.fit(burn_in_returns)
            best_model = search.best_estimator_

            portfolio = online_predict(
                best_model,
                returns,
                warmup_size=predict_warmup,
                portfolio_params={
                    "tag": name,
                    "annualization_factor": annualization_factor,
                },
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
                    run_latest_weights[name] = live_w
                elif term_w is not None:
                    run_latest_weights[name] = dict(term_w)
            except Exception:
                logger.warning(
                    "Streaming live partial_fit failed for strategy '%s'",
                    name,
                    exc_info=True,
                )
                if term_w is not None:
                    run_latest_weights[name] = dict(term_w)

            if type(portfolio).__name__ == "Population" or isinstance(portfolio, list):
                collected.extend(portfolio)
            else:
                collected.append(portfolio)

        return self._finalize_run(
            collected, run_latest_weights, return_result=return_result
        )
