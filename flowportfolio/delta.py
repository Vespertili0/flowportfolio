"""Portfolio delta engine module.

This module provides the :class:`PortfolioDeltaEngine` class, which bridges
the gap between an optimised target allocation (from the
``PortfolioExperimentEngine``) and a user's actual physical holdings. It
produces decision-ready trade analysis by quantifying the net risk/return
benefit of executing a portfolio rebalance versus holding current positions,
factoring in group drift, Total Expense Ratio (TER), slippage, and brokerage
friction.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from skfolio import Population
from skfolio.portfolio import Portfolio

from flowportfolio.core.protocols import UniverseProtocol, is_universe_protocol
from flowportfolio.core.validation import validate_weight

logger = logging.getLogger(__name__)

_DUST_TRADE_TOL: float = 1e-6


def _resolve_periods_per_year(universe: object) -> float:
    """Return a positive finite periods_per_year from universe, defaulting to 252.0."""
    raw = getattr(universe, "periods_per_year", 252)
    if (
        isinstance(raw, (int, float))
        and not isinstance(raw, bool)
        and math.isfinite(raw)
        and raw > 0
    ):
        return float(raw)
    return 252.0


def _align_returns_strict(
    universe_returns: pd.DataFrame, port_obj: Any
) -> pd.DataFrame:
    """Strictly align ``universe_returns`` to an OOS portfolio's timestamp index."""
    port_returns = getattr(port_obj, "returns_df", None)
    if port_returns is None:
        port_returns = getattr(port_obj, "returns", None)

    port_index = getattr(port_returns, "index", None)
    if port_index is None or not isinstance(port_index, pd.Index):
        raise ValueError(
            "Portfolio must expose timestamp-indexed returns_df aligned with universe.returns."
        )
    if len(port_index) == 0:
        raise ValueError("Portfolio returns must not be empty.")
    if not bool(port_index.isin(universe_returns.index).all()):
        raise ValueError(
            "Portfolio returns index is not aligned with universe.returns index."
        )
    return universe_returns.loc[port_index]


def _sub_portfolios(port_obj: Any) -> list[Any]:
    """Return constituent single-period folds for a MultiPeriodPortfolio, else ``[port_obj]``."""
    children = getattr(port_obj, "portfolios", None)
    if isinstance(children, list) and len(children) > 0:
        return list(children)
    return [port_obj]


def _has_positive_numeric_attr(obj: Any, attr: str) -> bool:
    """Return True if ``getattr(obj, attr)`` contains any strictly positive numeric value."""
    val = getattr(obj, attr, None)
    if val is None or isinstance(val, bool):
        return False
    if isinstance(val, (int, float, np.number)):
        f = float(val)
        return math.isfinite(f) and abs(f) > 0.0
    if isinstance(val, dict):
        return any(
            isinstance(v, (int, float, np.number))
            and not isinstance(v, bool)
            and math.isfinite(float(v))
            and abs(float(v)) > 0.0
            for v in val.values()
        )
    try:
        arr = np.asarray(val, dtype=float)
        return bool(np.any(np.isfinite(arr) & (np.abs(arr) > 0.0)))
    except (TypeError, ValueError):
        return False


def _sub_portfolio_has_fees(sub: Any) -> bool:
    """Return True if a single-period portfolio already deducts management fees."""
    return _has_positive_numeric_attr(sub, "total_fee") or _has_positive_numeric_attr(
        sub, "management_fees"
    )


def _sub_portfolio_has_costs(sub: Any) -> bool:
    """Return True if a single-period portfolio itself embeds positive transaction costs."""
    return _has_positive_numeric_attr(sub, "total_cost") or _has_positive_numeric_attr(
        sub, "transaction_costs"
    )


def _portfolio_has_embedded_costs(port_obj: Any) -> bool:
    """Return True if an OOS portfolio already carries embedded transaction costs."""
    return any(_sub_portfolio_has_costs(sub) for sub in _sub_portfolios(port_obj))


def _weights_array_to_dict(
    sub: Any, raw_weights: Any, fallback_columns: Sequence[str]
) -> dict[str, float] | None:
    """Align a 1D weights array with ``sub`` asset names or fallback columns."""
    if raw_weights is None:
        return None
    w_arr = np.asarray(raw_weights, dtype=float)
    if w_arr.ndim != 1 or len(w_arr) == 0:
        return None
    assets = getattr(sub, "assets", None)
    if assets is None or len(assets) != len(w_arr):
        assets = getattr(sub, "assets_names", None)
    if assets is None or len(assets) != len(w_arr):
        assets = fallback_columns if len(fallback_columns) == len(w_arr) else None
    if assets is None:
        return None
    return {str(a): float(w) for a, w in zip(assets, w_arr, strict=True)}


def _extract_portfolio_weights_dict(
    sub: Any, fallback_columns: Sequence[str]
) -> dict[str, float] | None:
    """Extract a ``{ticker: weight}`` mapping from a single-period portfolio."""
    raw = getattr(sub, "weights_dict", None)
    if isinstance(raw, dict) and len(raw) > 0:
        return {str(k): float(v) for k, v in raw.items()}
    return _weights_array_to_dict(sub, getattr(sub, "weights", None), fallback_columns)


def _extract_ending_weights_dict(
    sub: Any, fallback_columns: Sequence[str]
) -> dict[str, float] | None:
    """Extract drifted terminal weights from a single-period fold portfolio."""
    raw = getattr(sub, "ending_weights_dict", None)
    if isinstance(raw, dict) and len(raw) > 0:
        return {str(k): float(v) for k, v in raw.items()}
    resolved = _weights_array_to_dict(
        sub, getattr(sub, "ending_weights", None), fallback_columns
    )
    if resolved is not None:
        return resolved
    return _extract_portfolio_weights_dict(sub, fallback_columns)


def _extract_entry_weights(
    port_obj: Any, fallback_columns: Sequence[str]
) -> dict[str, float] | None:
    """Extract Fold-0 initial weights for an OOS portfolio path."""
    subs = _sub_portfolios(port_obj)
    if not subs:
        return None
    return _extract_portfolio_weights_dict(subs[0], fallback_columns)


def _validate_oos_asset_coverage(
    port_obj: Any,
    universe_columns: Sequence[str],
) -> None:
    """Reject asset-subsampled OOS paths (e.g., MultipleRandomizedCV with subset < N)."""
    univ_set = set(universe_columns)
    for sub in _sub_portfolios(port_obj):
        raw_assets = getattr(sub, "assets", None)
        if raw_assets is None:
            raw_assets = getattr(sub, "assets_names", None)
        if raw_assets is None:
            w_dict = getattr(sub, "weights_dict", None)
            if isinstance(w_dict, dict):
                raw_assets = list(w_dict.keys())
        if raw_assets is None:
            continue
        sub_set = {str(a) for a in raw_assets}
        if sub_set and sub_set <= univ_set and sub_set < univ_set:
            raise ValueError(
                "PortfolioDeltaEngine cannot evaluate buy-and-hold vs. rebalance against "
                "asset-subsampled OOS paths (e.g. MultipleRandomizedCV with asset_subset_size < N) "
                "because excluded universe tickers in current_weights create an asymmetric baseline. "
                "Use full-universe walk_forward or combinatorial cross-validation, or pass "
                "allow_asset_subsets=True if evaluating a direct pre-selection Pipeline output."
            )


def _compute_transition_drag_for_weights(
    current_weights: dict[str, float],
    target_weights: dict[str, float],
    *,
    tickers: Sequence[str],
    brokerage_bps: float,
    slippage_bps: float,
    portfolio_value: float | None,
    brokerage_flat: float,
    min_trade_weight: float,
) -> float:
    """Compute one-off portfolio transition drag for ``current_weights -> target_weights``."""
    total_bps_rate = (float(brokerage_bps) + float(slippage_bps)) / 10_000.0
    total_drag = 0.0
    for ticker in tickers:
        gross_traded = abs(
            float(target_weights.get(ticker, 0.0))
            - float(current_weights.get(ticker, 0.0))
        )
        var_cost = gross_traded * total_bps_rate
        flat_cost = (
            float(brokerage_flat) / float(portfolio_value)
            if (
                float(brokerage_flat) > 0.0
                and portfolio_value is not None
                and gross_traded > float(min_trade_weight)
            )
            else 0.0
        )
        total_drag += var_cost + flat_cost
    return total_drag


def _extract_portfolio_eval_settings(
    port_obj: Any, default_annualization: float
) -> tuple[float, float, bool]:
    """Extract ``(risk_free_rate, annualization_factor, compounded)`` from a portfolio."""
    first_sub = (
        port_obj.portfolios[0]
        if hasattr(port_obj, "portfolios") and port_obj.portfolios
        else None
    )

    rfr_raw = getattr(port_obj, "risk_free_rate", None)
    if rfr_raw is None and first_sub is not None:
        rfr_raw = getattr(first_sub, "risk_free_rate", None)
    rfr = (
        float(rfr_raw)
        if isinstance(rfr_raw, (int, float))
        and not isinstance(rfr_raw, bool)
        and math.isfinite(rfr_raw)
        else 0.0
    )

    ann_raw = getattr(port_obj, "annualization_factor", None)
    if ann_raw is None and first_sub is not None:
        ann_raw = getattr(first_sub, "annualization_factor", None)
    ann_factor = (
        float(ann_raw)
        if isinstance(ann_raw, (int, float))
        and not isinstance(ann_raw, bool)
        and math.isfinite(ann_raw)
        and ann_raw > 0
        else float(default_annualization)
    )

    comp_raw = getattr(port_obj, "compounded", None)
    if comp_raw is None and first_sub is not None:
        comp_raw = getattr(first_sub, "compounded", None)
    compounded = bool(comp_raw) if isinstance(comp_raw, bool) else False

    return rfr, ann_factor, compounded


def _extract_portfolio_metrics(port_obj: Any) -> dict[str, float]:
    """Extract ``cvar``, annualized ``sharpe``, and ``max_drawdown`` cleanly."""
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = (
            float(port_obj.annualized_sharpe_ratio)
            if hasattr(port_obj, "annualized_sharpe_ratio")
            else float(port_obj.sharpe_ratio)
        )
        return {
            "cvar": float(port_obj.cvar),
            "sharpe": sharpe,
            "max_drawdown": float(port_obj.max_drawdown),
        }


def _eval_univariate_metrics(
    returns_1d: np.ndarray,
    port_rfr: float,
    ann_factor: float,
    compounded: bool,
) -> dict[str, float]:
    """Evaluate portfolio risk/return metrics on a univariate net return stream."""
    port = Portfolio(
        X=returns_1d[:, None],
        weights=np.array([1.0], dtype=float),
        risk_free_rate=port_rfr,
        annualization_factor=ann_factor,
        compounded=compounded,
    )
    return _extract_portfolio_metrics(port)


def _evaluate_portfolio_with_drag(
    port_obj: Any,
    *,
    transition_drag: float,
    default_annualization: float,
    annual_fees: dict[str, float] | None = None,
    fallback_columns: Sequence[str] = (),
    check_embedded_costs: bool = False,
    rebalance_cost_fn: Callable[[dict[str, float], dict[str, float]], float]
    | None = None,
) -> dict[str, float]:
    """Evaluate portfolio metrics with symmetric fees, per-fold rebalance costs, and steady-state risk."""
    port_rfr, ann_factor, compounded = _extract_portfolio_eval_settings(
        port_obj, default_annualization
    )
    fees_map = annual_fees or {}
    has_universe_fees = any(float(v) > 0.0 for v in fees_map.values())
    subs = _sub_portfolios(port_obj)

    entry_drag = float(transition_drag)
    fold0 = subs[0] if subs else port_obj
    if check_embedded_costs and entry_drag > 0.0 and _sub_portfolio_has_costs(fold0):
        label = getattr(port_obj, "tag", None) or getattr(port_obj, "name", "unnamed")
        logger.warning(
            "Skipping external transition drag (%.6f) for OOS portfolio '%s' because "
            "it already embeds transaction_costs; avoiding double-counting.",
            entry_drag,
            label,
        )
        entry_drag = 0.0

    applied_fees = False
    rebalance_drag = 0.0
    slices: list[np.ndarray] = []
    prev_end_weights: dict[str, float] | None = None

    for idx, sub in enumerate(subs):
        sub_rets = np.asarray(sub.returns, dtype=float).copy()
        if has_universe_fees and not _sub_portfolio_has_fees(sub):
            sub_weights = _extract_portfolio_weights_dict(sub, fallback_columns)
            if sub_weights:
                fee_drag = sum(
                    abs(w) * float(fees_map.get(asset, 0.0)) / float(ann_factor)
                    for asset, w in sub_weights.items()
                )
                if fee_drag > 0.0:
                    sub_rets -= fee_drag
                    applied_fees = True

        if (
            idx >= 1
            and rebalance_cost_fn is not None
            and not _sub_portfolio_has_costs(sub)
            and prev_end_weights is not None
            and len(sub_rets) > 0
        ):
            curr_weights = _extract_portfolio_weights_dict(sub, fallback_columns)
            if curr_weights is not None:
                fold_drag = float(rebalance_cost_fn(prev_end_weights, curr_weights))
                if fold_drag > 0.0:
                    sub_rets -= fold_drag / float(len(sub_rets))
                    rebalance_drag += fold_drag

        prev_end_weights = _extract_ending_weights_dict(sub, fallback_columns)
        slices.append(sub_rets)

    if applied_fees:
        label = getattr(port_obj, "tag", None) or getattr(port_obj, "name", "unnamed")
        logger.warning(
            "Applied symmetric period-scaled universe management fees to fee-free OOS "
            "portfolio '%s' so hold and rebalance scenarios are compared net of fees.",
            label,
        )

    steady_returns = (
        np.concatenate(slices)
        if slices
        else np.asarray(port_obj.returns, dtype=float).copy()
    )
    if len(steady_returns) == 0:
        raise ValueError("Portfolio has an empty return stream.")

    steady_metrics = (
        _extract_portfolio_metrics(port_obj)
        if not applied_fees and rebalance_drag <= 0.0
        else _eval_univariate_metrics(steady_returns, port_rfr, ann_factor, compounded)
    )

    if entry_drag > 0.0:
        amortized_returns = steady_returns - entry_drag / float(len(steady_returns))
        sharpe_val = _eval_univariate_metrics(
            amortized_returns, port_rfr, ann_factor, compounded
        )["sharpe"]
    else:
        sharpe_val = steady_metrics["sharpe"]

    return {
        "cvar": steady_metrics["cvar"],
        "sharpe": sharpe_val,
        "max_drawdown": steady_metrics["max_drawdown"],
        "entry_drag": entry_drag,
        "rebalance_drag": rebalance_drag,
    }


def evaluate_rebalance_benefit(
    universe_returns: pd.DataFrame,
    current_weights: dict[str, float],
    target_weights: dict[str, float],
    *,
    oos_portfolios: Sequence[Any] | None = None,
    annual_fees: dict[str, float] | None = None,
    periods_per_year: float = 252.0,
    transition_drag: float = 0.0,
    transition_cost_fn: Callable[[dict[str, float]], float] | None = None,
    rebalance_cost_fn: Callable[[dict[str, float], dict[str, float]], float]
    | None = None,
    allow_asset_subsets: bool = False,
) -> dict[str, dict[str, float]]:
    """Compare buy-and-hold vs. rebalance performance without look-ahead bias.

    When ``oos_portfolios`` are supplied, the ``"rebalance"`` scenario reflects
    the realized out-of-sample ``Portfolio`` / ``MultiPeriodPortfolio`` return
    stream with symmetric period-scaled management fees (applied only to
    fee-free folds), recurring per-fold rebalance costs across Folds ``1..N``
    when omitted from OOS folds, and horizon-amortized Fold-0 entry drag (charged
    whenever Fold 0 itself has no embedded transaction cost, without distorting
    steady-state daily CVaR). The ``"hold"`` scenario is evaluated over the
    strictly aligned OOS dates using ``weight_drift=True`` and period-scaled
    management fees.
    """
    fees_map = annual_fees or {}
    columns = [str(c) for c in universe_returns.columns]
    period_fees_vector = np.array(
        [
            float(fees_map.get(col, 0.0)) / float(periods_per_year)
            for col in universe_returns.columns
        ],
        dtype=float,
    )
    current_weight_vector = np.array(
        [float(current_weights.get(col, 0.0)) for col in universe_returns.columns],
        dtype=float,
    )
    target_weight_vector = np.array(
        [float(target_weights.get(col, 0.0)) for col in universe_returns.columns],
        dtype=float,
    )

    if oos_portfolios:
        hold_list: list[dict[str, float]] = []
        reb_list: list[dict[str, float]] = []
        for p in oos_portfolios:
            if not allow_asset_subsets:
                _validate_oos_asset_coverage(p, columns)
            path_returns = _align_returns_strict(universe_returns, p)
            port_rfr, ann_factor, compounded = _extract_portfolio_eval_settings(
                p, periods_per_year
            )
            hold_fees_vector = np.array(
                [
                    float(fees_map.get(col, 0.0)) / float(ann_factor)
                    for col in universe_returns.columns
                ],
                dtype=float,
            )
            hold_kwargs: dict[str, Any] = {
                "weight_drift": True,
                "management_fees": hold_fees_vector,
                "annualization_factor": ann_factor,
                "compounded": compounded,
            }
            if port_rfr:
                hold_kwargs["risk_free_rate"] = port_rfr
            hold_port = Portfolio(
                X=path_returns,
                weights=current_weight_vector,
                **hold_kwargs,
            )
            hold_list.append(_extract_portfolio_metrics(hold_port))

            entry_w = _extract_entry_weights(p, columns)
            if rebalance_cost_fn is not None and entry_w is not None:
                path_drag = float(rebalance_cost_fn(current_weights, entry_w))
            elif transition_cost_fn is not None and entry_w is not None:
                path_drag = float(transition_cost_fn(entry_w))
            else:
                path_drag = float(transition_drag)

            reb_list.append(
                _evaluate_portfolio_with_drag(
                    p,
                    transition_drag=path_drag,
                    default_annualization=periods_per_year,
                    annual_fees=fees_map,
                    fallback_columns=columns,
                    check_embedded_costs=True,
                    rebalance_cost_fn=rebalance_cost_fn,
                )
            )

        if len(hold_list) == 1:
            return {"hold": hold_list[0], "rebalance": reb_list[0]}

        return {
            "hold": {
                key: float(np.median([m[key] for m in hold_list]))
                for key in ("cvar", "sharpe", "max_drawdown")
            },
            "rebalance": {
                key: float(np.median([m[key] for m in reb_list]))
                for key in (
                    "cvar",
                    "sharpe",
                    "max_drawdown",
                    "entry_drag",
                    "rebalance_drag",
                )
            },
        }

    # In-sample historical projection fallback (when no OOS population is supplied)
    ptf_kwargs: dict[str, Any] = {
        "weight_drift": True,
        "management_fees": period_fees_vector,
        "annualization_factor": float(periods_per_year),
    }
    hold_portfolio = Portfolio(
        X=universe_returns,
        weights=current_weight_vector,
        **ptf_kwargs,
    )
    rebalance_portfolio = Portfolio(
        X=universe_returns,
        weights=target_weight_vector,
        **ptf_kwargs,
    )
    if rebalance_cost_fn is not None:
        static_drag = float(rebalance_cost_fn(current_weights, target_weights))
    elif transition_cost_fn is not None:
        static_drag = float(transition_cost_fn(target_weights))
    else:
        static_drag = float(transition_drag)
    return {
        "hold": _extract_portfolio_metrics(hold_portfolio),
        "rebalance": _evaluate_portfolio_with_drag(
            rebalance_portfolio,
            transition_drag=static_drag,
            default_annualization=periods_per_year,
        ),
    }


class PortfolioDeltaEngine:
    """Evaluates the decision to rebalance a portfolio against current holdings.

    This engine quantifies the trade-off of executing a rebalance versus
    holding current positions (net of transaction friction and management
    fees).

    Parameters
    ----------
    universe : Universe
        A fully initialised Universe object containing metadata and fees.
    current_weights : dict[str, float]
        Dictionary mapping tickers to their current physical allocation
        weight. Must sum to 1.0 (within 1e-6 tolerance) and all tickers
        must be present in ``universe.tickers``.
    target_weights : dict[str, float]
        Dictionary mapping tickers to the proposed optimised allocation
        weight. Must sum to 1.0 (within 1e-6 tolerance) and all tickers
        must be present in ``universe.tickers``.

    Raises
    ------
    TypeError
        If any argument is not of the expected type.
    ValueError
        If ``current_weights`` or ``target_weights`` do not sum to 1.0
        (within 1e-6 tolerance), or if any ticker key is missing from
        ``universe.tickers``.
    """

    def __init__(
        self,
        universe: UniverseProtocol,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
    ) -> None:
        # --- type validation ---
        if not is_universe_protocol(universe):
            raise TypeError("universe must implement UniverseProtocol.")
        if not isinstance(current_weights, dict):
            raise TypeError("current_weights must be a dict.")
        if not isinstance(target_weights, dict):
            raise TypeError("target_weights must be a dict.")
        if not all(isinstance(k, str) for k in current_weights):
            raise TypeError("current_weights keys must all be strings.")
        if not all(isinstance(k, str) for k in target_weights):
            raise TypeError("target_weights keys must all be strings.")
        if not all(
            isinstance(v, (int, float)) and not isinstance(v, bool)
            for v in current_weights.values()
        ):
            raise TypeError(
                "current_weights values must all be numeric (int or float)."
            )
        if not all(
            isinstance(v, (int, float)) and not isinstance(v, bool)
            for v in target_weights.values()
        ):
            raise TypeError("target_weights values must all be numeric (int or float).")

        # --- finite value validation (guards against NaN / Inf) ---
        if not all(np.isfinite(v) for v in current_weights.values()):
            raise ValueError(
                "current_weights values must be finite numbers (no NaN or Inf)."
            )
        if not all(np.isfinite(v) for v in target_weights.values()):
            raise ValueError(
                "target_weights values must be finite numbers (no NaN or Inf)."
            )

        for v in current_weights.values():
            validate_weight(v)
        for v in target_weights.values():
            validate_weight(v)

        # --- sum-to-one validation ---
        current_sum = sum(current_weights.values())
        if abs(current_sum - 1.0) > 1e-6:
            raise ValueError(
                f"current_weights must sum to 1.0 (within 1e-6 tolerance); "
                f"got {current_sum}."
            )
        target_sum = sum(target_weights.values())
        if abs(target_sum - 1.0) > 1e-6:
            raise ValueError(
                f"target_weights must sum to 1.0 (within 1e-6 tolerance); "
                f"got {target_sum}."
            )

        # --- ticker coverage validation ---
        universe_tickers = set(universe.tickers)

        unknown_current = set(current_weights.keys()) - universe_tickers
        if unknown_current:
            raise ValueError(
                f"current_weights contains tickers not found in universe: "
                f"{sorted(unknown_current)}."
            )
        missing_current = universe_tickers - set(current_weights.keys())
        if missing_current:
            raise ValueError(
                f"current_weights is missing universe tickers: "
                f"{sorted(missing_current)}."
            )

        unknown_target = set(target_weights.keys()) - universe_tickers
        if unknown_target:
            raise ValueError(
                f"target_weights contains tickers not found in universe: "
                f"{sorted(unknown_target)}."
            )
        missing_target = universe_tickers - set(target_weights.keys())
        if missing_target:
            raise ValueError(
                f"target_weights is missing universe tickers: {sorted(missing_target)}."
            )

        self._universe: UniverseProtocol = universe
        self._current_weights: dict[str, float] = dict(current_weights)
        self._target_weights: dict[str, float] = dict(target_weights)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def calculate_group_drift(self) -> pd.DataFrame:
        """Calculate the absolute allocation drift per asset group.

        Maps each ticker's weight to its group via ``universe.metadata`` and
        computes the current allocation, target allocation, drift
        (target - current), and absolute drift per group.

        Returns
        -------
        pd.DataFrame
            A DataFrame indexed by group name with columns:
            ``'current_allocation'``, ``'target_allocation'``, ``'drift'``,
            and ``'abs_drift'``.

        Raises
        ------
        ValueError
            If ``current_weights`` or ``target_weights`` contain tickers not
            present in ``universe.metadata``.
        """
        metadata = self._universe.metadata  # dict[str, str] — ticker -> group

        # Validate all tickers are present in metadata
        missing_current = set(self._current_weights.keys()) - set(metadata.keys())
        if missing_current:
            raise ValueError(
                f"current_weights contains tickers not found in universe.metadata: "
                f"{sorted(missing_current)}."
            )
        missing_target = set(self._target_weights.keys()) - set(metadata.keys())
        if missing_target:
            raise ValueError(
                f"target_weights contains tickers not found in universe.metadata: "
                f"{sorted(missing_target)}."
            )

        # Accumulate per-group allocations
        groups: defaultdict[str, dict[str, float]] = defaultdict(
            lambda: {"current_allocation": 0.0, "target_allocation": 0.0}
        )

        for ticker, weight in self._current_weights.items():
            group = metadata[ticker]
            groups[group]["current_allocation"] += float(weight)

        for ticker, weight in self._target_weights.items():
            group = metadata[ticker]
            groups[group]["target_allocation"] += float(weight)

        # Build result rows
        rows = []
        for group, allocs in groups.items():
            current = allocs["current_allocation"]
            target = allocs["target_allocation"]
            drift = target - current
            rows.append(
                {
                    "group": group,
                    "current_allocation": current,
                    "target_allocation": target,
                    "drift": drift,
                    "abs_drift": abs(drift),
                }
            )

        df = pd.DataFrame(rows).set_index("group")
        return df

    def calculate_rebalance_delta(
        self,
        population: Population | None = None,
        strategy_tag: str | None = None,
        *,
        brokerage_bps: float = 0.0,
        slippage_bps: float = 0.0,
        portfolio_value: float | None = None,
        brokerage_flat: float = 0.0,
        min_trade_weight: float = _DUST_TRADE_TOL,
        allow_asset_subsets: bool = False,
    ) -> dict[str, dict[str, float]]:
        """Calculate the risk/return delta of holding vs rebalancing.

        Evaluates the current physical weights (hold scenario) against the
        realized out-of-sample portfolio(s) (rebalance scenario) across the
        identical, strictly aligned observation window. Computes CVaR,
        annualized Sharpe ratio, and Max Drawdown for both scenarios.

        When ``population`` is supplied, the rebalance scenario uses the
        matched out-of-sample portfolio's realized returns/metrics (net of any
        specified transition friction) so that no in-sample back-projection of
        terminal ``target_weights`` is performed. When no population is
        supplied, ``target_weights`` are evaluated over ``universe.returns`` as
        an in-sample historical projection and logged as such.

        Parameters
        ----------
        population : Population or None, optional
            The out-of-sample population object containing benchmarked strategy
            portfolios. If provided, its realized OOS returns and observation
            window are used.
        strategy_tag : str or None, optional
            Tag of the portfolio(s) within ``population`` to evaluate. If
            ``None``, the first portfolio in the population is used.
        brokerage_bps : float, optional
            Optional brokerage fee in basis points deducted as transition drag.
        slippage_bps : float, optional
            Optional slippage in basis points deducted as transition drag.
        portfolio_value : float or None, optional
            Total portfolio cash value (required when ``brokerage_flat > 0``).
        brokerage_flat : float, optional
            Optional flat brokerage fee per traded asset in currency units.
        min_trade_weight : float, optional
            Minimum gross weight change to trigger ``brokerage_flat``.
            Defaults to ``1e-6`` to ignore numerical solver dust.
        allow_asset_subsets : bool, optional
            If ``True``, allows OOS folds whose asset list is a strict subset of
            ``universe.tickers`` (for example, direct pre-selection ``Pipeline``
            outputs evaluated outside ``PortfolioExperimentEngine``), treating
            dropped tickers as 0% weight for transition/rebalance drag. Defaults
            to ``False`` to reject ``MultipleRandomizedCV`` asset-subsampled
            paths by default.

        Returns
        -------
        dict[str, dict[str, float]]
            A nested dictionary with keys ``"hold"`` and ``"rebalance"``,
            each containing ``"cvar"``, ``"sharpe"``, and
            ``"max_drawdown"`` keys mapping to ``float`` values (with
            ``"rebalance"`` additionally reporting ``"entry_drag"`` and
            ``"rebalance_drag"``).

        Raises
        ------
        TypeError
            If ``population`` is not None and not a ``skfolio.Population`` instance.
        ValueError
            If ``population`` is provided but empty, if ``strategy_tag`` matches
            no portfolio in ``population``, if OOS dates do not align strictly
            with ``universe.returns``, if OOS paths cover a strict asset subset
            of ``universe.tickers`` (when ``allow_asset_subsets=False``), or if
            ``universe.returns`` has not been populated.
        """
        if population is not None:
            if not isinstance(population, Population):
                raise TypeError("population must be a skfolio.Population instance.")
            if len(population) == 0:
                raise ValueError("population must not be empty.")

        returns: pd.DataFrame = self._universe.returns
        matched: list[Any] | None = None

        if population is not None:
            if strategy_tag is not None:
                matched = [
                    p for p in population if getattr(p, "tag", None) == strategy_tag
                ]
                if not matched:
                    raise ValueError(
                        f"No portfolio with tag '{strategy_tag}' found in population."
                    )
            else:
                matched = [population[0]]
        else:
            logger.warning(
                "calculate_rebalance_delta() called without an out-of-sample Population; "
                "evaluating target_weights over full universe.returns is an in-sample projection."
            )

        self._validate_bps(brokerage_bps, "brokerage_bps")
        self._validate_bps(slippage_bps, "slippage_bps")
        self._validate_bps(brokerage_flat, "brokerage_flat")
        self._validate_bps(min_trade_weight, "min_trade_weight")

        rebalance_cost_fn: (
            Callable[[dict[str, float], dict[str, float]], float] | None
        ) = None
        if brokerage_bps != 0.0 or slippage_bps != 0.0 or brokerage_flat != 0.0:
            self._validate_portfolio_value(portfolio_value, brokerage_flat)
            tickers = list(self._universe.tickers)

            def _pair_drag(
                from_weights: dict[str, float], to_weights: dict[str, float]
            ) -> float:
                return _compute_transition_drag_for_weights(
                    from_weights,
                    to_weights,
                    tickers=tickers,
                    brokerage_bps=brokerage_bps,
                    slippage_bps=slippage_bps,
                    portfolio_value=portfolio_value,
                    brokerage_flat=brokerage_flat,
                    min_trade_weight=min_trade_weight,
                )

            rebalance_cost_fn = _pair_drag

        fees_map = getattr(self._universe, "fees", None) or {}
        periods_per_year = _resolve_periods_per_year(self._universe)

        return evaluate_rebalance_benefit(
            universe_returns=returns,
            current_weights=self._current_weights,
            target_weights=self._target_weights,
            oos_portfolios=matched,
            annual_fees=fees_map,
            periods_per_year=periods_per_year,
            rebalance_cost_fn=rebalance_cost_fn,
            allow_asset_subsets=allow_asset_subsets,
        )

    @staticmethod
    def _validate_bps(value: float, name: str) -> None:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not np.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{name} must be non-negative; got {value}.")

    @staticmethod
    def _validate_portfolio_value(
        portfolio_value: float | None, brokerage_flat: float
    ) -> None:
        if portfolio_value is not None and (
            not isinstance(portfolio_value, (int, float))
            or isinstance(portfolio_value, bool)
            or not np.isfinite(portfolio_value)
            or portfolio_value <= 0
        ):
            raise ValueError("portfolio_value must be a positive finite number.")

        if brokerage_flat > 0.0 and portfolio_value is None:
            raise ValueError(
                "portfolio_value must be provided when brokerage_flat > 0."
            )

    def calculate_net_friction(
        self,
        brokerage_bps: float = 10.0,
        slippage_bps: float = 5.0,
        portfolio_value: float | None = None,
        brokerage_flat: float = 0.0,
        *,
        min_trade_weight: float = _DUST_TRADE_TOL,
    ) -> pd.DataFrame:
        """Calculate the transaction friction and management fee costs.

        Computes per-asset turnover as ``abs(target_weight -
        current_weight)``. Calculates transaction cost as
        ``turnover * (brokerage_bps + slippage_bps) / 10_000`` plus any
        proportional drag from ``brokerage_flat / portfolio_value`` for trades
        exceeding ``min_trade_weight`` (default ``1e-6`` to avoid charging flat
        tickets on numerical solver dust). Calculates annual TER cost as
        ``target_weight * universe.fees[ticker]``.

        Parameters
        ----------
        brokerage_bps : float, optional
            Brokerage fee in basis points, default ``10.0``.
        slippage_bps : float, optional
            Expected slippage in basis points, default ``5.0``.
        portfolio_value : float or None, optional
            Total portfolio cash value in currency units. Required if
            ``brokerage_flat > 0.0``.
        brokerage_flat : float, optional
            Flat brokerage fee per trade in currency units, default ``0.0``.
        min_trade_weight : float, optional
            Minimum gross weight change required to incur ``brokerage_flat``,
            default ``1e-6``.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: ``'ticker'``, ``'gross_weight_traded'``,
            ``'one_way_turnover'``, ``'transaction_cost'``, ``'annual_ter'``,
            and ``'annual_ter_savings'``. Includes a final summary row labelled
            ``'TOTAL'`` with summed values across all assets.

        Raises
        ------
        ValueError
            If ``brokerage_bps``, ``slippage_bps``, ``brokerage_flat``, or
            ``min_trade_weight`` are negative, if ``portfolio_value`` is
            non-positive or non-finite, or if ``brokerage_flat > 0`` without
            ``portfolio_value``.
        """
        self._validate_bps(brokerage_bps, "brokerage_bps")
        self._validate_bps(slippage_bps, "slippage_bps")
        self._validate_bps(brokerage_flat, "brokerage_flat")
        self._validate_bps(min_trade_weight, "min_trade_weight")
        self._validate_portfolio_value(portfolio_value, brokerage_flat)

        fees = self._universe.fees
        total_bps = brokerage_bps + slippage_bps

        # Union of all tickers across both weight dictionaries
        all_tickers = sorted(
            set(self._current_weights.keys()) | set(self._target_weights.keys())
        )

        rows = []
        for ticker in all_tickers:
            current_w = float(self._current_weights.get(ticker, 0.0))
            target_w = float(self._target_weights.get(ticker, 0.0))

            gross_weight_traded = abs(target_w - current_w)
            one_way_turnover = gross_weight_traded / 2.0

            flat_cost_weight = (
                (float(brokerage_flat) / float(portfolio_value))
                if (
                    brokerage_flat > 0.0
                    and portfolio_value is not None
                    and gross_weight_traded > min_trade_weight
                )
                else 0.0
            )
            transaction_cost = (
                gross_weight_traded * total_bps / 10_000.0 + flat_cost_weight
            )

            annual_ter = target_w * float(fees.get(ticker, 0.0))
            current_ter = current_w * float(fees.get(ticker, 0.0))
            annual_ter_savings = current_ter - annual_ter

            rows.append(
                {
                    "ticker": ticker,
                    "gross_weight_traded": gross_weight_traded,
                    "one_way_turnover": one_way_turnover,
                    "transaction_cost": transaction_cost,
                    "annual_ter": annual_ter,
                    "annual_ter_savings": annual_ter_savings,
                }
            )

        df = pd.DataFrame(
            rows,
            columns=[
                "ticker",
                "gross_weight_traded",
                "one_way_turnover",
                "transaction_cost",
                "annual_ter",
                "annual_ter_savings",
            ],
        )

        # Append summary row
        summary = pd.DataFrame(
            [
                {
                    "ticker": "TOTAL",
                    "gross_weight_traded": df["gross_weight_traded"].sum(),
                    "one_way_turnover": df["one_way_turnover"].sum(),
                    "transaction_cost": df["transaction_cost"].sum(),
                    "annual_ter": df["annual_ter"].sum(),
                    "annual_ter_savings": df["annual_ter_savings"].sum(),
                }
            ]
        )
        df = pd.concat([df, summary], ignore_index=True)

        return df
