"""Portfolio delta engine module.

This module provides the :class:`PortfolioDeltaEngine` class, which bridges
the gap between an optimised target allocation (from the
``PortfolioExperimentEngine``) and a user's actual physical holdings. It
produces decision-ready trade analysis by quantifying the net risk/return
benefit of executing a portfolio rebalance versus holding current positions,
factoring in group drift, Total Expense Ratio (TER), slippage, and brokerage
friction.

**"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from skfolio import Population
from skfolio.portfolio import Portfolio

from flowportfolio.core.protocols import UniverseProtocol, is_universe_protocol
from flowportfolio.core.validation import validate_weight

logger = logging.getLogger(__name__)


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
    ) -> dict:
        """Calculate the risk/return delta of holding vs rebalancing.

        Evaluates the current physical weights (hold scenario) and the target
        weights (rebalance scenario) across the identical returns evaluation
        window. If an out-of-sample ``Population`` is provided, the evaluation
        window is aligned to the selected portfolio's observation period
        (via its ``returns_df`` index). Computes CVaR, Sharpe ratio, and Max
        Drawdown for both scenarios.

        Note: The ``hold_portfolio`` models an un-rebalanced buy-and-hold
        portfolio whose weights drift with asset returns
        (``weight_drift=True``). When the selected out-of-sample portfolio
        exposes pre-computed metrics, those are used for the rebalance
        scenario so that no in-sample back-projection of terminal
        ``target_weights`` is performed. When no population is supplied,
        ``target_weights`` are evaluated over the full ``universe.returns``,
        which is an in-sample historical projection and is logged as such.

        Parameters
        ----------
        population : Population or None, optional
            The out-of-sample population object containing benchmarked strategy
            portfolios. If provided, its observation window is used to evaluate
            both current and target holdings.
        strategy_tag : str or None, optional
            Tag of the portfolio within ``population`` to evaluate. If
            ``None``, the first portfolio in the population is used.

        Returns
        -------
        dict
            A nested dictionary with keys ``"hold"`` and ``"rebalance"``,
            each containing ``"cvar"``, ``"sharpe"``, and
            ``"max_drawdown"`` keys mapping to ``float`` values.

        Raises
        ------
        TypeError
            If ``population`` is not None and not a ``skfolio.Population`` instance.
        ValueError
            If ``population`` is provided but empty, if ``strategy_tag`` matches
            no portfolio in ``population``, or if ``universe.returns`` has not
            been populated.
        """
        if population is not None:
            if not isinstance(population, Population):
                raise TypeError("population must be a skfolio.Population instance.")
            if len(population) == 0:
                raise ValueError("population must not be empty.")

        # Retrieve universe returns — propagates ValueError if not fetched
        returns: pd.DataFrame = self._universe.returns
        eval_returns = returns
        selected_port = None
        matched = []

        if population is not None and len(population) > 0:
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

            selected_port = matched[0]

            port_returns = getattr(selected_port, "returns_df", None)
            if port_returns is None:
                port_returns = getattr(selected_port, "returns", None)

            if isinstance(port_returns, (pd.Series, pd.DataFrame)):
                if (
                    hasattr(port_returns, "index")
                    and port_returns.index.isin(returns.index).any()
                ):
                    eval_returns = returns.loc[returns.index.isin(port_returns.index)]
                elif len(port_returns) <= len(returns):
                    eval_returns = returns.iloc[-len(port_returns) :]
            elif isinstance(port_returns, np.ndarray) and 0 < len(port_returns) <= len(
                returns
            ):
                eval_returns = returns.iloc[-len(port_returns) :]
        else:
            logger.warning(
                "calculate_rebalance_delta() called without an out-of-sample Population; "
                "evaluating target_weights over full universe.returns is an in-sample projection."
            )

        fees_map = getattr(self._universe, "fees", None) or {}
        daily_fees_vector = np.array(
            [float(fees_map.get(col, 0.0)) / 252.0 for col in eval_returns.columns],
            dtype=float,
        )

        # Build weight vectors aligned to returns column order
        current_weight_vector = np.array(
            [self._current_weights.get(col, 0.0) for col in eval_returns.columns],
            dtype=float,
        )
        target_weight_vector = np.array(
            [self._target_weights.get(col, 0.0) for col in eval_returns.columns],
            dtype=float,
        )

        hold_portfolio = Portfolio(
            X=eval_returns,
            weights=current_weight_vector,
            weight_drift=True,
            management_fees=daily_fees_vector,
        )

        # Use out-of-sample MultiPeriodPortfolio/Portfolio metrics when available on real skfolio objects
        has_oos_metrics = (
            bool(matched)
            and all(
                getattr(p, "returns_df", None) is not None
                and getattr(p, "cvar", None) is not None
                and getattr(p, "sharpe_ratio", None) is not None
                and getattr(p, "max_drawdown", None) is not None
                for p in matched
            )
        )
        if has_oos_metrics:
            rebalance_metrics = {
                "cvar": float(np.median([float(p.cvar) for p in matched])),
                "sharpe": float(np.median([float(p.sharpe_ratio) for p in matched])),
                "max_drawdown": float(np.median([float(p.max_drawdown) for p in matched])),
            }
        else:
            rebalance_portfolio = Portfolio(
                X=eval_returns,
                weights=target_weight_vector,
                weight_drift=True,
                management_fees=daily_fees_vector,
            )
            rebalance_metrics = {
                "cvar": float(rebalance_portfolio.cvar),
                "sharpe": float(rebalance_portfolio.sharpe_ratio),
                "max_drawdown": float(rebalance_portfolio.max_drawdown),
            }

        return {
            "hold": {
                "cvar": float(hold_portfolio.cvar),
                "sharpe": float(hold_portfolio.sharpe_ratio),
                "max_drawdown": float(hold_portfolio.max_drawdown),
            },
            "rebalance": rebalance_metrics,
        }

    @staticmethod
    def _validate_bps(value: float, name: str) -> None:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not np.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{name} must be non-negative; got {value}.")

    def calculate_net_friction(
        self,
        brokerage_bps: float = 10.0,
        slippage_bps: float = 5.0,
    ) -> pd.DataFrame:
        """Calculate the transaction friction and management fee costs.

        Computes per-asset turnover as ``abs(target_weight -
        current_weight)``. Calculates transaction cost as
        ``turnover * (brokerage_bps + slippage_bps) / 10_000``. Calculates
        annual TER cost as ``target_weight * universe.fees[ticker]``.

        Parameters
        ----------
        brokerage_bps : float, optional
            Brokerage fee in basis points, default ``10.0``.
        slippage_bps : float, optional
            Expected slippage in basis points, default ``5.0``.

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
            If ``brokerage_bps`` or ``slippage_bps`` are negative.
        """
        self._validate_bps(brokerage_bps, "brokerage_bps")
        self._validate_bps(slippage_bps, "slippage_bps")

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

            transaction_cost = gross_weight_traded * total_bps / 10_000.0

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
