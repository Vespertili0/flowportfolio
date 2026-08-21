"""Portfolio delta engine module.

This module provides the :class:`PortfolioDeltaEngine` class, which bridges
the gap between an optimised target allocation (from the
``PortfolioExperimentEngine``) and a user's actual physical holdings. It
produces decision-ready trade analysis by quantifying the net risk/return
benefit of executing a portfolio rebalance versus holding current positions,
factoring in group drift, Total Expense Ratio (TER), slippage, and brokerage
friction.

**Prefect Compatibility:** All public methods are pure functions of the
constructor's stored state. They accept explicit data structures and return
serialisable outputs, ensuring zero friction inside Prefect ``@task``
execution contexts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from skfolio import Population, RatioMeasure
from skfolio.portfolio import Portfolio

from flowportfolio.core.universe import Universe


class PortfolioDeltaEngine:
    """Evaluates the decision to rebalance a portfolio against current holdings.

    This engine quantifies the trade-off of executing a rebalance versus
    holding current positions (net of transaction friction and management
    fees).

    Prefect Compatibility: All methods are pure functions of the
    constructor's stored state and are safe to wrap as Prefect ``@task``
    decorators.

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
        universe: Universe,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
    ) -> None:
        # --- type validation ---
        if not isinstance(universe, Universe):
            raise TypeError("universe must be a Universe instance.")
        if not isinstance(current_weights, dict):
            raise TypeError("current_weights must be a dict.")
        if not isinstance(target_weights, dict):
            raise TypeError("target_weights must be a dict.")
        if not all(isinstance(v, (int, float)) for v in current_weights.values()):
            raise TypeError(
                "current_weights values must all be numeric (int or float)."
            )
        if not all(isinstance(v, (int, float)) for v in target_weights.values()):
            raise TypeError(
                "target_weights values must all be numeric (int or float)."
            )

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
        missing_current = set(current_weights.keys()) - universe_tickers
        if missing_current:
            raise ValueError(
                f"current_weights contains tickers not found in universe: "
                f"{sorted(missing_current)}."
            )
        missing_target = set(target_weights.keys()) - universe_tickers
        if missing_target:
            raise ValueError(
                f"target_weights contains tickers not found in universe: "
                f"{sorted(missing_target)}."
            )

        self._universe: Universe = universe
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
        groups: dict[str, dict[str, float]] = {}

        for ticker, weight in self._current_weights.items():
            group = metadata[ticker]
            if group not in groups:
                groups[group] = {"current_allocation": 0.0, "target_allocation": 0.0}
            groups[group]["current_allocation"] += float(weight)

        for ticker, weight in self._target_weights.items():
            group = metadata[ticker]
            if group not in groups:
                groups[group] = {"current_allocation": 0.0, "target_allocation": 0.0}
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

    def calculate_rebalance_delta(self, population: Population) -> dict:
        """Calculate the risk/return delta of holding vs rebalancing.

        Takes the skfolio ``Population`` object (e.g. from robustness
        testing) and creates a pseudo-portfolio representing the current
        physical weights applied to the universe's historical returns.
        Computes Mean CVaR, Sharpe ratio, and Max Drawdown for both the
        ``'hold'`` portfolio and the ``'rebalance'`` portfolio (best
        strategy from the population, selected by maximum CVaR ratio).

        Parameters
        ----------
        population : Population
            The out-of-sample population object containing the target
            strategy portfolios.

        Returns
        -------
        dict
            A nested dictionary with keys ``"hold"`` and ``"rebalance"``,
            each containing ``"cvar"``, ``"sharpe"``, and
            ``"max_drawdown"`` keys mapping to ``float`` values.

        Raises
        ------
        TypeError
            If ``population`` is not a ``skfolio.Population`` instance.
        ValueError
            If ``population`` is empty, or if ``universe.returns`` has not
            been populated (i.e. ``fetch_data()`` has not been called).
        """
        if not isinstance(population, Population):
            raise TypeError(
                "population must be a skfolio.Population instance."
            )
        if len(population) == 0:
            raise ValueError("population must not be empty.")

        # Retrieve universe returns — propagates ValueError if not fetched
        returns: pd.DataFrame = self._universe.returns

        # Build weight vector aligned to returns column order; default 0.0
        # for any universe ticker absent from current_weights.
        weight_vector = np.array(
            [self._current_weights.get(col, 0.0) for col in returns.columns],
            dtype=float,
        )

        # Construct skfolio Portfolio for the "hold" scenario
        hold_portfolio = Portfolio(
            X=returns.to_numpy(),
            weights=weight_vector,
        )

        # Select the best portfolio from the population by CVaR ratio
        rebalance_portfolio = population.max_measure(RatioMeasure.CVAR_RATIO)

        return {
            "hold": {
                "cvar": float(hold_portfolio.cvar),
                "sharpe": float(hold_portfolio.sharpe_ratio),
                "max_drawdown": float(hold_portfolio.max_drawdown),
            },
            "rebalance": {
                "cvar": float(rebalance_portfolio.cvar),
                "sharpe": float(rebalance_portfolio.sharpe_ratio),
                "max_drawdown": float(rebalance_portfolio.max_drawdown),
            },
        }

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
            DataFrame with columns: ``'ticker'``, ``'turnover'``,
            ``'transaction_cost'``, ``'annual_ter'``, and
            ``'total_friction'``. Includes a final summary row labelled
            ``'TOTAL'`` with summed values across all assets.

        Raises
        ------
        ValueError
            If ``brokerage_bps`` or ``slippage_bps`` are negative.
        """
        if brokerage_bps < 0:
            raise ValueError(
                f"brokerage_bps must be non-negative; got {brokerage_bps}."
            )
        if slippage_bps < 0:
            raise ValueError(
                f"slippage_bps must be non-negative; got {slippage_bps}."
            )

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

            turnover = abs(target_w - current_w)
            transaction_cost = turnover * total_bps / 10_000.0
            annual_ter = target_w * float(fees.get(ticker, 0.0))
            total_friction = transaction_cost + annual_ter

            rows.append(
                {
                    "ticker": ticker,
                    "turnover": turnover,
                    "transaction_cost": transaction_cost,
                    "annual_ter": annual_ter,
                    "total_friction": total_friction,
                }
            )

        df = pd.DataFrame(
            rows,
            columns=["ticker", "turnover", "transaction_cost", "annual_ter", "total_friction"],
        )

        # Append summary row
        summary = pd.DataFrame(
            [
                {
                    "ticker": "TOTAL",
                    "turnover": df["turnover"].sum(),
                    "transaction_cost": df["transaction_cost"].sum(),
                    "annual_ter": df["annual_ter"].sum(),
                    "total_friction": df["total_friction"].sum(),
                }
            ]
        )
        df = pd.concat([df, summary], ignore_index=True)

        return df
