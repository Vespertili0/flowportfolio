"""End-to-end remediation tests for the quantitative audit fixes.

These tests deliberately use real skfolio estimators, real ``Portfolio`` and
``MultiPeriodPortfolio`` objects, and synthetic daily price data. The only
test doubles are a stub data provider (no network access) and a ``Portfolio``
spy used to observe the evaluation window in ``PortfolioDeltaEngine``.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from skfolio import Population, RiskMeasure
from skfolio.optimization import HierarchicalRiskParity, MeanRisk
from skfolio.portfolio import MultiPeriodPortfolio, Portfolio
from skfolio.pre_selection import DropCorrelated
from sklearn import config_context

from flowportfolio.core.constraints import ConstraintBuilder
from flowportfolio.core.experiment import PortfolioExperimentEngine
from flowportfolio.core.reporting import Reporter
from flowportfolio.core.universe import Universe
from flowportfolio.delta import PortfolioDeltaEngine
from flowportfolio.priors import PriorSynthesiser
from flowportfolio.strategies import StrategyBuilder

TICKERS = ["SPY", "TLT", "GLD", "QQQ"]
METADATA = {"SPY": "core", "TLT": "core", "GLD": "satellite", "QQQ": "satellite"}
FEES = dict.fromkeys(TICKERS, 0.0050)


class StubProvider:
    """Deterministic in-memory data provider returning a yfinance-style frame."""

    def __init__(self, prices: pd.DataFrame) -> None:
        self._prices = prices

    def fetch(
        self,
        tickers: list[str],
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        return pd.concat({"Close": self._prices[tickers]}, axis=1)


def _synthetic_prices(n_days: int = 600, seed: int = 7) -> pd.DataFrame:
    """Build daily prices where SPY and QQQ are highly correlated by construction."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    market = rng.normal(0.0004, 0.010, n_days)
    returns = pd.DataFrame(
        {
            "SPY": market + rng.normal(0.0, 0.0010, n_days),
            "TLT": rng.normal(0.0001, 0.006, n_days),
            "GLD": rng.normal(0.0002, 0.008, n_days),
            "QQQ": 1.1 * market + rng.normal(0.0, 0.0010, n_days),
        },
        index=dates,
    )
    return 100.0 * (1.0 + returns).cumprod()


def _fetched_universe(n_days: int = 600) -> Universe:
    universe = Universe(TICKERS, METADATA, FEES)
    universe.fetch_data(data_provider=StubProvider(_synthetic_prices(n_days)))
    universe.anchor_history()
    return universe


def test_daily_fee_scaling_and_preselection() -> None:
    """Fees are daily-scaled dicts and survive a pre-selection that drops a column."""
    universe = _fetched_universe()
    current_weights = dict.fromkeys(TICKERS, 0.25)

    spec = ConstraintBuilder(universe).max_turnover(0.20, current_weights).build()

    assert spec.management_fees is not None
    assert spec.management_fees["SPY"] == pytest.approx(0.0050 / 252.0)
    assert isinstance(spec.previous_weights, dict)

    pipeline = (
        StrategyBuilder(constraints=spec)
        .add_pre_selection(DropCorrelated(threshold=0.5))
        .set_optimizer(MeanRisk())
        .build_pipeline()
    )

    # Pandas output preserves column names so dict-keyed fees/weights align by name.
    with config_context(transform_output="pandas"):
        pipeline.fit(universe.returns)

    fitted_weights = pipeline.named_steps["optimizer"].weights_
    assert len(fitted_weights) < len(TICKERS)


def test_prior_relative_and_group_views() -> None:
    """Only the RHS scalar is daily-scaled; group views and the BL rate are accepted."""
    universe = _fetched_universe()

    synth = PriorSynthesiser(universe)
    synth.add_market_view("1.0 * SPY - 0.5 * TLT >= 0.04", confidence=0.5)
    synth.add_market_view("core >= 0.06", confidence=0.5)

    assert synth._views[0]["view"] == "1.0 * SPY - 0.5 * TLT >= 0.00015873015873015873"
    assert synth._views[1]["view"] == f"core >= {0.06 / 252.0}"

    # skfolio's BlackLitterman supports equality views only, so the fitted check
    # uses equalities carrying the same relative and group structure.
    bl_synth = PriorSynthesiser(universe)
    bl_synth.add_market_view("1.0 * SPY - 0.5 * TLT == 0.04", confidence=0.5)
    bl_synth.add_market_view("core == 0.06", confidence=0.5)

    estimator = MeanRisk(
        prior_estimator=bl_synth.build_black_litterman_prior(risk_free_rate=0.04)
    )
    estimator.fit(universe.returns)

    assert estimator.prior_estimator.risk_free_rate == pytest.approx(0.04 / 252.0)


def test_unfetched_universe_initialisation() -> None:
    """Components can be constructed before fetch_data() without touching returns."""
    universe = Universe(TICKERS, METADATA, FEES)

    weights = dict.fromkeys(TICKERS, 0.25)

    ConstraintBuilder(universe)
    PriorSynthesiser(universe)
    PortfolioExperimentEngine(universe)
    PortfolioDeltaEngine(universe, current_weights=weights, target_weights=weights)


def test_holiday_ffill_and_truncation_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Post-inception holiday gaps are forward-filled and heavy truncation is logged."""
    n_days = 1000
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2018-01-01", periods=n_days)
    prices = pd.DataFrame(
        {
            "SPY": 100.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.01, n_days)),
            "NEW_ETF": 50.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.01, n_days)),
        },
        index=dates,
    )
    prices.iloc[800, prices.columns.get_loc("SPY")] = np.nan
    prices.iloc[:500, prices.columns.get_loc("NEW_ETF")] = np.nan

    universe = Universe(
        ["SPY", "NEW_ETF"],
        {"SPY": "core", "NEW_ETF": "satellite"},
        {"SPY": 0.001, "NEW_ETF": 0.002},
    )

    with caplog.at_level(logging.WARNING):
        universe.fetch_data(data_provider=StubProvider(prices))
        universe.anchor_history()

    assert "NEW_ETF" in caplog.text
    assert not universe.returns.isna().any().any()


def test_real_skfolio_reporting_and_delta() -> None:
    """Tearsheet shows real tickers and the delta engine uses the out-of-sample window."""
    universe = _fetched_universe(n_days=500)

    mean_risk = StrategyBuilder().set_optimizer(MeanRisk()).build_pipeline()
    hrp = StrategyBuilder().set_optimizer(HierarchicalRiskParity()).build_pipeline()

    engine = PortfolioExperimentEngine(universe, n_jobs=1)
    engine.add_strategy("MeanRisk", mean_risk, {"optimizer__l1_coef": [0.0, 0.0005]})
    engine.add_strategy(
        "HRP",
        hrp,
        {"optimizer__risk_measure": [RiskMeasure.VARIANCE, RiskMeasure.CVAR]},
    )

    population = engine.run_robustness_test(
        cv_type="walk_forward", train_size=100, test_size=25
    )

    assert isinstance(population, Population)
    assert all(isinstance(p, MultiPeriodPortfolio) for p in population)

    markdown = Reporter(population).to_markdown_artifact()
    holdings_line = next(
        line for line in markdown.splitlines() if "Top 5 Holdings" in line
    )
    assert "Asset_0" not in holdings_line
    assert any(ticker in holdings_line for ticker in TICKERS)

    weights = dict.fromkeys(TICKERS, 0.25)
    delta_engine = PortfolioDeltaEngine(
        universe, current_weights=weights, target_weights=weights
    )
    with patch("flowportfolio.delta.Portfolio", wraps=Portfolio) as portfolio_spy:
        delta_engine.calculate_rebalance_delta(population=population)

    evaluated = portfolio_spy.call_args_list[0].kwargs["X"]
    assert len(evaluated) == len(population[0].returns_df)
    assert len(evaluated) < len(universe.returns)
