"""Unit and unmocked skfolio oracle tests for PortfolioDeltaEngine (Cycle 9)."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
from skfolio import MultiPeriodPortfolio, Population, Portfolio

from flowportfolio import delta as delta_module
from flowportfolio.core.universe import Universe
from flowportfolio.delta import PortfolioDeltaEngine


@pytest.fixture
def stub_universe() -> Universe:
    """Real Universe with deterministic daily returns (zero mocks)."""
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2025-01-02", periods=30)
    u = Universe(
        tickers=["AAPL", "MSFT", "GOOG"],
        metadata={"AAPL": "tech", "MSFT": "tech", "GOOG": "tech"},
        fees={"AAPL": 0.001, "MSFT": 0.002, "GOOG": 0.0015},
    )
    u._returns = pd.DataFrame(
        {
            "AAPL": rng.normal(0.0005, 0.012, len(idx)),
            "MSFT": rng.normal(0.0007, 0.010, len(idx)),
            "GOOG": rng.normal(0.0004, 0.011, len(idx)),
        },
        index=idx,
    )
    return u


# ---------------------------------------------------------------------------
# Validation & Accounting Tests
# ---------------------------------------------------------------------------


def test_init_raises_type_error(stub_universe: Universe) -> None:
    with pytest.raises(TypeError, match="current_weights must be a dict"):
        PortfolioDeltaEngine(
            universe=stub_universe,
            current_weights=[],  # type: ignore[arg-type]
            target_weights={"AAPL": 1.0, "MSFT": 0.0, "GOOG": 0.0},
        )


def test_init_raises_type_error_non_string_keys(stub_universe: Universe) -> None:
    with pytest.raises(TypeError, match="current_weights keys must all be strings"):
        PortfolioDeltaEngine(
            universe=stub_universe,
            current_weights={1: 1.0, "MSFT": 0.0, "GOOG": 0.0},  # type: ignore[dict-item]
            target_weights={"AAPL": 1.0, "MSFT": 0.0, "GOOG": 0.0},
        )


def test_init_raises_type_error_bool_values(stub_universe: Universe) -> None:
    with pytest.raises(TypeError, match="current_weights values must all be numeric"):
        PortfolioDeltaEngine(
            universe=stub_universe,
            current_weights={"AAPL": True, "MSFT": False, "GOOG": False},
            target_weights={"AAPL": 1.0, "MSFT": 0.0, "GOOG": 0.0},
        )


def test_init_raises_value_error_nan_values(stub_universe: Universe) -> None:
    with pytest.raises(
        ValueError, match="current_weights values must be finite numbers"
    ):
        PortfolioDeltaEngine(
            universe=stub_universe,
            current_weights={"AAPL": float("nan"), "MSFT": 0.0, "GOOG": 0.0},
            target_weights={"AAPL": 1.0, "MSFT": 0.0, "GOOG": 0.0},
        )


def test_init_raises_value_error_sum_current(stub_universe: Universe) -> None:
    with pytest.raises(ValueError, match="current_weights must sum to 1.0"):
        PortfolioDeltaEngine(
            universe=stub_universe,
            current_weights={"AAPL": 0.5, "MSFT": 0.0, "GOOG": 0.0},
            target_weights={"AAPL": 1.0, "MSFT": 0.0, "GOOG": 0.0},
        )


def test_init_raises_value_error_sum_target(stub_universe: Universe) -> None:
    with pytest.raises(ValueError, match="target_weights must sum to 1.0"):
        PortfolioDeltaEngine(
            universe=stub_universe,
            current_weights={"AAPL": 1.0, "MSFT": 0.0, "GOOG": 0.0},
            target_weights={"AAPL": 0.5, "MSFT": 0.0, "GOOG": 0.0},
        )


def test_init_raises_value_error_missing_current(stub_universe: Universe) -> None:
    with pytest.raises(
        ValueError, match="current_weights contains tickers not found in universe"
    ):
        PortfolioDeltaEngine(
            universe=stub_universe,
            current_weights={"TSLA": 1.0},
            target_weights={"AAPL": 1.0, "MSFT": 0.0, "GOOG": 0.0},
        )


def test_init_raises_value_error_missing_target(stub_universe: Universe) -> None:
    with pytest.raises(
        ValueError, match="target_weights contains tickers not found in universe"
    ):
        PortfolioDeltaEngine(
            universe=stub_universe,
            current_weights={"AAPL": 1.0, "MSFT": 0.0, "GOOG": 0.0},
            target_weights={"TSLA": 1.0},
        )


def test_init_raises_value_error_missing_universe_current(
    stub_universe: Universe,
) -> None:
    with pytest.raises(ValueError, match="current_weights is missing universe tickers"):
        PortfolioDeltaEngine(
            universe=stub_universe,
            current_weights={"AAPL": 1.0},
            target_weights={"AAPL": 1.0, "MSFT": 0.0, "GOOG": 0.0},
        )


def test_init_raises_value_error_missing_universe_target(
    stub_universe: Universe,
) -> None:
    with pytest.raises(ValueError, match="target_weights is missing universe tickers"):
        PortfolioDeltaEngine(
            universe=stub_universe,
            current_weights={"AAPL": 1.0, "MSFT": 0.0, "GOOG": 0.0},
            target_weights={"AAPL": 1.0},
        )


def test_calculate_group_drift(stub_universe: Universe) -> None:
    engine = PortfolioDeltaEngine(
        universe=stub_universe,
        current_weights={"AAPL": 0.6, "MSFT": 0.4, "GOOG": 0.0},
        target_weights={"AAPL": 0.5, "MSFT": 0.3, "GOOG": 0.2},
    )
    df = engine.calculate_group_drift()

    assert "tech" in df.index
    assert df.loc["tech", "current_allocation"] == pytest.approx(1.0)
    assert df.loc["tech", "target_allocation"] == pytest.approx(1.0)
    assert df.loc["tech", "drift"] == pytest.approx(0.0)
    assert df.loc["tech", "abs_drift"] == pytest.approx(0.0)


def test_calculate_net_friction(stub_universe: Universe) -> None:
    engine = PortfolioDeltaEngine(
        universe=stub_universe,
        current_weights={"AAPL": 0.6, "MSFT": 0.4, "GOOG": 0.0},
        target_weights={"AAPL": 0.5, "MSFT": 0.3, "GOOG": 0.2},
    )
    df = engine.calculate_net_friction(brokerage_bps=10.0, slippage_bps=5.0)

    assert "gross_weight_traded" in df.columns
    assert "transaction_cost" in df.columns
    total_turnover = df[df["ticker"] == "TOTAL"]["gross_weight_traded"].iloc[0]
    assert np.isclose(total_turnover, 0.4)


def test_calculate_net_friction_raises_value_error(stub_universe: Universe) -> None:
    engine = PortfolioDeltaEngine(
        universe=stub_universe,
        current_weights={"AAPL": 1.0, "MSFT": 0.0, "GOOG": 0.0},
        target_weights={"AAPL": 1.0, "MSFT": 0.0, "GOOG": 0.0},
    )
    with pytest.raises(ValueError, match="brokerage_bps must be non-negative"):
        engine.calculate_net_friction(brokerage_bps=-5.0)


def test_delta_rejects_short_weights_and_applies_fees_to_hold() -> None:
    """Real skfolio check: short weights rejected and hold portfolio deducts universe fees."""
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2025-01-02", periods=40)
    rets = pd.DataFrame(
        {"SPY": rng.normal(0.001, 0.01, 40), "TLT": rng.normal(0.0005, 0.008, 40)},
        index=idx,
    )
    u_fees = Universe(
        tickers=["SPY", "TLT"],
        metadata={"SPY": "equity", "TLT": "bonds"},
        fees={"SPY": 0.0252, "TLT": 0.0504},  # distinct annual fees
    )
    u_fees._returns = rets

    with pytest.raises(ValueError, match="Weight must be between 0.0 and 1.0"):
        PortfolioDeltaEngine(
            u_fees, {"SPY": 1.2, "TLT": -0.2}, {"SPY": 0.5, "TLT": 0.5}
        )

    engine = PortfolioDeltaEngine(
        u_fees, {"SPY": 0.6, "TLT": 0.4}, {"SPY": 0.5, "TLT": 0.5}
    )
    res = engine.calculate_rebalance_delta()

    expected_hold = Portfolio(
        X=rets,
        weights=[0.6, 0.4],
        weight_drift=True,
        management_fees=np.array([0.0252 / 252.0, 0.0504 / 252.0]),
        annualization_factor=252.0,
    )
    assert res["hold"]["cvar"] == pytest.approx(float(expected_hold.cvar), abs=1e-12)
    assert res["hold"]["sharpe"] == pytest.approx(
        float(expected_hold.annualized_sharpe_ratio), abs=1e-12
    )
    assert res["hold"]["max_drawdown"] == pytest.approx(
        float(expected_hold.max_drawdown), abs=1e-12
    )


def test_calculate_rebalance_delta_multipath_window_alignment() -> None:
    """Multi-path Population uses median realized OOS metrics for rebalance and aligned hold."""
    idx = pd.bdate_range("2024-01-01", periods=30)
    rng = np.random.default_rng(7)
    returns = pd.DataFrame(
        rng.normal(0.0005, 0.01, (30, 2)), columns=["SPY", "TLT"], index=idx
    )
    u = Universe(
        ["SPY", "TLT"], {"SPY": "equity", "TLT": "bonds"}, {"SPY": 0.001, "TLT": 0.002}
    )
    u._returns = returns

    # Distinct realized OOS portfolios (different weights from target_weights!)
    p1 = Portfolio(X=returns.iloc[:15], weights=[0.9, 0.1], tag="CPCV")
    p2 = Portfolio(X=returns.iloc[15:], weights=[0.2, 0.8], tag="CPCV")
    pop = Population([p1, p2])

    engine = PortfolioDeltaEngine(u, {"SPY": 0.7, "TLT": 0.3}, {"SPY": 0.5, "TLT": 0.5})
    res = engine.calculate_rebalance_delta(pop, strategy_tag="CPCV")

    fees_vec = np.array([0.001 / 252.0, 0.002 / 252.0])
    h1 = Portfolio(
        X=returns.iloc[:15],
        weights=[0.7, 0.3],
        weight_drift=True,
        management_fees=fees_vec,
        annualization_factor=252.0,
    )
    h2 = Portfolio(
        X=returns.iloc[15:],
        weights=[0.7, 0.3],
        weight_drift=True,
        management_fees=fees_vec,
        annualization_factor=252.0,
    )

    assert res["hold"]["cvar"] == pytest.approx(
        np.median([h1.cvar, h2.cvar]), abs=1e-12
    )
    assert res["hold"]["sharpe"] == pytest.approx(
        np.median([h1.annualized_sharpe_ratio, h2.annualized_sharpe_ratio]), abs=1e-12
    )
    # Rebalance MUST equal median of realized OOS p1 & p2 (NOT back-projected target_weights [0.5, 0.5])
    assert res["rebalance"]["cvar"] == pytest.approx(
        np.median([p1.cvar, p2.cvar]), abs=1e-12
    )
    assert res["rebalance"]["sharpe"] == pytest.approx(
        np.median([p1.annualized_sharpe_ratio, p2.annualized_sharpe_ratio]), abs=1e-12
    )
    assert res["rebalance"]["max_drawdown"] == pytest.approx(
        np.median([p1.max_drawdown, p2.max_drawdown]), abs=1e-12
    )


def test_calculate_rebalance_delta_symmetric_when_hold_equals_target(
    stub_universe: Universe,
) -> None:
    """Verify that when current_weights == target_weights (in-sample mode), hold and rebalance match."""
    w = {"AAPL": 0.5, "MSFT": 0.3, "GOOG": 0.2}
    engine = PortfolioDeltaEngine(
        universe=stub_universe, current_weights=w, target_weights=w
    )
    res = engine.calculate_rebalance_delta()

    assert np.isclose(res["hold"]["cvar"], res["rebalance"]["cvar"])
    assert np.isclose(res["hold"]["sharpe"], res["rebalance"]["sharpe"])
    assert np.isclose(res["hold"]["max_drawdown"], res["rebalance"]["max_drawdown"])


def test_calculate_net_friction_with_portfolio_value_and_flat_brokerage(
    stub_universe: Universe,
) -> None:
    """Verify flat brokerage drag calculation and parameter validations."""
    engine = PortfolioDeltaEngine(
        universe=stub_universe,
        current_weights={"AAPL": 0.6, "MSFT": 0.4, "GOOG": 0.0},
        target_weights={"AAPL": 0.5, "MSFT": 0.3, "GOOG": 0.2},
    )
    df = engine.calculate_net_friction(
        portfolio_value=100000.0,
        brokerage_flat=10.0,
        brokerage_bps=10.0,
        slippage_bps=5.0,
    )

    aapl_cost = df[df["ticker"] == "AAPL"]["transaction_cost"].iloc[0]
    assert np.isclose(aapl_cost, 0.00025)

    with pytest.raises(
        ValueError, match="portfolio_value must be provided when brokerage_flat > 0"
    ):
        engine.calculate_net_friction(brokerage_flat=10.0, portfolio_value=None)

    with pytest.raises(
        ValueError, match="portfolio_value must be a positive finite number"
    ):
        engine.calculate_net_friction(brokerage_flat=10.0, portfolio_value=-1000.0)


# ---------------------------------------------------------------------------
# Cycle 9 (F-01, F-10, F-11): O1 Regime Flip, Alignment, Friction, Periodicity
# ---------------------------------------------------------------------------


def test_calculate_rebalance_delta_o1_regime_flip_uses_realized_oos_metrics() -> None:
    """F-01 (O1_REGIME_FLIP): Rebalance scenario must reflect realized OOS performance, not back-projection."""
    dates = pd.bdate_range("2025-01-02", periods=60)
    rng = np.random.default_rng(42)
    half1 = pd.DataFrame(
        {
            "A": 0.005 + rng.normal(0.0, 0.002, 30),
            "B": -0.005 + rng.normal(0.0, 0.002, 30),
        },
        index=dates[:30],
    )
    half2 = pd.DataFrame(
        {
            "A": -0.020 + rng.normal(0.0, 0.002, 30),
            "B": 0.020 + rng.normal(0.0, 0.002, 30),
        },
        index=dates[30:],
    )
    rets = pd.concat([half1, half2])

    # Fold 1 trained on Half 1 held [0.99, 0.01] during Half 2 -> terrible OOS outcome
    oos_fold = Portfolio(X=half2, weights=[0.99, 0.01], tag="TrendStrat")
    mpp = MultiPeriodPortfolio(portfolios=[oos_fold], tag="TrendStrat")
    pop = Population([mpp])

    u = Universe(["A", "B"], {"A": "eq", "B": "eq"}, {"A": 0.0, "B": 0.0})
    u._returns = rets
    # Live terminal weights (fitted on Half 2) flipped to [0.01, 0.99]
    eng = PortfolioDeltaEngine(
        u,
        current_weights={"A": 0.5, "B": 0.5},
        target_weights={"A": 0.01, "B": 0.99},
    )
    res = eng.calculate_rebalance_delta(pop, strategy_tag="TrendStrat")

    assert res["rebalance"]["sharpe"] < 0.0
    assert res["rebalance"]["sharpe"] == pytest.approx(
        float(mpp.annualized_sharpe_ratio), rel=0, abs=1e-12
    )
    assert res["rebalance"]["cvar"] == pytest.approx(float(mpp.cvar), rel=0, abs=1e-12)
    assert res["rebalance"]["max_drawdown"] == pytest.approx(
        float(mpp.max_drawdown), rel=0, abs=1e-12
    )


def test_calculate_rebalance_delta_deducts_transition_cost_drag() -> None:
    """F-01: Transition costs (bps + flat) are deducted from the rebalance comparison."""
    rng = np.random.default_rng(19)
    idx = pd.bdate_range("2025-01-02", periods=50)
    rets = pd.DataFrame(
        {"A": rng.normal(0.001, 0.01, 50), "B": rng.normal(0.0005, 0.01, 50)},
        index=idx,
    )
    oos = Portfolio(X=rets, weights=[0.8, 0.2], tag="Strat")
    pop = Population([oos])

    u = Universe(["A", "B"], {"A": "eq", "B": "eq"}, {"A": 0.0, "B": 0.0})
    u._returns = rets
    eng = PortfolioDeltaEngine(
        u, current_weights={"A": 0.2, "B": 0.8}, target_weights={"A": 0.8, "B": 0.2}
    )

    res_gross = eng.calculate_rebalance_delta(pop, strategy_tag="Strat")
    res_net = eng.calculate_rebalance_delta(
        pop,
        strategy_tag="Strat",
        brokerage_bps=50.0,
        slippage_bps=25.0,
        brokerage_flat=50.0,
        portfolio_value=10_000.0,
    )

    # Exact transition drag = 1.2 * 75 / 10_000 + 2 * (50 / 10_000) = 0.009 + 0.010 = 0.019
    expected_drag = 1.2 * 75.0 / 10_000.0 + 2.0 * (50.0 / 10_000.0)
    net_series = np.asarray(oos.returns, dtype=float).copy()
    net_series[0] -= expected_drag
    expected_net_port = Portfolio(
        X=net_series[:, None],
        weights=np.array([1.0]),
        annualization_factor=252.0,
    )

    assert res_net["rebalance"]["sharpe"] < res_gross["rebalance"]["sharpe"]
    assert res_net["rebalance"]["sharpe"] == pytest.approx(
        float(expected_net_port.annualized_sharpe_ratio), rel=0, abs=1e-12
    )
    assert res_net["rebalance"]["cvar"] == pytest.approx(
        float(expected_net_port.cvar), rel=0, abs=1e-12
    )


def test_evaluate_rebalance_benefit_pure_stateless_helper() -> None:
    """F-01: Pure evaluate_rebalance_benefit helper operates without Universe or mocks."""
    from flowportfolio.delta import evaluate_rebalance_benefit

    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2025-01-02", periods=40)
    rets = pd.DataFrame(
        {"A": rng.normal(0.001, 0.01, 40), "B": rng.normal(0.0002, 0.01, 40)},
        index=idx,
    )
    oos_port = Portfolio(X=rets, weights=[0.9, 0.1], tag="PureStrat")

    res = evaluate_rebalance_benefit(
        universe_returns=rets,
        current_weights={"A": 0.5, "B": 0.5},
        target_weights={"A": 0.9, "B": 0.1},
        oos_portfolios=[oos_port],
        annual_fees={"A": 0.0, "B": 0.0},
        periods_per_year=252.0,
    )
    assert res["rebalance"]["sharpe"] == pytest.approx(
        float(oos_port.annualized_sharpe_ratio), rel=0, abs=1e-12
    )
    assert res["rebalance"]["cvar"] == pytest.approx(
        float(oos_port.cvar), rel=0, abs=1e-12
    )


def test_calculate_rebalance_delta_rejects_disjoint_and_partial_indices() -> None:
    """F-01: Disjoint or partially overlapping portfolio indices must raise ValueError (no trailing iloc fallback)."""
    rng = np.random.default_rng(9)
    u_idx = pd.bdate_range("2025-01-02", periods=30)
    u_rets = pd.DataFrame(
        rng.normal(0.0005, 0.01, (30, 2)), columns=["A", "B"], index=u_idx
    )
    u = Universe(["A", "B"], {"A": "eq", "B": "eq"}, {"A": 0.0, "B": 0.0})
    u._returns = u_rets

    eng = PortfolioDeltaEngine(
        u, current_weights={"A": 0.5, "B": 0.5}, target_weights={"A": 0.6, "B": 0.4}
    )

    # 1. Completely disjoint dates (2020 vs 2025)
    disjoint_idx = pd.bdate_range("2020-01-02", periods=15)
    disjoint_rets = pd.DataFrame(
        rng.normal(0.0005, 0.01, (15, 2)), columns=["A", "B"], index=disjoint_idx
    )
    disjoint_pop = Population(
        [Portfolio(X=disjoint_rets, weights=[0.6, 0.4], tag="Old")]
    )
    with pytest.raises(ValueError, match="aligned"):
        eng.calculate_rebalance_delta(disjoint_pop, strategy_tag="Old")

    # 2. Partial overlap (5 dates outside universe + 10 dates inside universe)
    partial_idx = pd.bdate_range("2024-12-24", periods=15)
    partial_rets = pd.DataFrame(
        rng.normal(0.0005, 0.01, (15, 2)), columns=["A", "B"], index=partial_idx
    )
    partial_pop = Population(
        [Portfolio(X=partial_rets, weights=[0.6, 0.4], tag="Partial")]
    )
    with pytest.raises(ValueError, match="aligned"):
        eng.calculate_rebalance_delta(partial_pop, strategy_tag="Partial")


@pytest.mark.parametrize(("freq", "expected_ppy"), [("W-FRI", 52), ("BME", 12)])
def test_calculate_rebalance_delta_scales_fees_and_annualization_by_periods_per_year(
    freq: str, expected_ppy: int
) -> None:
    """F-01: Hold fees, annualization_factor, and compounded follow UniverseProtocol.periods_per_year & selected_port."""
    fees = {"A": 0.024, "B": 0.012}
    dates = pd.date_range("2025-01-03", periods=40, freq=freq)
    rng = np.random.default_rng(7)
    rets = pd.DataFrame(
        rng.normal(0.001, 0.01, size=(len(dates), 2)),
        index=dates,
        columns=["A", "B"],
    )

    class _FreqProtocolUniverse:
        def __init__(self) -> None:
            self.tickers = ["A", "B"]
            self.metadata = {"A": "eq", "B": "eq"}
            self.fees = fees
            self.returns = rets
            self.periods_per_year = expected_ppy

    u = _FreqProtocolUniverse()
    oos_port = Portfolio(
        X=rets,
        weights=[0.8, 0.2],
        annualization_factor=float(expected_ppy),
        compounded=True,
        tag="FreqStrat",
    )
    pop = Population([oos_port])

    eng = PortfolioDeltaEngine(
        u,  # type: ignore[arg-type]
        current_weights={"A": 0.5, "B": 0.5},
        target_weights={"A": 0.8, "B": 0.2},
    )
    res = eng.calculate_rebalance_delta(pop, strategy_tag="FreqStrat")

    expected_hold = Portfolio(
        X=rets,
        weights=[0.5, 0.5],
        weight_drift=True,
        management_fees=np.array(
            [fees["A"] / float(expected_ppy), fees["B"] / float(expected_ppy)]
        ),
        annualization_factor=float(expected_ppy),
        compounded=True,
    )
    assert res["hold"]["sharpe"] == pytest.approx(
        float(expected_hold.annualized_sharpe_ratio), rel=0, abs=1e-12
    )
    assert res["hold"]["cvar"] == pytest.approx(
        float(expected_hold.cvar), rel=0, abs=1e-12
    )
    assert res["hold"]["max_drawdown"] == pytest.approx(
        float(expected_hold.max_drawdown), rel=0, abs=1e-12
    )


def test_calculate_net_friction_ignores_solver_dust_for_flat_brokerage(
    stub_universe: Universe,
) -> None:
    """F-11: Numerical solver dust (< 1e-6) must NOT trigger flat brokerage tickets."""
    # AAPL and MSFT differ by 5e-7 (< 1e-6 dust); GOOG untouched (0.0)
    engine_dust = PortfolioDeltaEngine(
        universe=stub_universe,
        current_weights={"AAPL": 0.5, "MSFT": 0.3, "GOOG": 0.2},
        target_weights={"AAPL": 0.5 + 5e-7, "MSFT": 0.3 - 5e-7, "GOOG": 0.2},
    )
    df_dust = engine_dust.calculate_net_friction(
        brokerage_bps=0.0,
        slippage_bps=0.0,
        brokerage_flat=25.0,
        portfolio_value=10_000.0,
    )
    assert df_dust[df_dust["ticker"] == "TOTAL"]["transaction_cost"].iloc[0] == 0.0

    # Weight diff 2e-6 (> 1e-6) DOES trigger flat brokerage on both traded legs
    engine_real = PortfolioDeltaEngine(
        universe=stub_universe,
        current_weights={"AAPL": 0.5, "MSFT": 0.3, "GOOG": 0.2},
        target_weights={"AAPL": 0.5 + 2e-6, "MSFT": 0.3 - 2e-6, "GOOG": 0.2},
    )
    df_real = engine_real.calculate_net_friction(
        brokerage_bps=0.0,
        slippage_bps=0.0,
        brokerage_flat=25.0,
        portfolio_value=10_000.0,
    )
    assert df_real[df_real["ticker"] == "TOTAL"]["transaction_cost"].iloc[
        0
    ] == pytest.approx(2.0 * (25.0 / 10_000.0), rel=0, abs=1e-12)


def test_delta_module_free_of_mock_name() -> None:
    """F-10: flowportfolio/delta.py must contain zero _mock_name branches."""
    source = inspect.getsource(delta_module)
    assert "_mock_name" not in source
