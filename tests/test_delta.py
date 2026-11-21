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

    fees_vec = np.array([0.001 / 252.0, 0.002 / 252.0])
    # Distinct realized OOS portfolios (different weights from target_weights!)
    p1 = Portfolio(
        X=returns.iloc[:15], weights=[0.9, 0.1], management_fees=fees_vec, tag="CPCV"
    )
    p2 = Portfolio(
        X=returns.iloc[15:], weights=[0.2, 0.8], management_fees=fees_vec, tag="CPCV"
    )
    pop = Population([p1, p2])

    engine = PortfolioDeltaEngine(u, {"SPY": 0.7, "TLT": 0.3}, {"SPY": 0.5, "TLT": 0.5})
    res = engine.calculate_rebalance_delta(pop, strategy_tag="CPCV")

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

    # Exact one-off transition drag = 1.2 * 75 / 10_000 + 2 * (50 / 10_000) = 0.009 + 0.010 = 0.019
    expected_drag = 1.2 * 75.0 / 10_000.0 + 2.0 * (50.0 / 10_000.0)
    net_series = np.asarray(oos.returns, dtype=float).copy() - expected_drag / len(
        oos.returns
    )
    expected_net_port = Portfolio(
        X=net_series[:, None],
        weights=np.array([1.0]),
        annualization_factor=252.0,
    )

    assert res_net["rebalance"]["sharpe"] < res_gross["rebalance"]["sharpe"]
    assert res_net["rebalance"]["sharpe"] == pytest.approx(
        float(expected_net_port.annualized_sharpe_ratio), rel=0, abs=1e-12
    )
    # One-off day-0 execution fee must NOT corrupt steady-state daily CVaR or max_drawdown (#5c)
    assert res_net["rebalance"]["cvar"] == pytest.approx(
        res_gross["rebalance"]["cvar"], rel=0, abs=1e-12
    )
    assert res_net["rebalance"]["max_drawdown"] == pytest.approx(
        res_gross["rebalance"]["max_drawdown"], rel=0, abs=1e-12
    )
    assert res_net["rebalance"]["entry_drag"] == pytest.approx(
        expected_drag, rel=0, abs=1e-12
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


# ---------------------------------------------------------------------------
# Cycle 15 (P1-4 & P1-5): Fee Symmetry, Fold-0 Transition Drag & Double-Count Guard
# ---------------------------------------------------------------------------


def test_rebalance_delta_enforces_fee_symmetry_without_double_charging(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """P1-4: Fee-free OOS portfolios deduct symmetric period-scaled fees; fee-bearing ones are never double-charged."""
    import logging

    rng = np.random.default_rng(31)
    idx = pd.bdate_range("2025-01-02", periods=60)
    rets = pd.DataFrame(
        {"A": rng.normal(0.0008, 0.01, 60), "B": rng.normal(0.0004, 0.01, 60)},
        index=idx,
    )
    fees = {"A": 0.010, "B": 0.020}
    mf_vec = np.array([fees["A"] / 252.0, fees["B"] / 252.0])

    u = Universe(["A", "B"], {"A": "eq", "B": "eq"}, fees)
    u._returns = rets
    w = {"A": 0.6, "B": 0.4}
    eng = PortfolioDeltaEngine(u, current_weights=w, target_weights=w)

    # 1. Fee-free OOS portfolio (e.g., plain MeanRisk without ConstraintBuilder)
    gross_fold1 = Portfolio(
        X=rets.iloc[:30], weights=[0.6, 0.4], weight_drift=True, tag="Gross"
    )
    gross_fold2 = Portfolio(
        X=rets.iloc[30:], weights=[0.6, 0.4], weight_drift=True, tag="Gross"
    )
    gross_mpp = MultiPeriodPortfolio(portfolios=[gross_fold1, gross_fold2], tag="Gross")
    # Exact fee-bearing oracle over the same folds
    net_fold1 = Portfolio(
        X=rets.iloc[:30],
        weights=[0.6, 0.4],
        weight_drift=True,
        management_fees=mf_vec,
        tag="Net",
    )
    net_fold2 = Portfolio(
        X=rets.iloc[30:],
        weights=[0.6, 0.4],
        weight_drift=True,
        management_fees=mf_vec,
        tag="Net",
    )
    net_mpp = MultiPeriodPortfolio(portfolios=[net_fold1, net_fold2], tag="Net")

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="flowportfolio.delta"):
        res_gross = eng.calculate_rebalance_delta(
            Population([gross_mpp]), strategy_tag="Gross"
        )
    assert any("management fees" in rec.message.lower() for rec in caplog.records)
    assert res_gross["rebalance"]["sharpe"] < float(gross_mpp.annualized_sharpe_ratio)
    assert res_gross["rebalance"]["sharpe"] == pytest.approx(
        float(net_mpp.annualized_sharpe_ratio), rel=1e-9, abs=1e-12
    )
    assert res_gross["rebalance"]["cvar"] == pytest.approx(
        float(net_mpp.cvar), rel=1e-9, abs=1e-12
    )
    assert res_gross["rebalance"]["max_drawdown"] == pytest.approx(
        float(net_mpp.max_drawdown), rel=1e-9, abs=1e-12
    )

    # Single-period identical-weights symmetry check: hold == rebalance (zero bias)
    gross_single = Portfolio(X=rets, weights=[0.6, 0.4], weight_drift=True, tag="Same")
    res_sym = eng.calculate_rebalance_delta(
        Population([gross_single]), strategy_tag="Same"
    )
    assert res_sym["rebalance"]["sharpe"] == pytest.approx(
        res_sym["hold"]["sharpe"], rel=1e-9, abs=1e-12
    )

    # 2. Already fee-bearing OOS portfolio must NOT be charged fees a second time
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="flowportfolio.delta"):
        res_net = eng.calculate_rebalance_delta(
            Population([net_mpp]), strategy_tag="Net"
        )
    assert not any("management fees" in rec.message.lower() for rec in caplog.records)
    assert res_net["rebalance"]["sharpe"] == pytest.approx(
        float(net_mpp.annualized_sharpe_ratio), rel=1e-9, abs=1e-12
    )
    assert res_net["rebalance"]["cvar"] == pytest.approx(
        float(net_mpp.cvar), rel=1e-9, abs=1e-12
    )


def test_rebalance_delta_transition_drag_uses_fold0_trade_not_terminal_target() -> None:
    """Sprint 5 (#5a & #5c): Charges Fold-0 entry drag from current->fold0 plus Fold-1..N recurring rebalance drags."""
    rng = np.random.default_rng(44)
    idx = pd.bdate_range("2025-01-02", periods=60)
    rets = pd.DataFrame(
        {"A": rng.normal(0.001, 0.01, 60), "B": rng.normal(0.0005, 0.01, 60)},
        index=idx,
    )
    # Fold 0 holds [0.9, 0.1] (weight_drift=False); Fold 1 holds [0.2, 0.8]; live terminal target is [0.1, 0.9]
    f0 = Portfolio(X=rets.iloc[:30], weights=[0.9, 0.1], tag="Strat")
    f1 = Portfolio(X=rets.iloc[30:], weights=[0.2, 0.8], tag="Strat")
    mpp = MultiPeriodPortfolio(portfolios=[f0, f1], tag="Strat")
    pop = Population([mpp])

    u = Universe(["A", "B"], {"A": "eq", "B": "eq"}, {"A": 0.0, "B": 0.0})
    u._returns = rets

    # Case 1: current_weights == fold0_weights -> entry_drag == 0.0, while Fold 1 rebalance (L1=1.4) IS charged (#5a)
    eng_zero_entry = PortfolioDeltaEngine(
        u,
        current_weights={"A": 0.9, "B": 0.1},
        target_weights={"A": 0.1, "B": 0.9},
    )
    res_gross = eng_zero_entry.calculate_rebalance_delta(pop, strategy_tag="Strat")
    res_with_bps = eng_zero_entry.calculate_rebalance_delta(
        pop,
        strategy_tag="Strat",
        brokerage_bps=50.0,
        slippage_bps=25.0,
        brokerage_flat=50.0,
        portfolio_value=10_000.0,
    )
    # Fold 1 rebalance: L1(|0.2-0.9|+|0.8-0.1|) = 1.4 @ 75 bps + 2 flat tickets ($100/$10k = 0.01) = 0.0205
    expected_f1_drag = 1.4 * 75.0 / 10_000.0 + 2.0 * (50.0 / 10_000.0)
    assert res_with_bps["rebalance"]["entry_drag"] == pytest.approx(0.0, abs=1e-12)
    assert res_with_bps["rebalance"]["rebalance_drag"] == pytest.approx(
        expected_f1_drag, abs=1e-12
    )
    assert res_with_bps["rebalance"]["sharpe"] < res_gross["rebalance"]["sharpe"]

    # Case 2: current_weights = [0.7, 0.3] -> entry_drag uses current->fold0 (L1=0.4), Fold 1 uses fold0->fold1 (L1=1.4)
    eng_asym = PortfolioDeltaEngine(
        u,
        current_weights={"A": 0.7, "B": 0.3},
        target_weights={"A": 0.1, "B": 0.9},
    )
    res_asym = eng_asym.calculate_rebalance_delta(
        pop, strategy_tag="Strat", brokerage_bps=50.0, slippage_bps=50.0
    )
    expected_entry_drag = 0.4 * 100.0 / 10_000.0  # 0.004
    expected_reb_drag = 1.4 * 100.0 / 10_000.0  # 0.014
    assert res_asym["rebalance"]["entry_drag"] == pytest.approx(
        expected_entry_drag, abs=1e-12
    )
    assert res_asym["rebalance"]["rebalance_drag"] == pytest.approx(
        expected_reb_drag, abs=1e-12
    )

    steady_series = np.concatenate(
        [
            np.asarray(f0.returns, dtype=float),
            np.asarray(f1.returns, dtype=float) - expected_reb_drag / len(f1.returns),
        ]
    )
    steady_port = Portfolio(
        X=steady_series[:, None], weights=np.array([1.0]), annualization_factor=252.0
    )
    net_series = steady_series - expected_entry_drag / len(steady_series)
    net_port = Portfolio(
        X=net_series[:, None], weights=np.array([1.0]), annualization_factor=252.0
    )
    assert res_asym["rebalance"]["cvar"] == pytest.approx(
        float(steady_port.cvar), rel=1e-9, abs=1e-12
    )
    assert res_asym["rebalance"]["sharpe"] == pytest.approx(
        float(net_port.annualized_sharpe_ratio), rel=1e-9, abs=1e-12
    )


def test_rebalance_delta_skips_transition_drag_when_oos_already_embeds_transaction_costs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Sprint 5 (#5b): Skip external Fold-0 drag ONLY when Fold 0 itself embeds costs; charge Fold 0 when neutralized."""
    import logging

    rng = np.random.default_rng(51)
    idx = pd.bdate_range("2025-01-02", periods=60)
    rets = pd.DataFrame(
        {"A": rng.normal(0.001, 0.01, 60), "B": rng.normal(0.0005, 0.01, 60)},
        index=idx,
    )
    # 1. Both Fold 0 and Fold 1 embed transaction costs -> external drag completely skipped with warning
    f0_costed = Portfolio(
        X=rets.iloc[:30],
        weights=[0.8, 0.2],
        previous_weights=np.array([0.5, 0.5]),
        transaction_costs=0.001 / 30.0,
        tag="AllCosted",
    )
    f1_costed = Portfolio(
        X=rets.iloc[30:],
        weights=[0.2, 0.8],
        previous_weights=np.array([0.8, 0.2]),
        transaction_costs=0.001 / 30.0,
        tag="AllCosted",
    )
    assert float(f0_costed.total_cost) > 0.0
    assert float(f1_costed.total_cost) > 0.0
    mpp_all = MultiPeriodPortfolio(portfolios=[f0_costed, f1_costed], tag="AllCosted")
    pop_all = Population([mpp_all])

    u = Universe(["A", "B"], {"A": "eq", "B": "eq"}, {"A": 0.0, "B": 0.0})
    u._returns = rets
    eng = PortfolioDeltaEngine(
        u,
        current_weights={"A": 0.2, "B": 0.8},
        target_weights={"A": 0.8, "B": 0.2},
    )

    res_no_extra = eng.calculate_rebalance_delta(pop_all, strategy_tag="AllCosted")
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="flowportfolio.delta"):
        res_with_extra = eng.calculate_rebalance_delta(
            pop_all,
            strategy_tag="AllCosted",
            brokerage_bps=50.0,
            slippage_bps=25.0,
        )

    assert any("double-counting" in rec.message.lower() for rec in caplog.records)
    assert res_with_extra["rebalance"]["sharpe"] == pytest.approx(
        res_no_extra["rebalance"]["sharpe"], rel=0, abs=1e-12
    )
    assert res_with_extra["rebalance"]["cvar"] == pytest.approx(
        res_no_extra["rebalance"]["cvar"], rel=0, abs=1e-12
    )

    # 2. Neutralized Fold 0 (total_cost == 0.0 via entry_rebalancing_params) + costed Fold 1:
    # Fold-0 entry drag from current_weights -> f0 IS charged, Fold 1 is NOT double-charged (#5b)
    f0_neutral = Portfolio(
        X=rets.iloc[:30],
        weights=[0.8, 0.2],
        transaction_costs=0.0,
        tag="NeutralF0",
    )
    f1_paid = Portfolio(
        X=rets.iloc[30:],
        weights=[0.2, 0.8],
        previous_weights=np.array([0.8, 0.2]),
        transaction_costs=0.001 / 30.0,
        tag="NeutralF0",
    )
    assert float(f0_neutral.total_cost) == 0.0
    mpp_neutral = MultiPeriodPortfolio(
        portfolios=[f0_neutral, f1_paid], tag="NeutralF0"
    )
    res_neutral_gross = eng.calculate_rebalance_delta(
        Population([mpp_neutral]), strategy_tag="NeutralF0"
    )
    res_neutral_entry = eng.calculate_rebalance_delta(
        Population([mpp_neutral]),
        strategy_tag="NeutralF0",
        brokerage_bps=50.0,
        slippage_bps=25.0,
    )
    expected_entry = 1.2 * 75.0 / 10_000.0  # |0.8-0.2| + |0.2-0.8| = 1.2
    assert res_neutral_entry["rebalance"]["entry_drag"] == pytest.approx(
        expected_entry, abs=1e-12
    )
    assert res_neutral_entry["rebalance"]["rebalance_drag"] == pytest.approx(
        0.0, abs=1e-12
    )
    assert (
        res_neutral_entry["rebalance"]["sharpe"]
        < res_neutral_gross["rebalance"]["sharpe"]
    )


def test_rebalance_delta_rejects_randomized_asset_subset_paths() -> None:
    """Sprint 5 (N5): Strictly subsetted OOS paths (MultipleRandomizedCV) must raise ValueError."""
    rng = np.random.default_rng(77)
    idx = pd.bdate_range("2025-01-02", periods=60)
    rets = pd.DataFrame(
        rng.normal(0.0005, 0.01, (60, 4)),
        index=idx,
        columns=["A", "B", "C", "D"],
    )
    u = Universe(
        ["A", "B", "C", "D"],
        {"A": "core", "B": "core", "C": "sat", "D": "sat"},
        {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0},
    )
    u._returns = rets

    # Subsetted portfolios (e.g., MultipleRandomizedCV with asset_subset_size=2)
    p_sub = Portfolio(X=rets[["B", "C"]], weights=[0.5, 0.5], tag="RandSub")
    pop = Population([MultiPeriodPortfolio(portfolios=[p_sub], tag="RandSub")])
    cw = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
    eng = PortfolioDeltaEngine(u, current_weights=cw, target_weights=cw)

    with pytest.raises(ValueError, match="asset-subsampled"):
        eng.calculate_rebalance_delta(pop, strategy_tag="RandSub")


def test_rebalance_delta_allow_asset_subsets_supports_direct_preselection_pipeline() -> (
    None
):
    """Sprint 5 (S6): Direct pre-selection Pipeline subsets fail safely by default and work with allow_asset_subsets=True."""
    from skfolio.model_selection import WalkForward, cross_val_predict
    from skfolio.optimization import MeanRisk
    from skfolio.pre_selection import DropCorrelated

    from flowportfolio.core.constraints import ConstraintBuilder
    from flowportfolio.strategies import StrategyBuilder

    rng = np.random.default_rng(88)
    idx = pd.bdate_range("2022-01-03", periods=200)
    base = rng.normal(0.0005, 0.01, (200, 1))
    rets = pd.DataFrame(
        np.hstack(
            [
                base + rng.normal(0.0, 0.001, (200, 1)),
                base + rng.normal(0.0, 0.001, (200, 1)),
                rng.normal(0.0004, 0.01, (200, 1)),
                rng.normal(0.0003, 0.01, (200, 1)),
            ]
        ),
        index=idx,
        columns=["A", "B", "C", "D"],
    )
    u = Universe(
        ["A", "B", "C", "D"],
        {"A": "core", "B": "core", "C": "sat", "D": "sat"},
        {"A": 0.001, "B": 0.001, "C": 0.001, "D": 0.001},
    )
    u._returns = rets
    pipe = (
        StrategyBuilder(constraints=ConstraintBuilder(u).build())
        .add_pre_selection(DropCorrelated(threshold=0.5))
        .set_optimizer(MeanRisk())
        .build_pipeline()
    )
    mpp = cross_val_predict(pipe, rets, cv=WalkForward(train_size=100, test_size=50))
    mpp.tag = "DirectPre"
    cw = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
    eng = PortfolioDeltaEngine(u, current_weights=cw, target_weights=cw)

    # Default remains fail-fast with guidance mentioning allow_asset_subsets
    with pytest.raises(ValueError, match="allow_asset_subsets"):
        eng.calculate_rebalance_delta(Population([mpp]), strategy_tag="DirectPre")

    # Explicit opt-in succeeds and prices dropped ticker liquidation as 0% weight
    res = eng.calculate_rebalance_delta(
        Population([mpp]),
        strategy_tag="DirectPre",
        brokerage_bps=20.0,
        allow_asset_subsets=True,
    )
    assert np.isfinite(res["rebalance"]["sharpe"])
    assert res["rebalance"]["entry_drag"] > 0.0
