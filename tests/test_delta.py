from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from skfolio import Population

from flowportfolio.core.universe import Universe
from flowportfolio.delta import PortfolioDeltaEngine


@pytest.fixture
def stub_universe() -> Universe:
    u = MagicMock(spec=Universe)
    u.tickers = ["AAPL", "MSFT", "GOOG"]
    u.metadata = {"AAPL": "tech", "MSFT": "tech", "GOOG": "tech"}
    u.fees = {"AAPL": 0.001, "MSFT": 0.002, "GOOG": 0.0015}
    # Mock returns as a minimal DataFrame
    u.returns = pd.DataFrame(
        {"AAPL": [0.01, -0.01], "MSFT": [0.02, 0.0], "GOOG": [0.01, 0.01]}
    )
    return u


def test_init_raises_type_error(stub_universe):
    with pytest.raises(TypeError, match="current_weights must be a dict"):
        PortfolioDeltaEngine(
            universe=stub_universe,
            current_weights=[],
            target_weights={"AAPL": 1.0, "MSFT": 0.0, "GOOG": 0.0},
        )


def test_init_raises_type_error_non_string_keys(stub_universe):
    with pytest.raises(TypeError, match="current_weights keys must all be strings"):
        PortfolioDeltaEngine(
            universe=stub_universe,
            current_weights={1: 1.0, "MSFT": 0.0, "GOOG": 0.0},
            target_weights={"AAPL": 1.0, "MSFT": 0.0, "GOOG": 0.0},
        )


def test_init_raises_type_error_bool_values(stub_universe):
    with pytest.raises(TypeError, match="current_weights values must all be numeric"):
        PortfolioDeltaEngine(
            universe=stub_universe,
            current_weights={"AAPL": True, "MSFT": False, "GOOG": False},
            target_weights={"AAPL": 1.0, "MSFT": 0.0, "GOOG": 0.0},
        )


def test_init_raises_value_error_nan_values(stub_universe):
    with pytest.raises(
        ValueError, match="current_weights values must be finite numbers"
    ):
        PortfolioDeltaEngine(
            universe=stub_universe,
            current_weights={"AAPL": float("nan"), "MSFT": 0.0, "GOOG": 0.0},
            target_weights={"AAPL": 1.0, "MSFT": 0.0, "GOOG": 0.0},
        )


def test_init_raises_value_error_sum_current(stub_universe):
    with pytest.raises(ValueError, match="current_weights must sum to 1.0"):
        PortfolioDeltaEngine(
            universe=stub_universe,
            current_weights={"AAPL": 0.5, "MSFT": 0.0, "GOOG": 0.0},
            target_weights={"AAPL": 1.0, "MSFT": 0.0, "GOOG": 0.0},
        )


def test_init_raises_value_error_sum_target(stub_universe):
    with pytest.raises(ValueError, match="target_weights must sum to 1.0"):
        PortfolioDeltaEngine(
            universe=stub_universe,
            current_weights={"AAPL": 1.0, "MSFT": 0.0, "GOOG": 0.0},
            target_weights={"AAPL": 0.5, "MSFT": 0.0, "GOOG": 0.0},
        )


def test_init_raises_value_error_missing_current(stub_universe):
    with pytest.raises(
        ValueError, match="current_weights contains tickers not found in universe"
    ):
        PortfolioDeltaEngine(
            universe=stub_universe,
            current_weights={"TSLA": 1.0},
            target_weights={"AAPL": 1.0, "MSFT": 0.0, "GOOG": 0.0},
        )


def test_init_raises_value_error_missing_target(stub_universe):
    with pytest.raises(
        ValueError, match="target_weights contains tickers not found in universe"
    ):
        PortfolioDeltaEngine(
            universe=stub_universe,
            current_weights={"AAPL": 1.0, "MSFT": 0.0, "GOOG": 0.0},
            target_weights={"TSLA": 1.0},
        )


def test_init_raises_value_error_missing_universe_current(stub_universe):
    with pytest.raises(ValueError, match="current_weights is missing universe tickers"):
        PortfolioDeltaEngine(
            universe=stub_universe,
            current_weights={"AAPL": 1.0},
            target_weights={"AAPL": 1.0, "MSFT": 0.0, "GOOG": 0.0},
        )


def test_init_raises_value_error_missing_universe_target(stub_universe):
    with pytest.raises(ValueError, match="target_weights is missing universe tickers"):
        PortfolioDeltaEngine(
            universe=stub_universe,
            current_weights={"AAPL": 1.0, "MSFT": 0.0, "GOOG": 0.0},
            target_weights={"AAPL": 1.0},
        )


def test_calculate_group_drift(stub_universe):
    engine = PortfolioDeltaEngine(
        universe=stub_universe,
        current_weights={"AAPL": 0.6, "MSFT": 0.4, "GOOG": 0.0},
        target_weights={"AAPL": 0.5, "MSFT": 0.3, "GOOG": 0.2},
    )
    df = engine.calculate_group_drift()

    assert "tech" in df.index
    # All are tech, so current = 1.0, target = 1.0, drift = 0.0
    assert df.loc["tech", "current_allocation"] == 1.0
    assert df.loc["tech", "target_allocation"] == 1.0
    assert df.loc["tech", "drift"] == 0.0
    assert df.loc["tech", "abs_drift"] == 0.0


def test_calculate_net_friction(stub_universe):
    engine = PortfolioDeltaEngine(
        universe=stub_universe,
        current_weights={"AAPL": 0.6, "MSFT": 0.4, "GOOG": 0.0},
        target_weights={"AAPL": 0.5, "MSFT": 0.3, "GOOG": 0.2},
    )
    df = engine.calculate_net_friction(brokerage_bps=10.0, slippage_bps=5.0)

    # Check shape
    assert "gross_weight_traded" in df.columns
    assert "transaction_cost" in df.columns

    # AAPL turnover = 0.1, MSFT = 0.1, GOOG = 0.2
    # Total turnover = 0.4
    total_turnover = df[df["ticker"] == "TOTAL"]["gross_weight_traded"].iloc[0]
    assert np.isclose(total_turnover, 0.4)


def test_calculate_net_friction_raises_value_error(stub_universe):
    engine = PortfolioDeltaEngine(
        universe=stub_universe,
        current_weights={"AAPL": 1.0, "MSFT": 0.0, "GOOG": 0.0},
        target_weights={"AAPL": 1.0, "MSFT": 0.0, "GOOG": 0.0},
    )
    with pytest.raises(ValueError, match="brokerage_bps must be non-negative"):
        engine.calculate_net_friction(brokerage_bps=-5.0)


@patch("flowportfolio.delta.Portfolio")
def test_calculate_rebalance_delta(mock_portfolio, stub_universe):
    engine = PortfolioDeltaEngine(
        universe=stub_universe,
        current_weights={"AAPL": 0.6, "MSFT": 0.4, "GOOG": 0.0},
        target_weights={"AAPL": 0.5, "MSFT": 0.3, "GOOG": 0.2},
    )

    mock_hold = MagicMock()
    mock_hold.cvar = 0.05
    mock_hold.sharpe_ratio = 1.5
    mock_hold.max_drawdown = 0.1

    mock_rebalance = MagicMock()
    mock_rebalance.cvar = 0.04
    mock_rebalance.sharpe_ratio = 2.0
    mock_rebalance.max_drawdown = 0.08

    mock_portfolio.side_effect = [mock_hold, mock_rebalance]

    mock_pop = MagicMock(spec=Population)
    mock_pop.__len__.return_value = 1
    mock_item = MagicMock()
    mock_item.returns_df = None
    mock_item.tag = None
    mock_pop.__getitem__.return_value = mock_item

    res = engine.calculate_rebalance_delta(mock_pop)

    assert "hold" in res
    assert "rebalance" in res
    assert res["hold"]["cvar"] == 0.05
    assert res["rebalance"]["sharpe"] == 2.0
    assert mock_portfolio.call_count == 2


def test_delta_rejects_short_weights_and_applies_fees_to_hold():
    u = MagicMock(spec=Universe)
    u.tickers = ["SPY", "TLT"]
    u.metadata = {"SPY": "equity", "TLT": "bonds"}
    u.fees = {"SPY": 0.0003, "TLT": 0.0015}
    u.returns = pd.DataFrame({"SPY": [0.01, -0.01, 0.02], "TLT": [-0.005, 0.01, 0.0]})

    with pytest.raises(ValueError, match="Weight must be between 0.0 and 1.0"):
        PortfolioDeltaEngine(u, {"SPY": 1.2, "TLT": -0.2}, {"SPY": 0.5, "TLT": 0.5})

    engine = PortfolioDeltaEngine(u, {"SPY": 0.6, "TLT": 0.4}, {"SPY": 0.5, "TLT": 0.5})
    with patch("flowportfolio.delta.Portfolio") as mock_port:
        mock_p = MagicMock()
        mock_p.cvar = 0.02
        mock_p.sharpe_ratio = 1.0
        mock_p.max_drawdown = 0.05
        mock_port.return_value = mock_p

        engine.calculate_rebalance_delta()
        # Verify hold portfolio was called with management_fees matching fees / 252.0
        assert mock_port.call_count >= 1
        _, first_kwargs = mock_port.call_args_list[0]
        assert "management_fees" in first_kwargs
        expected_fees = np.array([0.0003 / 252.0, 0.0015 / 252.0])
        np.testing.assert_allclose(first_kwargs["management_fees"], expected_fees)


def test_calculate_rebalance_delta_multipath_window_alignment():
    idx = pd.date_range("2024-01-01", periods=30)
    rng = np.random.default_rng(7)
    returns = pd.DataFrame(
        rng.normal(0.0005, 0.01, (30, 2)), columns=["SPY", "TLT"], index=idx
    )
    u = Universe(
        ["SPY", "TLT"], {"SPY": "equity", "TLT": "bonds"}, {"SPY": 0.001, "TLT": 0.002}
    )
    u._returns = returns

    from skfolio import Portfolio

    p1 = Portfolio(X=returns.iloc[:15], weights=[0.5, 0.5], tag="CPCV")
    p2 = Portfolio(X=returns.iloc[15:], weights=[0.5, 0.5], tag="CPCV")
    pop = Population([p1, p2])

    engine = PortfolioDeltaEngine(u, {"SPY": 0.7, "TLT": 0.3}, {"SPY": 0.5, "TLT": 0.5})
    res = engine.calculate_rebalance_delta(pop, strategy_tag="CPCV")

    fees_vec = np.array([0.001 / 252.0, 0.002 / 252.0])
    h1 = Portfolio(
        X=returns.iloc[:15],
        weights=[0.7, 0.3],
        weight_drift=True,
        management_fees=fees_vec,
    )
    h2 = Portfolio(
        X=returns.iloc[15:],
        weights=[0.7, 0.3],
        weight_drift=True,
        management_fees=fees_vec,
    )
    r1 = Portfolio(
        X=returns.iloc[:15],
        weights=[0.5, 0.5],
        weight_drift=True,
        management_fees=fees_vec,
    )
    r2 = Portfolio(
        X=returns.iloc[15:],
        weights=[0.5, 0.5],
        weight_drift=True,
        management_fees=fees_vec,
    )

    assert np.isclose(res["hold"]["cvar"], np.median([h1.cvar, h2.cvar]))
    assert np.isclose(
        res["hold"]["sharpe"], np.median([h1.annualized_sharpe_ratio, h2.annualized_sharpe_ratio])
    )
    assert np.isclose(res["rebalance"]["cvar"], np.median([r1.cvar, r2.cvar]))
    assert np.isclose(
        res["rebalance"]["sharpe"], np.median([r1.annualized_sharpe_ratio, r2.annualized_sharpe_ratio])
    )


def test_calculate_rebalance_delta_symmetric_when_hold_equals_target(stub_universe):
    """Verify that when current_weights == target_weights, hold and rebalance metrics are identical."""
    w = {"AAPL": 0.5, "MSFT": 0.3, "GOOG": 0.2}
    engine = PortfolioDeltaEngine(universe=stub_universe, current_weights=w, target_weights=w)
    res = engine.calculate_rebalance_delta()

    assert np.isclose(res["hold"]["cvar"], res["rebalance"]["cvar"])
    assert np.isclose(res["hold"]["sharpe"], res["rebalance"]["sharpe"])
    assert np.isclose(res["hold"]["max_drawdown"], res["rebalance"]["max_drawdown"])


def test_calculate_net_friction_with_portfolio_value_and_flat_brokerage(stub_universe):
    """Verify flat brokerage drag calculation and parameter validations."""
    engine = PortfolioDeltaEngine(
        universe=stub_universe,
        current_weights={"AAPL": 0.6, "MSFT": 0.4, "GOOG": 0.0},
        target_weights={"AAPL": 0.5, "MSFT": 0.3, "GOOG": 0.2},
    )
    # Traded assets: AAPL (0.1 traded), MSFT (0.1 traded), GOOG (0.2 traded) -> 3 traded assets
    df = engine.calculate_net_friction(
        portfolio_value=100000.0,
        brokerage_flat=10.0,
        brokerage_bps=10.0,
        slippage_bps=5.0,
    )

    # Flat fee drag per traded asset: 10 / 100_000 = 0.0001
    # For AAPL: 0.1 * 15 / 10000 + 0.0001 = 0.00015 + 0.0001 = 0.00025
    aapl_cost = df[df["ticker"] == "AAPL"]["transaction_cost"].iloc[0]
    assert np.isclose(aapl_cost, 0.00025)

    # Validation: flat fee > 0 without portfolio_value
    with pytest.raises(ValueError, match="portfolio_value must be provided when brokerage_flat > 0"):
        engine.calculate_net_friction(brokerage_flat=10.0, portfolio_value=None)

    # Validation: invalid portfolio_value
    with pytest.raises(ValueError, match="portfolio_value must be a positive finite number"):
        engine.calculate_net_friction(brokerage_flat=10.0, portfolio_value=-1000.0)

