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
    assert "turnover" in df.columns
    assert "transaction_cost" in df.columns

    # AAPL turnover = 0.1, MSFT = 0.1, GOOG = 0.2
    # Total turnover = 0.4
    total_turnover = df[df["ticker"] == "TOTAL"]["turnover"].iloc[0]
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
    mock_portfolio.return_value = mock_hold

    mock_pop = MagicMock(spec=Population)
    mock_pop.__len__.return_value = 1

    mock_best = MagicMock()
    mock_best.cvar = 0.04
    mock_best.sharpe_ratio = 2.0
    mock_best.max_drawdown = 0.08
    mock_pop.max_measure.return_value = mock_best

    res = engine.calculate_rebalance_delta(mock_pop)

    assert "hold" in res
    assert "rebalance" in res
    assert res["hold"]["cvar"] == 0.05
    assert res["rebalance"]["sharpe"] == 2.0
