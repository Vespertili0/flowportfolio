"""Unit tests for the PortfolioExperimentEngine class.

This module verifies the strategy registration, hyperparameter tuning, and
cross-validation execution logic in :class:`flowportfolio.core.experiment.PortfolioExperimentEngine`.
All heavy compute and scikit-learn fitting are mocked to ensure fast, offline tests.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from skfolio.model_selection import (
    CombinatorialPurgedCV,
    MultipleRandomizedCV,
    WalkForward,
)

from flowportfolio.core.experiment import PortfolioExperimentEngine
from flowportfolio.core.universe import Universe

# ---------------------------------------------------------------------------
# Shared fixtures and stubs
# ---------------------------------------------------------------------------

class DummyEstimator(BaseEstimator):
    """A minimal valid estimator for testing type assertions."""
    def fit(self, X, y=None):
        return self
        
    def predict(self, X):
        return X

@pytest.fixture
def stub_universe() -> Universe:
    """Provide a minimal Universe stub with synthetic returns."""
    tickers = ["A", "B"]
    metadata = {"A": "core", "B": "satellite"}
    fees = {"A": 0.001, "B": 0.002}
    universe = Universe(tickers=tickers, metadata=metadata, fees=fees)
    
    # Inject synthetic returns to bypass yfinance fetch
    df = pd.DataFrame(
        {"A": [0.01, 0.02, -0.01], "B": [-0.01, 0.01, 0.02]},
        index=pd.date_range("2026-01-01", periods=3),
    )
    universe._returns = df
    return universe

@pytest.fixture
def stub_universe_no_returns() -> Universe:
    """Provide a Universe stub that has not fetched data."""
    return Universe(
        tickers=["A"], metadata={"A": "core"}, fees={"A": 0.0}
    )

# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def test_init_valid(stub_universe: Universe) -> None:
    """Test engine accepts valid Universe and list of constraints."""
    engine = PortfolioExperimentEngine(stub_universe, ["core >= 0.5"])
    assert engine._universe is stub_universe
    assert engine._constraints == ["core >= 0.5"]
    assert engine._strategies == {}
    assert engine._n_jobs == -1


def test_init_bad_universe() -> None:
    """Test TypeError is raised for non-Universe first arg."""
    with pytest.raises(TypeError, match="universe must be a Universe instance"):
        PortfolioExperimentEngine(universe="invalid", constraints=[])  # type: ignore


def test_init_bad_constraints(stub_universe: Universe) -> None:
    """Test TypeError is raised for non-list constraints."""
    with pytest.raises(TypeError, match="constraints must be a list"):
        PortfolioExperimentEngine(stub_universe, constraints="core >= 0.5")  # type: ignore

# ---------------------------------------------------------------------------
# Strategy Registration
# ---------------------------------------------------------------------------

def test_add_strategy_valid(stub_universe: Universe) -> None:
    """Test valid strategy is stored correctly."""
    engine = PortfolioExperimentEngine(stub_universe, [])
    est = DummyEstimator()
    grid = {"param": [1, 2]}
    
    engine.add_strategy("MyStrat", est, grid)
    assert "MyStrat" in engine._strategies
    assert engine._strategies["MyStrat"]["estimator"] is est
    assert engine._strategies["MyStrat"]["grid"] == grid


def test_add_strategy_duplicate_name(stub_universe: Universe) -> None:
    """Test ValueError is raised on duplicate registration."""
    engine = PortfolioExperimentEngine(stub_universe, [])
    engine.add_strategy("MyStrat", DummyEstimator(), {})
    
    with pytest.raises(ValueError, match="already registered"):
        engine.add_strategy("MyStrat", DummyEstimator(), {})


def test_add_strategy_bad_estimator(stub_universe: Universe) -> None:
    """Test TypeError is raised for non-BaseEstimator."""
    engine = PortfolioExperimentEngine(stub_universe, [])
    with pytest.raises(TypeError, match="scikit-learn BaseEstimator"):
        engine.add_strategy("MyStrat", estimator="not_an_estimator", grid={})  # type: ignore


def test_add_strategy_bad_grid(stub_universe: Universe) -> None:
    """Test TypeError is raised for non-dict grid."""
    engine = PortfolioExperimentEngine(stub_universe, [])
    with pytest.raises(TypeError, match="grid must be a dictionary"):
        engine.add_strategy("MyStrat", DummyEstimator(), grid=["not", "dict"])  # type: ignore

# ---------------------------------------------------------------------------
# Execution (Robustness Test)
# ---------------------------------------------------------------------------

@patch("flowportfolio.core.experiment.Population")
@patch("flowportfolio.core.experiment.cross_val_predict")
@patch("flowportfolio.core.experiment.GridSearchCV")
def test_run_robustness_test_walk_forward(
    mock_gscv: MagicMock,
    mock_cv_predict: MagicMock,
    mock_population: MagicMock,
    stub_universe: Universe
) -> None:
    """Test execution with WalkForward cross-validator."""
    # Setup mocks
    mock_gscv_instance = MagicMock()
    mock_gscv_instance.best_estimator_ = DummyEstimator()
    mock_gscv.return_value = mock_gscv_instance
    
    mock_portfolio = MagicMock()
    mock_portfolio.tag = "MyStrat"
    mock_cv_predict.return_value = mock_portfolio
    
    mock_pop_instance = MagicMock()
    mock_population.return_value = mock_pop_instance

    # Setup engine
    engine = PortfolioExperimentEngine(stub_universe, [])
    engine.add_strategy("MyStrat", DummyEstimator(), {"p": [1]})
    
    # Run
    population = engine.run_robustness_test(cv_type="walk_forward", train_size=252, test_size=63)
    
    # Verify CV object type and instantiation
    call_args = mock_gscv.call_args[1]
    assert isinstance(call_args["cv"], WalkForward)
    assert call_args["cv"].train_size == 252
    assert call_args["cv"].test_size == 63
    
    # Verify fit and predict were called
    mock_gscv_instance.fit.assert_called_once_with(stub_universe.returns)
    mock_cv_predict.assert_called_once()
    
    # Verify tag injection and output
    predict_kwargs = mock_cv_predict.call_args[1]
    assert predict_kwargs["portfolio_params"] == {"tag": "MyStrat"}
    
    # Verify Population was called with collected mock
    mock_population.assert_called_once_with([mock_portfolio])
    assert population is mock_pop_instance


@patch("flowportfolio.core.experiment.Population")
@patch("flowportfolio.core.experiment.cross_val_predict")
@patch("flowportfolio.core.experiment.GridSearchCV")
def test_run_robustness_test_combinatorial(
    mock_gscv: MagicMock,
    mock_cv_predict: MagicMock,
    mock_population: MagicMock,
    stub_universe: Universe
) -> None:
    """Test execution with CombinatorialPurgedCV."""
    engine = PortfolioExperimentEngine(stub_universe, [])
    engine.add_strategy("Strat", DummyEstimator(), {})
    engine.run_robustness_test(cv_type="combinatorial", n_folds=5, n_test_folds=2)
    
    call_args = mock_gscv.call_args[1]
    assert isinstance(call_args["cv"], CombinatorialPurgedCV)


@patch("flowportfolio.core.experiment.Population")
@patch("flowportfolio.core.experiment.cross_val_predict")
@patch("flowportfolio.core.experiment.GridSearchCV")
def test_run_robustness_test_randomised(
    mock_gscv: MagicMock,
    mock_cv_predict: MagicMock,
    mock_population: MagicMock,
    stub_universe: Universe
) -> None:
    """Test execution with MultipleRandomizedCV."""
    engine = PortfolioExperimentEngine(stub_universe, [])
    engine.add_strategy("Strat", DummyEstimator(), {})
    # Need a base CV for randomized
    base_cv = WalkForward(train_size=252, test_size=63)
    engine.run_robustness_test(cv_type="randomised", n_subsamples=10, walk_forward=base_cv, asset_subset_size=2)
    
    call_args = mock_gscv.call_args[1]
    assert isinstance(call_args["cv"], MultipleRandomizedCV)


def test_run_robustness_test_unknown_cv_type(stub_universe: Universe) -> None:
    """Test ValueError is raised for unsupported CV types."""
    engine = PortfolioExperimentEngine(stub_universe, [])
    with pytest.raises(ValueError, match="Unknown cv_type"):
        engine.run_robustness_test(cv_type="unknown_cv")


def test_run_robustness_test_no_fetch(stub_universe_no_returns: Universe) -> None:
    """Test engine propagates ValueError if returns are not available."""
    engine = PortfolioExperimentEngine(stub_universe_no_returns, [])
    engine.add_strategy("Strat", DummyEstimator(), {})
    with pytest.raises(ValueError, match="Returns are not yet available"):
        engine.run_robustness_test(train_size=252, test_size=63)
