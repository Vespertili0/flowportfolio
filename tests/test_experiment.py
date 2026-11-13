"""Unit tests for the PortfolioExperimentEngine class.

This module verifies the strategy registration, hyperparameter tuning, and
cross-validation execution logic in :class:`flowportfolio.core.experiment.PortfolioExperimentEngine`.
All heavy compute and scikit-learn fitting are mocked to ensure fast, offline tests.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from skfolio.model_selection import (
    CombinatorialPurgedCV,
    MultipleRandomizedCV,
    WalkForward,
)
from sklearn.base import BaseEstimator

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


class EWMu:
    def partial_fit(self, X, y=None):
        return self


class EWCovariance:
    def partial_fit(self, X, y=None):
        return self


class DummyOnlineEstimator(BaseEstimator):
    """Estimator satisfying online estimation contract."""

    def __init__(self, mu=None, cov=None):
        self.mu = mu
        self.cov = cov
        self.mu_estimator = mu if mu is not None else EWMu()
        self.covariance_estimator = cov if cov is not None else EWCovariance()
        prior = MagicMock()
        prior.partial_fit = MagicMock(return_value=prior)
        prior.mu_estimator = self.mu_estimator
        prior.covariance_estimator = self.covariance_estimator
        self.prior_estimator = prior

    def fit(self, X, y=None):
        return self

    def partial_fit(self, X, y=None):
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
    return Universe(tickers=["A"], metadata={"A": "core"}, fees={"A": 0.0})


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def test_init_valid(stub_universe: Universe) -> None:
    """Test engine accepts Universe."""
    engine = PortfolioExperimentEngine(stub_universe)
    assert engine._universe is stub_universe
    assert engine._strategies == {}
    assert engine._n_jobs == -1


def test_init_bad_universe() -> None:
    """Test TypeError is raised for non-Universe first arg."""
    with pytest.raises(TypeError, match="universe must implement UniverseProtocol"):
        PortfolioExperimentEngine(universe="invalid")  # type: ignore


def test_add_strategy_valid(stub_universe: Universe) -> None:
    """Test valid strategy is stored correctly."""
    engine = PortfolioExperimentEngine(stub_universe)
    est = DummyEstimator()
    grid = {"param": [1, 2]}

    engine.add_strategy("MyStrat", est, grid)

    assert "MyStrat" in engine._strategies
    assert isinstance(engine._strategies["MyStrat"]["estimator"], DummyEstimator)
    assert engine._strategies["MyStrat"]["grid"] == grid


def test_add_strategy_grid_defensive_copy(stub_universe: Universe) -> None:
    """Test grid dict is defensively copied upon registration."""
    engine = PortfolioExperimentEngine(stub_universe)
    est = DummyEstimator()
    grid = {"param": [1, 2]}

    engine.add_strategy("MyStrat", est, grid)
    grid["param"] = [999]

    assert engine._strategies["MyStrat"]["grid"] == {"param": [1, 2]}


def test_add_strategy_duplicate_name(stub_universe: Universe) -> None:
    """Test ValueError is raised on duplicate registration."""
    engine = PortfolioExperimentEngine(stub_universe)
    engine.add_strategy("MyStrat", DummyEstimator(), {})

    with pytest.raises(ValueError, match="already registered"):
        engine.add_strategy("MyStrat", DummyEstimator(), {})


def test_add_strategy_bad_estimator(stub_universe: Universe) -> None:
    """Test TypeError is raised for non-BaseEstimator."""
    engine = PortfolioExperimentEngine(stub_universe)
    with pytest.raises(TypeError, match="scikit-learn BaseEstimator"):
        engine.add_strategy("MyStrat", estimator="not_an_estimator", grid={})  # type: ignore


def test_add_strategy_bad_grid(stub_universe: Universe) -> None:
    """Test TypeError is raised for non-dict grid."""
    engine = PortfolioExperimentEngine(stub_universe)
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
    stub_universe: Universe,
) -> None:
    """Test execution with WalkForward cross-validator."""
    # Setup mocks
    mock_gscv_instance = MagicMock()
    mock_gscv.return_value = mock_gscv_instance

    mock_portfolio = MagicMock()
    mock_portfolio.tag = "MyStrat"
    mock_cv_predict.return_value = mock_portfolio

    mock_pop_instance = MagicMock()
    mock_population.return_value = mock_pop_instance

    # Setup engine
    engine = PortfolioExperimentEngine(stub_universe)
    engine.add_strategy("MyStrat", DummyEstimator(), {"p": [1]})

    # Run
    population = engine.run_robustness_test(
        cv_type="walk_forward", train_size=252, test_size=63
    )

    # Verify inner CV was derived dynamically to fit inside outer train_size
    call_args = mock_gscv.call_args[1] if mock_gscv.call_args else {}
    assert isinstance(call_args["cv"], WalkForward)
    assert call_args["cv"].train_size == 151
    assert call_args["cv"].test_size == 50

    # Verify predict was called with outer CV
    mock_cv_predict.assert_called_once()
    predict_kwargs = mock_cv_predict.call_args[1]
    assert isinstance(predict_kwargs["cv"], WalkForward)
    assert predict_kwargs["cv"].train_size == 252
    assert predict_kwargs["cv"].test_size == 63
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
    stub_universe: Universe,
) -> None:
    """Test execution with CombinatorialPurgedCV."""
    engine = PortfolioExperimentEngine(stub_universe)
    engine.add_strategy("Strat", DummyEstimator(), {})
    engine.run_robustness_test(cv_type="combinatorial", n_folds=5, n_test_folds=2)

    # Inner CV is restricted to standard single-test-fold WalkForward
    call_args = mock_gscv.call_args[1] if mock_gscv.call_args else {}
    assert isinstance(call_args["cv"], WalkForward)

    # Outer CV passed to cross_val_predict is CombinatorialPurgedCV
    predict_cv = mock_cv_predict.call_args[1]["cv"]
    assert isinstance(predict_cv, CombinatorialPurgedCV)


@patch("flowportfolio.core.experiment.Population")
@patch("flowportfolio.core.experiment.cross_val_predict")
@patch("flowportfolio.core.experiment.GridSearchCV")
def test_run_robustness_test_randomised(
    mock_gscv: MagicMock,
    mock_cv_predict: MagicMock,
    mock_population: MagicMock,
    stub_universe: Universe,
) -> None:
    """Test execution with MultipleRandomizedCV."""
    engine = PortfolioExperimentEngine(stub_universe)
    engine.add_strategy("Strat", DummyEstimator(), {})
    # Need a base CV for randomized
    base_cv = WalkForward(train_size=252, test_size=63)
    engine.run_robustness_test(
        cv_type="randomised", n_subsamples=10, walk_forward=base_cv, asset_subset_size=2
    )

    # Inner CV is restricted to standard single-test-fold WalkForward
    call_args = mock_gscv.call_args[1] if mock_gscv.call_args else {}
    assert isinstance(call_args["cv"], WalkForward)

    # Outer CV passed to cross_val_predict is MultipleRandomizedCV
    predict_cv = mock_cv_predict.call_args[1]["cv"]
    assert isinstance(predict_cv, MultipleRandomizedCV)


def test_run_robustness_test_invalid_inner_cv_type(stub_universe: Universe) -> None:
    """Test ValueError is raised when non-single-test-fold splitter is requested for inner CV."""
    engine = PortfolioExperimentEngine(stub_universe)
    engine.add_strategy("Strat", DummyEstimator(), {})
    with pytest.raises(
        ValueError, match="Inner cross-validation splitter cannot be 'combinatorial'"
    ):
        engine.run_robustness_test(
            cv_type="walk_forward",
            inner_cv_kwargs={"cv_type": "combinatorial"},
        )


@patch("flowportfolio.core.experiment.Population")
@patch("flowportfolio.core.experiment.cross_val_predict")
@patch("flowportfolio.core.experiment.GridSearchCV")
def test_run_robustness_test_collection_flattening(
    mock_gscv: MagicMock,
    mock_cv_predict: MagicMock,
    mock_population: MagicMock,
    stub_universe: Universe,
) -> None:
    """Test that multi-path Population predictions are flattened without nesting."""
    p1 = MagicMock()
    p2 = MagicMock()
    mock_cv_predict.return_value = [p1, p2]

    engine = PortfolioExperimentEngine(stub_universe)
    engine.add_strategy("Strat", DummyEstimator(), {})
    engine.run_robustness_test(cv_type="walk_forward", train_size=10, test_size=1)

    mock_population.assert_called_once_with([p1, p2])


def test_run_robustness_test_unknown_cv_type(stub_universe: Universe) -> None:
    """Test ValueError is raised for unsupported CV types."""
    engine = PortfolioExperimentEngine(stub_universe)
    with pytest.raises(ValueError, match="Unknown cv_type"):
        engine.run_robustness_test(cv_type="unknown_cv")


def test_run_robustness_test_no_fetch(stub_universe_no_returns: Universe) -> None:
    """Test engine propagates ValueError if returns are not available."""
    engine = PortfolioExperimentEngine(stub_universe_no_returns)
    engine.add_strategy("Strat", DummyEstimator(), {})
    with pytest.raises(ValueError, match="Returns are not yet available"):
        engine.run_robustness_test(train_size=252, test_size=63)


@patch("flowportfolio.core.experiment.Population")
@patch("flowportfolio.core.experiment.cross_val_predict")
@patch("flowportfolio.core.experiment.GridSearchCV")
def test_run_robustness_test_explicit_inner_cv_kwargs(
    mock_gscv: MagicMock,
    mock_cv_predict: MagicMock,
    mock_population: MagicMock,
    stub_universe: Universe,
) -> None:
    """Test execution with explicit inner_cv_kwargs."""
    engine = PortfolioExperimentEngine(stub_universe)
    engine.add_strategy("MyStrat", DummyEstimator(), {"p": [1]})

    engine.run_robustness_test(
        cv_type="walk_forward",
        inner_cv_kwargs={"train_size": 100, "test_size": 25},
        train_size=252,
        test_size=63,
    )

    call_args = mock_gscv.call_args[1] if mock_gscv.call_args else {}
    assert isinstance(call_args["cv"], WalkForward)
    assert call_args["cv"].train_size == 100
    assert call_args["cv"].test_size == 25


@patch("flowportfolio.core.experiment.Population")
@patch("flowportfolio.core.experiment.online_predict")
@patch("flowportfolio.core.experiment.OnlineGridSearch")
def test_run_online_evaluation_burn_in(
    mock_online_search: MagicMock,
    mock_online_predict: MagicMock,
    mock_population: MagicMock,
    stub_universe: Universe,
) -> None:
    """Test run_online_evaluation fits strictly on burn_in_size to prevent look-ahead bias."""
    df = pd.DataFrame(
        {"A": [0.01] * 10, "B": [-0.01] * 10},
        index=pd.date_range("2026-01-01", periods=10),
    )
    stub_universe._returns = df

    search_instance = MagicMock()
    mock_online_search.return_value = search_instance

    engine = PortfolioExperimentEngine(stub_universe)
    engine.add_strategy("OnlineStrat", DummyOnlineEstimator(), {})

    engine.run_online_evaluation(burn_in_size=4)

    # Verify fit was called strictly on the burn-in slice of returns (first 4 rows)
    search_instance.fit.assert_called_once()
    fit_arg = search_instance.fit.call_args[0][0]
    assert len(fit_arg) == 4

    # Verify online_predict was called with warmup_size=4
    mock_online_predict.assert_called_once()
    predict_kwargs = mock_online_predict.call_args[1]
    assert predict_kwargs["warmup_size"] == 4
    assert predict_kwargs["portfolio_params"] == {"tag": "OnlineStrat"}


@patch("flowportfolio.core.experiment.Population")
@patch("flowportfolio.core.experiment.online_predict")
@patch("flowportfolio.core.experiment.OnlineGridSearch")
@patch("flowportfolio.core.experiment.cross_val_predict")
@patch("flowportfolio.core.experiment.GridSearchCV")
def test_online_evaluation_with_online_kwargs_and_explicit_train_size(
    mock_gscv: MagicMock,
    mock_cv_predict: MagicMock,
    mock_online_search: MagicMock,
    mock_online_predict: MagicMock,
    mock_pop: MagicMock,
    stub_universe: Universe,
) -> None:
    df = pd.DataFrame(
        {"A": [0.01] * 300, "B": [-0.01] * 300},
        index=pd.date_range("2026-01-01", periods=300),
    )
    stub_universe._returns = df

    search_instance = MagicMock()
    mock_online_search.return_value = search_instance

    engine = PortfolioExperimentEngine(stub_universe)
    est = DummyOnlineEstimator(mu=EWMu(), cov=EWCovariance())
    engine.add_strategy("OnlineStrat", est, {})

    # Verify no TypeError or zero-split ValueError occurs with online predict kwargs
    engine.run_online_evaluation(
        warmup_size=100, train_size=100, test_size=20, previous=False
    )

    # Verify engine.latest_weights is populated after run_robustness_test()
    mock_p = MagicMock()
    mock_p.weights_dict = {"A": 0.6, "B": 0.4}
    mock_cv_predict.return_value = [mock_p]

    robustness_search = MagicMock()
    robustness_search.weights_ = [0.5, 0.5]
    robustness_search.feature_names_in_ = ["A", "B"]
    mock_gscv.return_value = robustness_search

    engine.run_robustness_test(cv_type="walk_forward", train_size=252, test_size=63)
    assert "OnlineStrat" in engine.latest_weights
    assert engine.latest_weights["OnlineStrat"] == {"A": 0.5, "B": 0.5}


def test_walk_forward_with_calendar_freq_resolves_row_count(
    stub_universe: Universe,
) -> None:
    """Verify _resolve_outer_train_size extracts row count when freq is supplied."""
    # 500 business days
    dates = pd.date_range("2024-01-01", periods=500, freq="B")
    df = pd.DataFrame({"A": [0.01] * 500, "B": [-0.01] * 500}, index=dates)
    stub_universe._returns = df

    engine = PortfolioExperimentEngine(stub_universe)
    wf = WalkForward(train_size=12, test_size=3, freq="ME")
    outer_train = engine._resolve_outer_train_size(
        "walk_forward",
        wf,
        {"train_size": 12, "test_size": 3, "freq": "ME"},
        len(df),
        df,
    )
    # 12 months is ~250 trading days, not 12 rows!
    assert outer_train > 50


def test_skfolio_grid_search_cv_inner_folds_causality() -> None:
    """Verify SkfolioGridSearchCV uses equal weights during inner CV and refits with previous_weights."""
    import numpy as np
    from skfolio.optimization import MeanRisk

    from flowportfolio.core.optimisation import SkfolioGridSearchCV

    df = pd.DataFrame(
        np.random.randn(100, 3) * 0.01,
        columns=["A", "B", "C"],
        index=pd.date_range("2026-01-01", periods=100),
    )
    opt = MeanRisk()
    opt.max_turnover = 0.05
    user_pw = {"A": 0.8, "B": 0.1, "C": 0.1}

    gscv = SkfolioGridSearchCV(
        estimator=opt,
        param_grid={"l1_coef": [0.0]},
        cv=2,
    )
    gscv.previous_weights = user_pw
    gscv.fit(df)

    # best_estimator_ should be refitted with user_pw and satisfy max_turnover
    inner = gscv._get_target_optimizer(gscv.best_estimator_)
    assert inner.previous_weights == user_pw
    assert gscv.weights_ is not None
    # Turnover from user_pw: sum(|w - w_prev|) / 2 <= 0.05
    turnover = np.sum(np.abs(gscv.weights_ - np.array([0.8, 0.1, 0.1]))) / 2.0
    assert turnover <= 0.05 + 1e-4


def test_run_robustness_test_preserves_user_weights_with_preselection_and_freq() -> (
    None
):
    """Verify live terminal refit satisfies max_turnover against user current_weights and supports freq='ME'."""
    from skfolio.optimization import MeanRisk
    from skfolio.pre_selection import DropCorrelated

    from flowportfolio import ConstraintBuilder, StrategyBuilder
    from tests.test_audit_remediation_integration import _fetched_universe

    u = _fetched_universe(400)
    cw = {"SPY": 0.70, "TLT": 0.10, "GLD": 0.10, "QQQ": 0.10}
    spec = ConstraintBuilder(u).max_turnover(0.05, cw).build()
    pipe = (
        StrategyBuilder(constraints=spec)
        .add_pre_selection(DropCorrelated(threshold=0.85))
        .set_optimizer(MeanRisk())
        .build_pipeline()
    )

    engine = PortfolioExperimentEngine(u, n_jobs=1)
    engine.add_strategy("TurnoverPreSel", pipe, {"optimizer__l1_coef": [0.0]})
    engine.run_robustness_test(
        cv_type="walk_forward", train_size=6, test_size=2, freq="ME"
    )
    lw = engine.latest_weights["TurnoverPreSel"]
    for k, w_val in cw.items():
        assert abs(lw[k] - w_val) <= 0.05 + 1e-5
