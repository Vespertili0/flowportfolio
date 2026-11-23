"""Unit and unmocked skfolio oracle tests for PortfolioExperimentEngine (Cycle 8).

Verifies strategy registration, nested cross-validation, Fold-0 causal isolation,
gap-aware CPCV inner tuning, streaming online partial_fit parity, and immutable
ExperimentResult live-weight binding without mocking core skfolio execution.
"""

from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import pandas as pd
import pytest
from skfolio import MultiPeriodPortfolio, Population
from skfolio.model_selection import CombinatorialPurgedCV, WalkForward
from skfolio.moments import EWCovariance, EWMu
from skfolio.optimization import MeanRisk, ObjectiveFunction
from skfolio.pre_selection import DropCorrelated
from skfolio.prior import EmpiricalPrior
from sklearn.base import BaseEstimator

from flowportfolio import ConstraintBuilder, StrategyBuilder
from flowportfolio.core import experiment as exp_module
from flowportfolio.core.experiment import PortfolioExperimentEngine
from flowportfolio.core.optimisation import SkfolioGridSearchCV
from flowportfolio.core.universe import Universe

# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------


class DummyEstimator(BaseEstimator):
    """A minimal valid estimator for testing registration type assertions."""

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return X


def _make_universe(
    n_obs: int = 120,
    tickers: list[str] | None = None,
    seed: int = 42,
) -> Universe:
    cols = tickers or ["A", "B", "C"]
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-02", periods=n_obs)
    data = {
        col: rng.normal(0.0004 + 0.0002 * i, 0.01, n_obs) for i, col in enumerate(cols)
    }
    u = Universe(
        tickers=cols,
        metadata={c: "core" if i == 0 else "satellite" for i, c in enumerate(cols)},
        fees={c: 0.001 for c in cols},
    )
    u._returns = pd.DataFrame(data, index=idx)
    return u


def _make_twin_universe(n_obs: int = 80, seed: int = 42) -> Universe:
    """O3_CLONE_TURNOVER twin universe where TLT ~= GLD."""
    rng = np.random.default_rng(seed)
    spy = rng.normal(2e-4, 0.01, n_obs)
    tlt = rng.normal(5e-4, 0.01, n_obs)
    gld = tlt + rng.normal(0.0, 1e-4, n_obs)
    df = pd.DataFrame(
        {"SPY": spy, "TLT": tlt, "GLD": gld},
        index=pd.bdate_range("2025-01-02", periods=n_obs),
    )
    u = Universe(
        tickers=["SPY", "TLT", "GLD"],
        metadata={"SPY": "core", "TLT": "core", "GLD": "core"},
        fees={"SPY": 0.0, "TLT": 0.0, "GLD": 0.0},
    )
    u._returns = df
    return u


@pytest.fixture
def stub_universe() -> Universe:
    """Provide a deterministic synthetic Universe without network access."""
    return _make_universe(n_obs=120, tickers=["A", "B", "C"], seed=42)


@pytest.fixture
def stub_universe_no_returns() -> Universe:
    """Provide a Universe stub that has not fetched data."""
    return Universe(tickers=["A"], metadata={"A": "core"}, fees={"A": 0.0})


# ---------------------------------------------------------------------------
# Initialisation & Strategy Registration
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
        PortfolioExperimentEngine(universe="invalid")  # type: ignore[arg-type]


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
        engine.add_strategy("MyStrat", estimator="not_an_estimator", grid={})  # type: ignore[arg-type]


def test_add_strategy_bad_grid(stub_universe: Universe) -> None:
    """Test TypeError is raised for non-dict grid."""
    engine = PortfolioExperimentEngine(stub_universe)
    with pytest.raises(TypeError, match="grid must be a dictionary"):
        engine.add_strategy("MyStrat", DummyEstimator(), grid=["not", "dict"])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Validation Errors & Calendar Resolution
# ---------------------------------------------------------------------------


def test_run_robustness_test_invalid_inner_cv_type(stub_universe: Universe) -> None:
    """Test ValueError is raised when non-single-test-fold splitter is requested for inner CV."""
    engine = PortfolioExperimentEngine(stub_universe)
    engine.add_strategy("Strat", MeanRisk(), {})
    with pytest.raises(
        ValueError, match="Inner cross-validation splitter cannot be 'combinatorial'"
    ):
        engine.run_robustness_test(
            cv_type="walk_forward",
            inner_cv_kwargs={"cv_type": "combinatorial"},
        )


def test_run_robustness_test_unknown_cv_type(stub_universe: Universe) -> None:
    """Test ValueError is raised for unsupported CV types."""
    engine = PortfolioExperimentEngine(stub_universe)
    with pytest.raises(ValueError, match="Unknown cv_type"):
        engine.run_robustness_test(cv_type="unknown_cv")


def test_run_robustness_test_no_fetch(stub_universe_no_returns: Universe) -> None:
    """Test engine propagates ValueError if returns are not available."""
    engine = PortfolioExperimentEngine(stub_universe_no_returns)
    engine.add_strategy("Strat", MeanRisk(), {})
    with pytest.raises(ValueError, match="Returns are not yet available"):
        engine.run_robustness_test(train_size=40, test_size=20)


def test_walk_forward_with_calendar_freq_resolves_row_count(
    stub_universe: Universe,
) -> None:
    """Verify _resolve_outer_train_size extracts row count when freq is supplied."""
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
    assert outer_train > 50


# ---------------------------------------------------------------------------
# Unmocked Real skfolio Execution (WalkForward, Combinatorial, Randomised, Online)
# ---------------------------------------------------------------------------


def test_run_robustness_test_walk_forward_and_explicit_inner_cv(
    stub_universe: Universe,
) -> None:
    """Unmocked WalkForward execution produces tagged MultiPeriodPortfolio and live weights."""
    engine = PortfolioExperimentEngine(stub_universe, n_jobs=1)
    engine.add_strategy("MyStrat", MeanRisk(), {"l2_coef": [0.0, 0.1]})

    pop = engine.run_robustness_test(
        cv_type="walk_forward",
        train_size=60,
        test_size=20,
        inner_cv_kwargs={"train_size": 30, "test_size": 10},
    )

    assert isinstance(pop, Population)
    assert len(pop) == 1
    assert isinstance(pop[0], MultiPeriodPortfolio)
    assert pop[0].tag == "MyStrat"
    assert len(pop[0].portfolios) == 3  # (120 - 60) / 20 = 3 OOS folds
    assert "MyStrat" in engine.latest_weights
    assert sum(engine.latest_weights["MyStrat"].values()) == pytest.approx(
        1.0, abs=1e-6
    )


def test_run_robustness_test_randomised_flattens_population(
    stub_universe: Universe,
) -> None:
    """Unmocked MultipleRandomizedCV returns a flat Population of tagged portfolios."""
    engine = PortfolioExperimentEngine(stub_universe, n_jobs=1)
    engine.add_strategy("RandStrat", MeanRisk(), {"l2_coef": [0.0]})

    base_cv = WalkForward(train_size=60, test_size=20)
    pop = engine.run_robustness_test(
        cv_type="randomised",
        n_subsamples=3,
        walk_forward=base_cv,
        asset_subset_size=2,
        random_state=42,
    )

    assert isinstance(pop, Population)
    assert len(pop) == 3
    assert all(p.tag == "RandStrat" for p in pop)
    assert "RandStrat" in engine.latest_weights


def test_run_online_evaluation_burn_in_and_kwargs(stub_universe: Universe) -> None:
    """Unmocked online evaluation tunes strictly on burn_in and produces OOS MultiPeriodPortfolio."""
    engine = PortfolioExperimentEngine(stub_universe, n_jobs=1)
    est = MeanRisk(
        prior_estimator=EmpiricalPrior(
            mu_estimator=EWMu(half_life=15),
            covariance_estimator=EWCovariance(half_life=15),
        )
    )
    engine.add_strategy("OnlineStrat", est, {"l2_coef": [0.0, 0.1]})

    pop = engine.run_online_evaluation(burn_in_size=60, test_size=20, previous=False)

    assert isinstance(pop, Population)
    assert len(pop) == 1
    assert isinstance(pop[0], MultiPeriodPortfolio)
    assert pop[0].tag == "OnlineStrat"
    assert len(pop[0].portfolios) == 3  # (120 - 60) / 20 = 3 post-burn-in steps
    assert "OnlineStrat" in engine.latest_weights
    assert sum(engine.latest_weights["OnlineStrat"].values()) == pytest.approx(
        1.0, abs=1e-6
    )


def test_skfolio_grid_search_cv_per_asset_turnover_contract() -> None:
    """Verify SkfolioGridSearchCV enforces skfolio's per-asset |w_i - w_prev_i| <= limit."""
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        rng.normal(0.0005, 0.01, (100, 3)),
        columns=["A", "B", "C"],
        index=pd.bdate_range("2026-01-01", periods=100),
    )
    opt = MeanRisk(max_turnover=0.05)
    user_pw = {"A": 0.8, "B": 0.1, "C": 0.1}

    gscv = SkfolioGridSearchCV(
        estimator=opt,
        param_grid={"l1_coef": [0.0]},
        cv=WalkForward(train_size=50, test_size=25),
        previous_weights=user_pw,
    )
    gscv.fit(df)

    inner = gscv._get_target_optimizer(gscv.best_estimator_)
    assert inner.previous_weights == user_pw
    assert gscv.weights_ is not None
    per_asset_diff = np.max(np.abs(gscv.weights_ - np.array([0.8, 0.1, 0.1])))
    assert per_asset_diff <= 0.05 + 1e-5


def test_run_robustness_test_preserves_user_weights_with_preselection_and_freq() -> (
    None
):
    """Verify live terminal refit satisfies max_turnover against user current_weights and supports freq='ME'."""
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


# ---------------------------------------------------------------------------
# Cycle 8 (F-02, F-05, F-10): Causal Isolation, CPCV Gaps, Streaming & Result
# ---------------------------------------------------------------------------


def test_fold0_causal_isolation_under_transaction_costs_without_max_turnover() -> None:
    """F-02.1 (O3_CLONE_TURNOVER): Present-day holdings must NEVER leak into historical Fold 0."""
    u1 = _make_twin_universe(n_obs=80, seed=42)
    u2 = _make_twin_universe(n_obs=80, seed=42)

    pw_tlt = {"SPY": 0.0, "TLT": 1.0, "GLD": 0.0}
    pw_gld = {"SPY": 0.0, "TLT": 0.0, "GLD": 1.0}

    eng1 = PortfolioExperimentEngine(u1, n_jobs=1)
    eng1.add_strategy(
        "TC",
        MeanRisk(
            objective_function=ObjectiveFunction.MAXIMIZE_UTILITY,
            transaction_costs=0.01,
            previous_weights=pw_tlt,
        ),
        {"l2_coef": [0.0]},
    )
    pop1 = eng1.run_robustness_test(cv_type="walk_forward", train_size=40, test_size=20)

    eng2 = PortfolioExperimentEngine(u2, n_jobs=1)
    eng2.add_strategy(
        "TC",
        MeanRisk(
            objective_function=ObjectiveFunction.MAXIMIZE_UTILITY,
            transaction_costs=0.01,
            previous_weights=pw_gld,
        ),
        {"l2_coef": [0.0]},
    )
    pop2 = eng2.run_robustness_test(cv_type="walk_forward", train_size=40, test_size=20)

    # Historical Fold-0 OOS weights MUST be invariant to present-day TLT=1 vs GLD=1 holdings
    w1_fold0 = pop1[0].portfolios[0].weights
    w2_fold0 = pop2[0].portfolios[0].weights
    np.testing.assert_allclose(w1_fold0, w2_fold0, rtol=0, atol=1e-12)

    # Live terminal refit MUST still respect each user's actual physical holdings under transaction_costs
    assert eng1.latest_weights["TC"]["TLT"] > 0.90
    assert eng2.latest_weights["TC"]["GLD"] > 0.90


def test_online_fold0_causal_isolation_under_transaction_costs_without_max_turnover() -> (
    None
):
    """F-02.1: run_online_evaluation must also isolate Fold 0 from user holdings under transaction_costs."""
    u1 = _make_twin_universe(n_obs=120, seed=42)
    u2 = _make_twin_universe(n_obs=120, seed=42)

    pw_tlt = {"SPY": 0.0, "TLT": 1.0, "GLD": 0.0}
    pw_gld = {"SPY": 0.0, "TLT": 0.0, "GLD": 1.0}

    def _make_est(pw: dict[str, float]) -> MeanRisk:
        return MeanRisk(
            objective_function=ObjectiveFunction.MAXIMIZE_UTILITY,
            transaction_costs=0.01,
            previous_weights=pw,
            prior_estimator=EmpiricalPrior(
                mu_estimator=EWMu(half_life=15),
                covariance_estimator=EWCovariance(half_life=15),
            ),
        )

    eng1 = PortfolioExperimentEngine(u1, n_jobs=1)
    eng1.add_strategy("TCOnline", _make_est(pw_tlt), {"l2_coef": [0.0]})
    pop1 = eng1.run_online_evaluation(burn_in_size=60, test_size=20)

    eng2 = PortfolioExperimentEngine(u2, n_jobs=1)
    eng2.add_strategy("TCOnline", _make_est(pw_gld), {"l2_coef": [0.0]})
    pop2 = eng2.run_online_evaluation(burn_in_size=60, test_size=20)

    np.testing.assert_allclose(
        pop1[0].portfolios[0].weights,
        pop2[0].portfolios[0].weights,
        rtol=0,
        atol=1e-12,
    )
    assert eng1.latest_weights["TCOnline"]["TLT"] > 0.90
    assert eng2.latest_weights["TCOnline"]["GLD"] > 0.90


def test_walk_forward_neutral_entry_and_sequential_turnover_enforcement() -> None:
    """F-02.2: Walk-forward Fold 0 enters unconstrained from cash; Fold 1+ enforces max_turnover."""
    rng = np.random.default_rng(13)
    n_obs = 100
    # C strongly dominates A and B; neutral Fold 0 should allocate heavily into C (> 0.8),
    # NOT stay artificially pinned within 0.05 of 1/N (<= 0.383) or user A=0.90!
    df = pd.DataFrame(
        {
            "A": rng.normal(-0.001, 0.01, n_obs),
            "B": rng.normal(0.000, 0.01, n_obs),
            "C": rng.normal(0.004, 0.01, n_obs),
        },
        index=pd.bdate_range("2025-01-02", periods=n_obs),
    )
    u = Universe(
        tickers=["A", "B", "C"],
        metadata={"A": "core", "B": "core", "C": "core"},
        fees={"A": 0.0, "B": 0.0, "C": 0.0},
    )
    u._returns = df

    user_pw = {"A": 0.90, "B": 0.05, "C": 0.05}
    opt = MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_RETURN,
        max_turnover=0.05,
        previous_weights=user_pw,
    )
    eng = PortfolioExperimentEngine(u, n_jobs=1)
    eng.add_strategy("TurnoverWF", opt, {"l2_coef": [0.0]})
    pop = eng.run_robustness_test(cv_type="walk_forward", train_size=40, test_size=20)

    w0 = pop[0].portfolios[0].weights
    w1 = pop[0].portfolios[1].weights
    # Fold 0 must NOT be artificially capped at 1/3 + 0.05 = 0.3833 on C
    assert w0[2] > 0.80
    # Fold 1 enforces per-asset max_turnover=0.05 from Fold 0
    assert float(np.max(np.abs(w1 - w0))) <= 0.05 + 1e-5
    # Live terminal refit enforces per-asset max_turnover=0.05 from user_pw
    live_w = eng.latest_weights["TurnoverWF"]
    for col, prev_val in user_pw.items():
        assert abs(live_w[col] - prev_val) <= 0.05 + 1e-5


def test_combinatorial_inner_cv_never_spans_purged_index_gaps() -> None:
    """F-02.3: CombinatorialPurgedCV inner WalkForward splits must stay within contiguous segments."""
    from flowportfolio.core.experiment import _GapAwareWalkForward

    u = _make_universe(n_obs=120, tickers=["A", "B", "C"], seed=5)
    returns = u.returns
    cpcv = CombinatorialPurgedCV(n_folds=4, n_test_folds=2, purged_size=5)
    inner_cv = _GapAwareWalkForward(
        train_size=15, test_size=5, reference_index=returns.index
    )
    assert isinstance(inner_cv, WalkForward)

    checked_discontinuous_splits = 0
    for outer_train_idx, _ in cpcv.split(returns):
        if np.any(np.diff(outer_train_idx) > 1):
            checked_discontinuous_splits += 1
        X_outer_train = returns.iloc[outer_train_idx]
        inner_splits = list(inner_cv.split(X_outer_train))
        assert len(inner_splits) >= 1
        for in_tr, in_te in inner_splits:
            orig_tr = returns.index.get_indexer(X_outer_train.index[in_tr])
            orig_te = returns.index.get_indexer(X_outer_train.index[in_te])
            # Zero discontinuities inside inner train, inside inner test, or across train->test boundary
            assert np.all(np.diff(orig_tr) == 1)
            assert np.all(np.diff(orig_te) == 1)
            assert int(orig_te[0] - orig_tr[-1]) == 1

    assert checked_discontinuous_splits >= 1

    # End-to-end combinatorial execution succeeds cleanly
    eng = PortfolioExperimentEngine(u, n_jobs=1)
    eng.add_strategy("CPCVStrat", MeanRisk(), {"l2_coef": [0.0, 0.1]})
    pop = eng.run_robustness_test(
        cv_type="combinatorial", n_folds=4, n_test_folds=2, purged_size=5
    )
    assert isinstance(pop, Population)
    assert len(pop) > 0
    assert "CPCVStrat" in eng.latest_weights


def test_run_online_evaluation_streaming_partial_fit_matches_continuous_oracle() -> (
    None
):
    """F-05.2 (O5): Online live weights must match continuous causal partial_fit to 1e-12."""
    u = _make_universe(n_obs=160, tickers=["A", "B", "C"], seed=7)
    returns = u.returns

    def _make_ew_opt() -> MeanRisk:
        return MeanRisk(
            prior_estimator=EmpiricalPrior(
                mu_estimator=EWMu(half_life=40),
                covariance_estimator=EWCovariance(half_life=40),
            ),
            l2_coef=0.0,
        )

    eng = PortfolioExperimentEngine(u, n_jobs=1)
    eng.add_strategy("EW", _make_ew_opt(), {"l2_coef": [0.0]})
    eng.run_online_evaluation(burn_in_size=100, test_size=20)

    # Exact continuous causal partial_fit oracle across the full 160-day stream
    oracle = _make_ew_opt()
    oracle.partial_fit(returns.iloc[:100])
    for start in range(100, 160, 20):
        oracle.partial_fit(returns.iloc[start : start + 20])

    actual_w = np.array([eng.latest_weights["EW"][c] for c in ["A", "B", "C"]])
    np.testing.assert_allclose(actual_w, oracle.weights_, rtol=0, atol=1e-12)


def test_experiment_result_immutable_live_weights_survive_population_rewrap(
    stub_universe: Universe,
) -> None:
    """F-05.1: ExperimentResult preserves immutable live_weights independent of Population(list(pop))."""
    from flowportfolio.core.experiment import ExperimentResult

    eng = PortfolioExperimentEngine(stub_universe, n_jobs=1)
    eng.add_strategy("StratA", MeanRisk(), {"l2_coef": [0.0]})

    result = eng.run_robustness_test(
        cv_type="walk_forward",
        train_size=60,
        test_size=20,
        return_result=True,
    )
    assert isinstance(result, ExperimentResult)
    assert dataclasses.is_dataclass(result)
    assert eng.last_result == result
    assert isinstance(result.population, Population)
    assert "StratA" in result.live_weights

    # Re-wrapping Population(list(pop)) strips dynamic attributes on skfolio Population,
    # but ExperimentResult and engine.last_result preserve live_weights losslessly.
    rewrapped = Population(list(result.population))
    assert not hasattr(rewrapped, "latest_weights_by_tag")
    assert result.live_weights == eng.latest_weights
    assert eng.last_result is not None
    assert eng.last_result.live_weights == eng.latest_weights

    # Frozen immutability
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.live_weights = {}  # type: ignore[misc]

    # Also verify return_result=True on run_online_evaluation
    eng_online = PortfolioExperimentEngine(stub_universe, n_jobs=1)
    eng_online.add_strategy(
        "OnlineA",
        MeanRisk(
            prior_estimator=EmpiricalPrior(
                mu_estimator=EWMu(half_life=15),
                covariance_estimator=EWCovariance(half_life=15),
            )
        ),
        {"l2_coef": [0.0]},
    )
    res_online = eng_online.run_online_evaluation(
        burn_in_size=60, test_size=20, return_result=True
    )
    assert isinstance(res_online, ExperimentResult)
    assert eng_online.last_result == res_online
    assert "OnlineA" in res_online.live_weights


def test_experiment_module_free_of_mock_name_and_slots_monkey_patch() -> None:
    """F-05.1 & F-10: experiment.py contains zero _mock_name branches or __slots__ monkey-patches."""
    source = inspect.getsource(exp_module)
    assert "_mock_name" not in source
    assert "latest_weights_dict" not in source


# ---------------------------------------------------------------------------
# Cycle 14 (P1-1, P1-2, P2-7, P2-9c, P2-9d): Annualisation, Cost Horizon,
# Drifted Refit Seeding, Concurrency Isolation & Friction n_jobs
# ---------------------------------------------------------------------------


def test_engine_forwards_universe_periods_per_year_as_annualization_factor() -> None:
    """P1-1: Both run_robustness_test and run_online_evaluation forward periods_per_year as annualization_factor."""
    import math

    rng = np.random.default_rng(17)
    dates = pd.date_range("2015-01-31", periods=96, freq="BME")
    rets = pd.DataFrame(
        rng.normal([0.006, 0.003, 0.004], [0.04, 0.02, 0.03], size=(96, 3)),
        index=dates,
        columns=["A", "B", "C"],
    )

    class _MonthlyProtocolUniverse:
        def __init__(self) -> None:
            self.tickers = ["A", "B", "C"]
            self.metadata = {"A": "eq", "B": "fi", "C": "alt"}
            self.fees = {"A": 0.001, "B": 0.001, "C": 0.001}
            self.returns = rets
            self.periods_per_year = 12

    monthly_u = _MonthlyProtocolUniverse()

    # 1. Batch walk-forward
    eng_batch = PortfolioExperimentEngine(monthly_u, n_jobs=1)  # type: ignore[arg-type]
    eng_batch.add_strategy("M", MeanRisk(), {"l2_coef": [0.0]})
    pop_batch = eng_batch.run_robustness_test(
        cv_type="walk_forward", train_size=36, test_size=12
    )
    mpp_batch = pop_batch[0]
    assert mpp_batch.portfolios[0].annualization_factor == pytest.approx(12.0)
    assert mpp_batch.annualized_sharpe_ratio == pytest.approx(
        float(mpp_batch.sharpe_ratio) * math.sqrt(12.0), rel=1e-9, abs=1e-12
    )

    # 2. Online evaluation
    eng_online = PortfolioExperimentEngine(monthly_u, n_jobs=1)  # type: ignore[arg-type]
    eng_online.add_strategy(
        "MOnline",
        MeanRisk(
            prior_estimator=EmpiricalPrior(
                mu_estimator=EWMu(half_life=12),
                covariance_estimator=EWCovariance(half_life=12),
            )
        ),
        {"l2_coef": [0.0]},
    )
    pop_online = eng_online.run_online_evaluation(burn_in_size=36, test_size=12)
    mpp_online = pop_online[0]
    assert mpp_online.portfolios[0].annualization_factor == pytest.approx(12.0)
    assert mpp_online.annualized_sharpe_ratio == pytest.approx(
        float(mpp_online.sharpe_ratio) * math.sqrt(12.0), rel=1e-9, abs=1e-12
    )


def test_transaction_costs_reconciled_to_outer_test_size_and_mismatch_warned(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """P1-2: Default unamortized ConstraintBuilder transaction_costs scale by outer test_size so fold cost == cost * turnover."""
    import logging

    from flowportfolio.core.constraints import ConstraintBuilder
    from flowportfolio.strategies import StrategyBuilder

    rng = np.random.default_rng(99)
    n_obs = 320
    dates = pd.bdate_range("2024-01-02", periods=n_obs)
    # Regime shift at index 120: Fold 0 trains on [0:120] (A wins), Fold 2 trains on [126:246] (B wins)
    r_a = np.concatenate([rng.normal(0.005, 0.01, 120), rng.normal(-0.005, 0.01, 200)])
    r_b = np.concatenate([rng.normal(-0.005, 0.01, 120), rng.normal(0.005, 0.01, 200)])
    r_c = rng.normal(0.0005, 0.008, n_obs)
    rets = pd.DataFrame({"A": r_a, "B": r_b, "C": r_c}, index=dates)

    u = Universe(
        ["A", "B", "C"],
        {"A": "eq", "B": "eq", "C": "fi"},
        {"A": 0.0, "B": 0.0, "C": 0.0},
    )
    u._returns = rets
    u._prices = 100.0 * (1.0 + rets).cumprod()

    raw_cost = 0.001
    # 1. Default ConstraintBuilder (no explicit holding_period) -> reconciled to test_size=63
    spec_default = ConstraintBuilder(u).transaction_costs(raw_cost).build()
    pipe_default = (
        StrategyBuilder(constraints=spec_default)
        .set_optimizer(MeanRisk(objective_function=ObjectiveFunction.MAXIMIZE_UTILITY))
        .build_pipeline()
    )

    eng = PortfolioExperimentEngine(u, n_jobs=1)
    eng.add_strategy("DefaultTC", pipe_default, {"optimizer__l2_coef": [0.0]})
    pop = eng.run_robustness_test(cv_type="walk_forward", train_size=120, test_size=63)
    mpp = pop[0]
    assert len(mpp.portfolios) >= 2
    assert mpp.portfolios[0].total_cost == pytest.approx(0.0, abs=1e-12)

    checked_turnover = False
    for prev_fold, fold in zip(mpp.portfolios[:-1], mpp.portfolios[1:], strict=True):
        l1_turnover = float(np.sum(np.abs(fold.weights - prev_fold.ending_weights)))
        if l1_turnover > 1e-3:
            checked_turnover = True
            integrated_cost = len(fold.returns) * float(fold.total_cost)
            assert integrated_cost == pytest.approx(
                raw_cost * l1_turnover, rel=1e-6, abs=1e-9
            )
    assert checked_turnover

    # 2. Explicit holding_period=1 with test_size=63 -> respects user setting AND logs mismatch warning
    spec_explicit = (
        ConstraintBuilder(u, holding_period=1.0).transaction_costs(raw_cost).build()
    )
    pipe_explicit = (
        StrategyBuilder(constraints=spec_explicit)
        .set_optimizer(MeanRisk(objective_function=ObjectiveFunction.MAXIMIZE_UTILITY))
        .build_pipeline()
    )
    eng_warn = PortfolioExperimentEngine(u, n_jobs=1)
    eng_warn.add_strategy("ExplicitTC", pipe_explicit, {"optimizer__l2_coef": [0.0]})
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="flowportfolio.core.experiment"):
        eng_warn.run_robustness_test(
            cv_type="walk_forward", train_size=120, test_size=63
        )
    assert any("holding_period" in rec.message for rec in caplog.records)


def test_extract_terminal_oos_weights_uses_drifted_ending_weights_dict() -> None:
    """P2-7: _extract_terminal_oos_weights must prefer drifted ending_weights_dict over starting weights_dict."""
    from skfolio import MultiPeriodPortfolio, Portfolio

    from flowportfolio.core.experiment import _extract_terminal_oos_weights

    dates = pd.bdate_range("2025-01-02", periods=10)
    X = pd.DataFrame({"A": [0.05] * 10, "B": [-0.02] * 10}, index=dates)
    p_drifted = Portfolio(X=X, weights=np.array([0.5, 0.5]), weight_drift=True)
    mpp = MultiPeriodPortfolio(portfolios=[p_drifted], name="drifted", tag="D")

    extracted = _extract_terminal_oos_weights(mpp)
    assert extracted is not None
    assert extracted["A"] == pytest.approx(
        p_drifted.ending_weights_dict["A"], rel=0, abs=1e-12
    )
    assert extracted["B"] == pytest.approx(
        p_drifted.ending_weights_dict["B"], rel=0, abs=1e-12
    )
    assert extracted["A"] > 0.60


def test_engine_run_local_state_preserves_prior_results_on_failure_and_concurrency(
    stub_universe: Universe,
) -> None:
    """P2-9c: Failed runs do not clobber prior latest_weights, and concurrent runs return isolated ExperimentResults."""
    from concurrent.futures import ThreadPoolExecutor

    eng = PortfolioExperimentEngine(stub_universe, n_jobs=1)
    eng.add_strategy("S", MeanRisk(), {"l2_coef": [0.0]})

    first = eng.run_robustness_test(
        cv_type="walk_forward", train_size=60, test_size=20, return_result=True
    )
    # Invalid cv_type must fail WITHOUT wiping prior latest_weights / last_result
    with pytest.raises(ValueError, match="Unknown cv_type"):
        eng.run_robustness_test(cv_type="bogus")
    assert eng.latest_weights == first.live_weights
    assert eng.last_result == first

    # Independent oracles for two distinct train_size configurations
    eng_o1 = PortfolioExperimentEngine(stub_universe, n_jobs=1)
    eng_o1.add_strategy("S", MeanRisk(), {"l2_coef": [0.0]})
    oracle_60 = eng_o1.run_robustness_test(
        cv_type="walk_forward", train_size=60, test_size=20, return_result=True
    )
    eng_o2 = PortfolioExperimentEngine(stub_universe, n_jobs=1)
    eng_o2.add_strategy("S", MeanRisk(), {"l2_coef": [0.0]})
    oracle_100 = eng_o2.run_robustness_test(
        cv_type="walk_forward", train_size=100, test_size=20, return_result=True
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(
            eng.run_robustness_test,
            cv_type="walk_forward",
            train_size=60,
            test_size=20,
            return_result=True,
        )
        f2 = pool.submit(
            eng.run_robustness_test,
            cv_type="walk_forward",
            train_size=100,
            test_size=20,
            return_result=True,
        )
        r1 = f1.result()
        r2 = f2.result()

    for col in stub_universe.tickers:
        assert r1.live_weights["S"][col] == pytest.approx(
            oracle_60.live_weights["S"][col], abs=1e-12
        )
        assert r2.live_weights["S"][col] == pytest.approx(
            oracle_100.live_weights["S"][col], abs=1e-12
        )


def test_default_n_jobs_with_friction_emits_no_skfolio_parallel_warning(
    stub_universe: Universe,
) -> None:
    """P2-9d: Default n_jobs=-1 with rebalancing friction must not trigger skfolio's parallel-disabled UserWarning."""
    import warnings

    from flowportfolio.core.constraints import ConstraintBuilder
    from flowportfolio.strategies import StrategyBuilder

    spec = ConstraintBuilder(stub_universe).transaction_costs(0.001).build()
    pipe = StrategyBuilder(constraints=spec).set_optimizer(MeanRisk()).build_pipeline()
    eng = PortfolioExperimentEngine(stub_universe)  # default n_jobs=-1
    eng.add_strategy("FrictionStrat", pipe, {"optimizer__l2_coef": [0.0]})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        eng.run_robustness_test(cv_type="walk_forward", train_size=60, test_size=20)

    parallel_warns = [
        w
        for w in caught
        if issubclass(w.category, UserWarning)
        and "Parallel processing has been disabled" in str(w.message)
    ]
    assert parallel_warns == []


# ---------------------------------------------------------------------------
# Cycle 18 (Sprint 5): #4, #6c, N1, N4, M1, M2
# ---------------------------------------------------------------------------


def test_run_online_evaluation_with_calendar_freq_scales_tc_and_succeeds() -> None:
    """Sprint 5 (#4): run_online_evaluation with freq='ME' infers monthly test size, handles burn-in warmup, and scales tc."""
    from skfolio.moments import EWCovariance, EWMu
    from skfolio.prior import EmpiricalPrior

    from flowportfolio.core.constraints import ConstraintBuilder
    from flowportfolio.strategies import StrategyBuilder

    rng = np.random.default_rng(104)
    dates = pd.bdate_range("2022-01-03", periods=320)
    rets = pd.DataFrame(
        {
            "A": rng.normal(0.0008, 0.01, len(dates)),
            "B": rng.normal(0.0004, 0.01, len(dates)),
            "C": rng.normal(0.0003, 0.01, len(dates)),
        },
        index=dates,
    )
    u = Universe(
        ["A", "B", "C"],
        {"A": "core", "B": "core", "C": "sat"},
        {"A": 0.0, "B": 0.0, "C": 0.0},
    )
    u._returns = rets

    raw_tc = 0.005
    spec = ConstraintBuilder(u).transaction_costs(raw_tc).build()
    pipe = (
        StrategyBuilder(constraints=spec)
        .set_optimizer(
            MeanRisk(
                prior_estimator=EmpiricalPrior(
                    mu_estimator=EWMu(), covariance_estimator=EWCovariance()
                )
            )
        )
        .build_pipeline()
    )

    eng = PortfolioExperimentEngine(u, n_jobs=1)
    eng.add_strategy("OnlineFreq", pipe, {"optimizer__l2_coef": [0.0, 0.01]})
    result = eng.run_online_evaluation(
        burn_in_size=150, test_size=1, freq="ME", return_result=True
    )
    mpp = result.population[0]
    assert len(mpp.portfolios) >= 3
    # Fold >= 1 transaction_costs must be amortized by ~21 trading days per month, NOT 1.0
    sub_tc = mpp.portfolios[1].transaction_costs
    assert isinstance(sub_tc, (int, float, np.ndarray))
    assert float(np.mean(sub_tc)) < raw_tc / 10.0
    assert result.live_weights["OnlineFreq"]


def test_resolve_outer_train_size_and_randomised_with_freq_and_offset() -> None:
    """Sprint 5 (#6c): _resolve_outer_train_size infers true fold lengths for WalkForward(freq) and DateOffset."""
    from skfolio.model_selection import MultipleRandomizedCV, WalkForward

    rng = np.random.default_rng(106)
    dates = pd.bdate_range("2022-01-03", periods=360)
    rets = pd.DataFrame(
        rng.normal(0.0005, 0.01, (360, 4)),
        index=dates,
        columns=["A", "B", "C", "D"],
    )
    u = Universe(
        ["A", "B", "C", "D"],
        {"A": "core", "B": "core", "C": "sat", "D": "sat"},
        {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0},
    )
    u._returns = rets

    wf_freq = WalkForward(train_size=6, test_size=1, freq="ME")
    mrcv_freq = MultipleRandomizedCV(
        walk_forward=wf_freq,
        n_subsamples=2,
        asset_subset_size=3,
        window_size=220,
        random_state=0,
    )
    tr_len_freq = PortfolioExperimentEngine._resolve_outer_train_size(
        "randomised", mrcv_freq, {"walk_forward": wf_freq}, len(rets), rets
    )
    assert tr_len_freq > 100

    wf_offset = WalkForward(train_size=pd.DateOffset(months=6), test_size=1, freq="ME")
    mrcv_offset = MultipleRandomizedCV(
        walk_forward=wf_offset,
        n_subsamples=2,
        asset_subset_size=3,
        window_size=220,
        random_state=0,
    )
    tr_len_offset = PortfolioExperimentEngine._resolve_outer_train_size(
        "randomised", mrcv_offset, {"walk_forward": wf_offset}, len(rets), rets
    )
    assert 100 <= tr_len_offset < 200

    eng = PortfolioExperimentEngine(u, n_jobs=1)
    eng.add_strategy("RandFreq", MeanRisk(), {"l2_coef": [0.0, 0.01]})
    pop = eng.run_robustness_test(
        cv_type="randomised",
        walk_forward=wf_freq,
        n_subsamples=2,
        asset_subset_size=3,
        window_size=220,
        random_state=0,
    )
    assert len(pop) == 2


def test_randomised_cv_with_turnover_and_tc_neutralizes_fold0_and_avoids_infeasible() -> (
    None
):
    """Sprint 5 (N1): MultipleRandomizedCV with asset_subset_size < N and max_turnover+tc must not fail or charge Fold 0."""
    from skfolio.model_selection import WalkForward

    from flowportfolio.core.constraints import ConstraintBuilder
    from flowportfolio.strategies import StrategyBuilder

    rng = np.random.default_rng(111)
    dates = pd.bdate_range("2022-01-03", periods=320)
    cols = ["A", "B", "C", "D", "E", "F"]
    rets = pd.DataFrame(rng.normal(0.0005, 0.01, (320, 6)), index=dates, columns=cols)
    u = Universe(
        cols,
        {c: "core" for c in cols},
        {c: 0.0 for c in cols},
    )
    u._returns = rets

    cw = {c: 1.0 / 6.0 for c in cols}
    spec = ConstraintBuilder(u).max_turnover(0.05, cw).transaction_costs(0.005).build()
    pipe = StrategyBuilder(constraints=spec).set_optimizer(MeanRisk()).build_pipeline()

    eng = PortfolioExperimentEngine(u, n_jobs=1)
    eng.add_strategy("RandFriction", pipe, {"optimizer__l2_coef": [0.0, 0.01]})
    pop = eng.run_robustness_test(
        cv_type="randomised",
        walk_forward=WalkForward(train_size=120, test_size=30),
        n_subsamples=2,
        asset_subset_size=3,
        window_size=220,
        random_state=0,
    )
    assert len(pop) == 2
    for mpp in pop:
        assert float(mpp.portfolios[0].total_cost) == pytest.approx(0.0, abs=1e-12)


def test_combinatorial_cv_with_transaction_costs_emits_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Sprint 5 (N4): CombinatorialPurgedCV with positive transaction_costs emits a clear warning."""
    import logging

    from flowportfolio.core.constraints import ConstraintBuilder
    from flowportfolio.strategies import StrategyBuilder

    rng = np.random.default_rng(114)
    dates = pd.bdate_range("2022-01-03", periods=200)
    rets = pd.DataFrame(
        rng.normal(0.0005, 0.01, (200, 3)),
        index=dates,
        columns=["A", "B", "C"],
    )
    u = Universe(
        ["A", "B", "C"],
        {"A": "core", "B": "core", "C": "sat"},
        {"A": 0.0, "B": 0.0, "C": 0.0},
    )
    u._returns = rets
    spec = ConstraintBuilder(u).transaction_costs(0.005).build()
    pipe = StrategyBuilder(constraints=spec).set_optimizer(MeanRisk()).build_pipeline()

    eng = PortfolioExperimentEngine(u, n_jobs=1)
    eng.add_strategy("CPCV_TC", pipe, {"optimizer__l2_coef": [0.0]})
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="flowportfolio.core.experiment"):
        eng.run_robustness_test(cv_type="combinatorial", n_folds=4, n_test_folds=2)
    assert any(
        "combinatorial" in rec.message.lower()
        and "transaction_costs" in rec.message.lower()
        for rec in caplog.records
    )


def test_run_robustness_test_explicit_inner_cv_freq_without_sizes_succeeds(
    stub_universe: Universe,
) -> None:
    """Sprint 5 (M1): inner_cv_kwargs={'freq': 'ME'} without train_size/test_size strips freq for row counts."""
    eng = PortfolioExperimentEngine(stub_universe, n_jobs=1)
    eng.add_strategy("MR", MeanRisk(), {"l2_coef": [0.0, 0.01]})
    pop = eng.run_robustness_test(
        cv_type="walk_forward",
        train_size=80,
        test_size=20,
        inner_cv_kwargs={"freq": "ME"},
    )
    assert len(pop) == 1


def test_online_evaluation_all_positive_returns_prefers_high_return_candidate() -> None:
    """Sprint 5 (M2): OnlineGridSearch handles zero/negative CVaR folds without inverted ranking."""
    from skfolio.moments import EWCovariance, EWMu
    from skfolio.prior import EmpiricalPrior

    rng = np.random.default_rng(121)
    dates = pd.bdate_range("2023-01-02", periods=160)
    # SAFE_GAIN has strictly positive returns (negative CVaR -> negative raw CVAR_RATIO!)
    # NOISY_LOW has lower mean + small occasional loss (tiny positive CVaR -> huge positive raw CVAR_RATIO!)
    rets = pd.DataFrame(
        {
            "SAFE_GAIN": rng.uniform(0.0020, 0.0040, 160),
            "NOISY_LOW": rng.uniform(-0.0005, 0.0020, 160),
        },
        index=dates,
    )
    u = Universe(
        ["SAFE_GAIN", "NOISY_LOW"],
        {"SAFE_GAIN": "core", "NOISY_LOW": "core"},
        {"SAFE_GAIN": 0.0, "NOISY_LOW": 0.0},
    )
    u._returns = rets
    est = MeanRisk(
        prior_estimator=EmpiricalPrior(
            mu_estimator=EWMu(), covariance_estimator=EWCovariance()
        )
    )
    eng = PortfolioExperimentEngine(u, n_jobs=1)
    # Put NOISY_LOW option first so ties or inverted scores pick it
    eng.add_strategy(
        "PosOn",
        est,
        {
            "min_weights": [
                {"SAFE_GAIN": 0.0, "NOISY_LOW": 1.0},
                {"SAFE_GAIN": 1.0, "NOISY_LOW": 0.0},
            ]
        },
    )
    res = eng.run_online_evaluation(burn_in_size=80, test_size=20, return_result=True)
    # Strictly dominant SAFE_GAIN (positive returns in [0.2%, 0.4%]/day) must beat NOISY_LOW ([-0.05%, 0.2%]/day)
    assert res.live_weights["PosOn"]["SAFE_GAIN"] >= 0.99


def test_black_litterman_risk_free_rate_propagates_to_outer_multi_period_portfolio() -> (
    None
):
    """Sprint 5 (#6b gap): BL risk_free_rate propagates to outer MultiPeriodPortfolio Sharpe in run_robustness_test."""
    from skfolio import Portfolio
    from skfolio.optimization import HierarchicalRiskParity
    from skfolio.prior import EmpiricalPrior

    from flowportfolio.priors import PriorSynthesiser
    from flowportfolio.strategies import StrategyBuilder

    rng = np.random.default_rng(131)
    dates = pd.bdate_range("2022-01-03", periods=240)
    rets = pd.DataFrame(
        rng.normal(0.0005, 0.01, (240, 3)),
        index=dates,
        columns=["A", "B", "C"],
    )
    u = Universe(
        ["A", "B", "C"],
        {"A": "core", "B": "core", "C": "sat"},
        {"A": 0.0, "B": 0.0, "C": 0.0},
    )
    u._returns = rets
    rfr_ann = 0.0504
    expected_rfr = rfr_ann / 252.0

    ps = PriorSynthesiser(u).add_market_view("A == 0.0504", confidence=0.8)
    bl_excess = ps.build_black_litterman_prior(risk_free_rate=rfr_ann)
    bl_total = ps.build_black_litterman_prior(
        risk_free_rate=rfr_ann, prior_estimator=EmpiricalPrior()
    )

    eng = PortfolioExperimentEngine(u, n_jobs=1)
    eng.add_strategy(
        "BL_MR",
        StrategyBuilder(prior=bl_excess)
        .set_optimizer(MeanRisk(objective_function=ObjectiveFunction.MAXIMIZE_RATIO))
        .build_pipeline(),
        {"optimizer__l2_coef": [0.0]},
    )
    eng.add_strategy(
        "BL_HRP",
        StrategyBuilder(prior=bl_total)
        .set_optimizer(HierarchicalRiskParity())
        .build_pipeline(),
        {},
    )
    pop = eng.run_robustness_test(cv_type="walk_forward", train_size=120, test_size=60)
    for mpp in pop:
        assert float(mpp.risk_free_rate) == pytest.approx(expected_rfr, abs=1e-12)
        oracle = Portfolio(
            X=mpp.returns[:, None],
            weights=[1.0],
            risk_free_rate=expected_rfr,
            annualization_factor=252.0,
        )
        assert float(mpp.annualized_sharpe_ratio) == pytest.approx(
            float(oracle.annualized_sharpe_ratio), rel=1e-9, abs=1e-12
        )


def test_reconcile_transaction_costs_warns_when_inner_test_diverges_from_outer_horizon(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Sprint 5 (plan gap): Warn when inner CV test_size diverges substantially from effective transaction cost horizon."""
    import logging

    from flowportfolio.core.constraints import ConstraintBuilder
    from flowportfolio.strategies import StrategyBuilder

    rng = np.random.default_rng(141)
    dates = pd.bdate_range("2022-01-03", periods=220)
    rets = pd.DataFrame(
        rng.normal(0.0005, 0.01, (220, 3)),
        index=dates,
        columns=["A", "B", "C"],
    )
    u = Universe(
        ["A", "B", "C"],
        {"A": "core", "B": "core", "C": "sat"},
        {"A": 0.0, "B": 0.0, "C": 0.0},
    )
    u._returns = rets
    spec = ConstraintBuilder(u).transaction_costs(0.005).build()
    pipe = StrategyBuilder(constraints=spec).set_optimizer(MeanRisk()).build_pipeline()

    # 1. Divergent: outer_train=150 -> derived inner_test=30 vs outer_test_size=5 (6x divergence)
    eng_div = PortfolioExperimentEngine(u, n_jobs=1)
    eng_div.add_strategy("DivTC", pipe, {"optimizer__l2_coef": [0.0]})
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="flowportfolio.core.experiment"):
        eng_div.run_robustness_test(cv_type="walk_forward", train_size=150, test_size=5)
    assert any(
        "inner" in rec.message.lower() and "transaction_costs" in rec.message.lower()
        for rec in caplog.records
    )

    # 2. Matched: outer_train=150 -> derived inner_test=30 == outer_test_size=30 -> no inner divergence warning
    eng_match = PortfolioExperimentEngine(u, n_jobs=1)
    eng_match.add_strategy("MatchTC", pipe, {"optimizer__l2_coef": [0.0]})
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="flowportfolio.core.experiment"):
        eng_match.run_robustness_test(
            cv_type="walk_forward", train_size=150, test_size=30
        )
    assert not any(
        "inner" in rec.message.lower() and "transaction_costs" in rec.message.lower()
        for rec in caplog.records
    )


def test_walk_forward_sweeping_max_turnover_and_transaction_costs_neutralizes_fold0() -> (
    None
):
    """Sweeping max_turnover or transaction_costs in config['grid'] must neutralize Fold 0 without crashing or charging entry cost."""
    import warnings

    from flowportfolio.core.constraints import ConstraintBuilder
    from flowportfolio.delta import PortfolioDeltaEngine
    from flowportfolio.strategies import StrategyBuilder

    rng = np.random.default_rng(142)
    dates = pd.bdate_range("2022-01-03", periods=200)
    rets = pd.DataFrame(
        rng.normal(0.0005, 0.01, (200, 3)),
        index=dates,
        columns=["A", "B", "C"],
    )
    u = Universe(
        ["A", "B", "C"],
        {"A": "core", "B": "core", "C": "sat"},
        {"A": 0.0, "B": 0.0, "C": 0.0},
    )
    u._returns = rets
    cw = {"A": 0.34, "B": 0.33, "C": 0.33}

    # 1. Sweeping max_turnover=[0.1, 0.2] on 3 assets in walk_forward must not crash Fold 0
    #    or emit parallel-disabled UserWarning, and must enforce max_turnover on Fold 1+ and live refit.
    pipe_turn = (
        StrategyBuilder(constraints=ConstraintBuilder(u).build())
        .set_optimizer(MeanRisk())
        .build_pipeline()
    )
    eng_turn = PortfolioExperimentEngine(u, n_jobs=-1)
    eng_turn.add_strategy(
        "SweepTurnover", pipe_turn, {"optimizer__max_turnover": [0.1, 0.2]}
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res_turn = eng_turn.run_robustness_test(
            cv_type="walk_forward",
            train_size=100,
            test_size=40,
            inner_cv_kwargs={"train_size": 40, "test_size": 20},
            return_result=True,
        )
    assert not any(
        "Parallel processing has been disabled" in str(w.message) for w in caught
    )
    mpp_turn = res_turn.population[0]
    assert len(mpp_turn.portfolios) == 2
    assert "SweepTurnover" in res_turn.live_weights
    assert sum(res_turn.live_weights["SweepTurnover"].values()) == pytest.approx(
        1.0, abs=1e-6
    )

    # Also verify combinatorial CV with grid-swept max_turnover does not crash with infeasibility
    pop_cpcv = eng_turn.run_robustness_test(
        cv_type="combinatorial",
        n_folds=4,
        n_test_folds=2,
        purged_size=2,
        inner_cv_kwargs={"train_size": 40, "test_size": 20},
    )
    assert len(pop_cpcv) > 0

    # 2. Sweeping transaction_costs=[0.005, 0.01] on a clean base estimator:
    #    Fold 0 must have total_cost == 0.0, Fold 1 must have positive cost,
    #    and PortfolioDeltaEngine must charge positive external entry_drag.
    pipe_tc = (
        StrategyBuilder(constraints=ConstraintBuilder(u).build())
        .set_optimizer(MeanRisk())
        .build_pipeline()
    )
    eng_tc = PortfolioExperimentEngine(u, n_jobs=-1)
    eng_tc.add_strategy(
        "SweepTC", pipe_tc, {"optimizer__transaction_costs": [0.005, 0.01]}
    )
    pop_tc = eng_tc.run_robustness_test(
        cv_type="walk_forward",
        train_size=100,
        test_size=40,
        inner_cv_kwargs={"train_size": 40, "test_size": 20},
    )
    mpp_tc = pop_tc[0]
    assert float(mpp_tc.portfolios[0].total_cost) == pytest.approx(0.0, abs=1e-12)
    assert float(mpp_tc.portfolios[1].total_cost) > 0.0
    de = PortfolioDeltaEngine(u, current_weights=cw, target_weights=cw)
    delta_res = de.calculate_rebalance_delta(
        pop_tc, strategy_tag="SweepTC", brokerage_bps=50.0
    )
    assert delta_res["rebalance"]["entry_drag"] > 0.0

    # 3. Base estimator ALREADY has transaction_costs=0.005 AND grid ALSO sweeps transaction_costs:
    #    Fold 0 must STILL be neutralized (total_cost == 0.0), not overwritten by best_params_.
    pipe_both = (
        StrategyBuilder(
            constraints=ConstraintBuilder(u)
            .transaction_costs(0.005, holding_period=40)
            .build()
        )
        .set_optimizer(MeanRisk())
        .build_pipeline()
    )
    eng_both = PortfolioExperimentEngine(u, n_jobs=1)
    eng_both.add_strategy(
        "BaseAndSweepTC", pipe_both, {"optimizer__transaction_costs": [0.005, 0.01]}
    )
    pop_both = eng_both.run_robustness_test(
        cv_type="walk_forward",
        train_size=100,
        test_size=40,
        inner_cv_kwargs={"train_size": 40, "test_size": 20},
    )
    mpp_both = pop_both[0]
    assert float(mpp_both.portfolios[0].total_cost) == pytest.approx(0.0, abs=1e-12)
    assert float(mpp_both.portfolios[1].total_cost) > 0.0


def test_online_evaluation_sweeping_friction_neutralizes_fold0() -> None:
    """run_online_evaluation must also neutralize Fold 0 when config['grid'] sweeps max_turnover or transaction_costs."""
    from skfolio.moments import EWCovariance, EWMu
    from skfolio.prior import EmpiricalPrior

    from flowportfolio.core.constraints import ConstraintBuilder
    from flowportfolio.strategies import StrategyBuilder

    rng = np.random.default_rng(143)
    dates = pd.bdate_range("2022-01-03", periods=160)
    rets = pd.DataFrame(
        rng.normal(0.0005, 0.01, (160, 3)),
        index=dates,
        columns=["A", "B", "C"],
    )
    u = Universe(
        ["A", "B", "C"],
        {"A": "core", "B": "core", "C": "sat"},
        {"A": 0.0, "B": 0.0, "C": 0.0},
    )
    u._returns = rets
    online_prior = EmpiricalPrior(
        mu_estimator=EWMu(half_life=10),
        covariance_estimator=EWCovariance(half_life=10),
    )
    pipe = (
        StrategyBuilder(constraints=ConstraintBuilder(u).build(), prior=online_prior)
        .set_optimizer(MeanRisk())
        .build_pipeline()
    )
    eng = PortfolioExperimentEngine(u, n_jobs=1)
    eng.add_strategy(
        "OnlineSweepFriction",
        pipe,
        {
            "optimizer__max_turnover": [0.1, 0.2],
            "optimizer__transaction_costs": [0.005, 0.01],
        },
    )
    pop = eng.run_online_evaluation(
        cv_type="walk_forward",
        warmup_size=80,
        test_size=20,
    )
    mpp = pop[0]
    assert len(mpp.portfolios) >= 2
    assert float(mpp.portfolios[0].total_cost) == pytest.approx(0.0, abs=1e-12)
    assert float(mpp.portfolios[1].transaction_costs) in (0.005, 0.01)
