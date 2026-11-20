"""Unmocked ground-truth end-to-end integration test harness.

Verifies quantitative non-leakage and architectural invariants across the full
``flowportfolio`` pipeline using deterministic synthetic data and zero mocks
of ``skfolio``:
Universe -> ConstraintBuilder -> StrategyBuilder -> PortfolioExperimentEngine
-> PortfolioDeltaEngine -> PersistenceManager -> Reporter.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from skfolio import Population, RiskMeasure
from skfolio.optimization import MeanRisk, ObjectiveFunction
from skfolio.portfolio import BasePortfolio, MultiPeriodPortfolio

from flowportfolio.core.constraints import ConstraintBuilder
from flowportfolio.core.experiment import ExperimentResult, PortfolioExperimentEngine
from flowportfolio.core.reporting import Reporter
from flowportfolio.core.universe import Universe
from flowportfolio.delta import PortfolioDeltaEngine
from flowportfolio.persistence import PersistenceManager
from flowportfolio.strategies import StrategyBuilder


@pytest.fixture
def deterministic_universe() -> Universe:
    """Construct a deterministic synthetic 3-asset Universe with pre-computed returns."""
    tickers = ["ASSET_A", "ASSET_B", "ASSET_C"]
    metadata = {
        "ASSET_A": "equity",
        "ASSET_B": "equity",
        "ASSET_C": "bond",
    }
    fees = {
        "ASSET_A": 0.0010,
        "ASSET_B": 0.0015,
        "ASSET_C": 0.0005,
    }

    universe = Universe(tickers=tickers, metadata=metadata, fees=fees)

    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2024-01-01", periods=250)
    mu = np.array([0.0005, 0.0006, 0.0002])
    cov = np.array(
        [
            [0.0004, 0.0002, 0.00005],
            [0.0002, 0.0005, 0.00003],
            [0.00005, 0.00003, 0.0001],
        ]
    )
    raw_returns = rng.multivariate_normal(mu, cov, size=250)
    returns_df = pd.DataFrame(raw_returns, index=dates, columns=tickers)

    universe._returns = returns_df
    universe._prices = (1.0 + returns_df).cumprod() * 100.0
    return universe


def test_unmocked_pipeline_walk_forward_quantitative_invariants(
    deterministic_universe: Universe,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full WalkForward lifecycle enforces OOS delta parity, live-weight preservation, multi-fold boxplots, and zero warnings."""
    shown: list[go.Figure] = []
    monkeypatch.setattr(go.Figure, "show", lambda self, *a, **kw: shown.append(self))

    universe = deterministic_universe

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        # 1. ConstraintBuilder
        spec = (
            ConstraintBuilder(universe)
            .min_group("equity", 0.30)
            .max_group("bond", 0.60)
            .build()
        )
        assert len(spec.linear_constraints) == 2

        # 2. StrategyBuilder
        baseline_pipeline = (
            StrategyBuilder(constraints=spec)
            .set_optimizer(MeanRisk(risk_measure=RiskMeasure.VARIANCE))
            .build_pipeline()
        )
        candidate_pipeline = (
            StrategyBuilder(constraints=spec)
            .set_optimizer(MeanRisk(risk_measure=RiskMeasure.CVAR))
            .build_pipeline()
        )

        # 3. PortfolioExperimentEngine with immutable ExperimentResult
        engine = PortfolioExperimentEngine(universe, n_jobs=1)
        engine.add_strategy(
            "Baseline",
            baseline_pipeline,
            {"optimizer__l1_coef": [0.0, 0.0005]},
        )
        engine.add_strategy(
            "Candidate",
            candidate_pipeline,
            {"optimizer__l1_coef": [0.0, 0.0005]},
        )

        result = engine.run_robustness_test(
            cv_type="walk_forward",
            train_size=120,
            test_size=30,
            return_result=True,
        )
        assert isinstance(result, ExperimentResult)
        assert len(result.population) == 2
        assert set(result.live_weights.keys()) == {"Baseline", "Candidate"}

        baseline_port = next(p for p in result.population if p.tag == "Baseline")
        assert isinstance(baseline_port, MultiPeriodPortfolio)
        n_folds = len(baseline_port.portfolios)
        assert n_folds >= 4

        # 4. PortfolioDeltaEngine — realized OOS parity (F-01)
        current_weights = {"ASSET_A": 0.40, "ASSET_B": 0.30, "ASSET_C": 0.30}
        target_weights = result.live_weights["Baseline"]

        delta_engine = PortfolioDeltaEngine(
            universe,
            current_weights=current_weights,
            target_weights=target_weights,
        )
        friction_df = delta_engine.calculate_net_friction(
            brokerage_bps=10.0, slippage_bps=5.0
        )
        assert (
            friction_df.loc[friction_df["ticker"] == "TOTAL", "transaction_cost"].iloc[
                0
            ]
            >= 0.0
        )

        rebalance_metrics = delta_engine.calculate_rebalance_delta(
            population=result.population, strategy_tag="Baseline"
        )
        assert rebalance_metrics["rebalance"]["sharpe"] == pytest.approx(
            float(baseline_port.annualized_sharpe_ratio), rel=1e-9, abs=1e-9
        )
        assert rebalance_metrics["rebalance"]["cvar"] == pytest.approx(
            float(baseline_port.cvar), rel=1e-9, abs=1e-9
        )
        assert rebalance_metrics["rebalance"]["max_drawdown"] == pytest.approx(
            float(baseline_port.max_drawdown), rel=1e-9, abs=1e-9
        )

        # 5. PersistenceManager — live weights survive Population re-wrap & UTC trajectory (F-05, F-09)
        pm = PersistenceManager(base_dir=tmp_path)
        rewrapped = Population(list(result.population))
        rewrapped_result = ExperimentResult(
            population=rewrapped, live_weights=result.live_weights
        )
        artifact_path = pm.export_gitops_artifact(rewrapped_result, str(tmp_path))
        saved = pm.load_snapshot(artifact_path)
        saved_by_tag = {p["tag"]: p["weights"] for p in saved["portfolios"]}
        for ticker, weight in result.live_weights["Baseline"].items():
            assert saved_by_tag["Baseline"][ticker] == pytest.approx(weight, abs=1e-12)

        traj_df = pm.calculate_trajectory_history(str(tmp_path))
        assert len(traj_df) == 2
        assert traj_df.index.tz is not None
        assert not traj_df[["ASSET_A", "ASSET_B", "ASSET_C"]].isna().any().any()

        # 6. Reporter — multi-fold unrolling & live-weights parity (F-05, F-08)
        reporter = Reporter(rewrapped_result)
        figures = reporter.get_plotly_figures(baseline_tag="Baseline")
        traces_by_name = {t.name: t for t in figures["cvar_boxplot"].data}
        assert len(traces_by_name["Baseline"].y) == n_folds
        assert len(traces_by_name["Candidate"].y) == n_folds

        markdown = reporter.to_markdown_artifact(baseline_tag="Baseline")
        assert "ASSET_" in markdown

        reporter.plot_stress_impact("Candidate", "Baseline")
        reporter.generate_tearsheet(baseline_tag="Baseline")
        assert len(shown) == 5

    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert runtime_warnings == [], f"Unexpected RuntimeWarnings: {runtime_warnings}"


def test_unmocked_pipeline_fold0_causal_isolation_under_transaction_costs() -> None:
    """F-02 (O3): Conflicting present-day holdings under transaction_costs do not leak into Fold-0 OOS weights."""
    rng = np.random.default_rng(7)
    n_days = 100
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    tlt = rng.normal(0.0005, 0.005, n_days)
    returns = pd.DataFrame(
        {
            "SPY": rng.normal(0.0002, 0.010, n_days),
            "TLT": tlt,
            "GLD": tlt + rng.normal(0.0, 1e-4, n_days),
        },
        index=dates,
    )
    universe = Universe(
        tickers=["SPY", "TLT", "GLD"],
        metadata={"SPY": "eq", "TLT": "fi", "GLD": "alt"},
        fees={"SPY": 0.0, "TLT": 0.0, "GLD": 0.0},
    )
    universe._returns = returns
    universe._prices = (1.0 + returns).cumprod() * 100.0

    fold0_weights: list[np.ndarray] = []
    for prev_holding in (
        {"SPY": 0.0, "TLT": 1.0, "GLD": 0.0},
        {"SPY": 0.0, "TLT": 0.0, "GLD": 1.0},
    ):
        spec = (
            ConstraintBuilder(universe)
            .transaction_costs(0.01, holding_period=1)
            .build()
        )
        opt = MeanRisk(
            objective_function=ObjectiveFunction.MAXIMIZE_UTILITY,
            previous_weights=prev_holding,
        )
        pipe = StrategyBuilder(constraints=spec).set_optimizer(opt).build_pipeline()
        engine = PortfolioExperimentEngine(universe, n_jobs=1)
        engine.add_strategy("Twin", pipe, {"optimizer__l2_coef": [0.0]})
        pop = engine.run_robustness_test(
            cv_type="walk_forward", train_size=50, test_size=25
        )
        fold0_weights.append(np.asarray(pop[0].portfolios[0].weights, dtype=float))

    np.testing.assert_allclose(fold0_weights[0], fold0_weights[1], atol=1e-12)


def test_unmocked_pipeline_combinatorial(
    deterministic_universe: Universe,
) -> None:
    """CombinatorialPurgedCV flattens multi-path Population and integrates cleanly with Reporter."""
    universe = deterministic_universe

    baseline_pipeline = (
        StrategyBuilder()
        .set_optimizer(MeanRisk(risk_measure=RiskMeasure.VARIANCE))
        .build_pipeline()
    )
    candidate_pipeline = (
        StrategyBuilder()
        .set_optimizer(MeanRisk(risk_measure=RiskMeasure.CVAR))
        .build_pipeline()
    )

    engine = PortfolioExperimentEngine(universe, n_jobs=1)
    engine.add_strategy("Baseline", baseline_pipeline, {"optimizer__l1_coef": [0.0]})
    engine.add_strategy("Candidate", candidate_pipeline, {"optimizer__l1_coef": [0.0]})

    result = engine.run_robustness_test(
        cv_type="combinatorial",
        n_folds=4,
        n_test_folds=2,
        return_result=True,
    )

    assert isinstance(result, ExperimentResult)
    comb_population = result.population
    assert isinstance(comb_population, Population)
    assert len(comb_population) > 2
    for item in comb_population:
        assert isinstance(item, BasePortfolio)
        assert not isinstance(item, Population)
        assert item.tag in {"Baseline", "Candidate"}

    figs = Reporter(result).get_plotly_figures(baseline_tag="Baseline")
    assert set(figs.keys()) == {
        "cvar_boxplot",
        "drawdown_distribution",
        "best_composition",
    }
