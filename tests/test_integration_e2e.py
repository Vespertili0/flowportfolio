"""Unmocked ground-truth end-to-end integration test harness.

This test module verifies the complete flowportfolio analytics pipeline
using deterministic synthetic data with zero mocks of skfolio:
Universe -> ConstraintBuilder -> StrategyBuilder -> PortfolioExperimentEngine
-> PortfolioDeltaEngine -> PersistenceManager -> Reporter.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from skfolio import Population, RiskMeasure
from skfolio.optimization import MeanRisk
from skfolio.portfolio import BasePortfolio, MultiPeriodPortfolio

from flowportfolio.core.constraints import ConstraintBuilder
from flowportfolio.core.experiment import PortfolioExperimentEngine
from flowportfolio.core.reporting import Reporter
from flowportfolio.core.universe import Universe
from flowportfolio.delta import PortfolioDeltaEngine
from flowportfolio.persistence import PersistenceManager
from flowportfolio.strategies import StrategyBuilder


@pytest.fixture
def deterministic_universe() -> Universe:
    """Construct a deterministic synthetic 3-asset Universe with pre-computed returns.

    Bypasses external yfinance network retrieval by directly populating
    the internal returns and prices states with a seeded covariance structure.
    """
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

    # Generate 250 daily returns with deterministic multivariate normal structure
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=250, freq="B")
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


def test_unmocked_pipeline_walk_forward(
    deterministic_universe: Universe,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execute the full lifecycle under WalkForward cross-validation with zero mocks of skfolio."""
    universe = deterministic_universe

    # 1. ConstraintBuilder
    cb = ConstraintBuilder(universe)
    cb.min_group("equity", 0.30).max_group("bond", 0.60)
    spec = cb.build()

    assert len(spec.linear_constraints) == 2
    assert "equity" in spec.groups["ASSET_A"]
    assert "bond" in spec.groups["ASSET_C"]

    # 2. StrategyBuilder
    sb_baseline = StrategyBuilder(constraints=spec)
    sb_baseline.set_optimizer(MeanRisk(risk_measure=RiskMeasure.VARIANCE))
    baseline_pipeline = sb_baseline.build_pipeline()

    sb_candidate = StrategyBuilder(constraints=spec)
    sb_candidate.set_optimizer(MeanRisk(risk_measure=RiskMeasure.CVAR))
    candidate_pipeline = sb_candidate.build_pipeline()

    # 3. PortfolioExperimentEngine
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

    population = engine.run_robustness_test(
        cv_type="walk_forward",
        train_size=120,
        test_size=30,
    )

    assert isinstance(population, Population)
    assert len(population) == 2
    for portfolio in population:
        assert isinstance(portfolio, MultiPeriodPortfolio)
        assert portfolio.tag in {"Baseline", "Candidate"}
        assert len(portfolio.portfolios) > 0

    # 4. PortfolioDeltaEngine
    current_weights = {"ASSET_A": 0.40, "ASSET_B": 0.30, "ASSET_C": 0.30}
    target_weights = {"ASSET_A": 0.50, "ASSET_B": 0.25, "ASSET_C": 0.25}

    delta_engine = PortfolioDeltaEngine(
        universe,
        current_weights=current_weights,
        target_weights=target_weights,
    )
    friction_df = delta_engine.calculate_net_friction(
        brokerage_bps=10.0, slippage_bps=5.0
    )
    assert "TOTAL" in friction_df["ticker"].values
    assert (
        friction_df.loc[friction_df["ticker"] == "TOTAL", "total_friction"].iloc[0] > 0
    )

    rebalance_metrics = delta_engine.calculate_rebalance_delta(population=population)
    assert "hold" in rebalance_metrics
    assert "rebalance" in rebalance_metrics
    for scenario in ("hold", "rebalance"):
        assert np.isfinite(rebalance_metrics[scenario]["cvar"])
        assert np.isfinite(rebalance_metrics[scenario]["sharpe"])
        assert np.isfinite(rebalance_metrics[scenario]["max_drawdown"])

    # 5. PersistenceManager
    pm = PersistenceManager()
    snapshot_path = tmp_path / "walk_forward_snapshot.json"
    pm.save_snapshot(population, str(snapshot_path))

    assert snapshot_path.exists()
    with open(snapshot_path, encoding="utf-8") as f:
        snapshot_data = json.load(f)
    assert "timestamp" in snapshot_data
    assert len(snapshot_data["portfolios"]) == 2
    for p_record in snapshot_data["portfolios"]:
        assert p_record["tag"] in {"Baseline", "Candidate"}
        assert "weights" in p_record

    # 6. Reporter
    reporter = Reporter(population)
    figures = reporter.get_plotly_figures(baseline_tag="Baseline")
    assert "cvar_boxplot" in figures
    assert "drawdown_distribution" in figures
    assert "best_composition" in figures

    # Suppress interactive window popup in headless automated testing
    show_mock = MagicMock()
    monkeypatch.setattr(go.Figure, "show", show_mock)

    reporter.plot_stress_impact("Candidate", "Baseline")
    assert show_mock.call_count == 2

    reporter.generate_tearsheet(baseline_tag="Baseline")


def test_unmocked_pipeline_combinatorial(
    deterministic_universe: Universe,
) -> None:
    """Execute unmocked pipeline with CombinatorialPurgedCV to verify collection flattening."""
    universe = deterministic_universe

    sb_baseline = StrategyBuilder()
    sb_baseline.set_optimizer(MeanRisk(risk_measure=RiskMeasure.VARIANCE))
    baseline_pipeline = sb_baseline.build_pipeline()

    sb_candidate = StrategyBuilder()
    sb_candidate.set_optimizer(MeanRisk(risk_measure=RiskMeasure.CVAR))
    candidate_pipeline = sb_candidate.build_pipeline()

    engine = PortfolioExperimentEngine(universe, n_jobs=1)
    engine.add_strategy(
        "Baseline",
        baseline_pipeline,
        {"optimizer__l1_coef": [0.0]},
    )
    engine.add_strategy(
        "Candidate",
        candidate_pipeline,
        {"optimizer__l1_coef": [0.0]},
    )

    # CombinatorialPurgedCV generates multi-path populations
    comb_population = engine.run_robustness_test(
        cv_type="combinatorial",
        n_folds=4,
        n_test_folds=2,
    )

    assert isinstance(comb_population, Population)
    # Verify flattening: all elements must be BasePortfolio instances, no nested Population
    assert len(comb_population) > 2
    for item in comb_population:
        assert isinstance(item, BasePortfolio)
        assert not isinstance(item, Population)
        assert item.tag in {"Baseline", "Candidate"}
