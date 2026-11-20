"""Unmocked unit tests for the Reporter class.

Verifies visual tearsheet generation, Markdown artifact export, metric
extraction, finite-ratio ranking, live-weight preservation, and warning-free
handling of degenerate portfolios using real ``skfolio`` objects.
"""

from __future__ import annotations

import inspect
import sys
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from skfolio import MultiPeriodPortfolio, Population, Portfolio

from flowportfolio.core import reporting as reporting_module
from flowportfolio.core.experiment import ExperimentResult
from flowportfolio.core.reporting import Reporter


def _make_portfolio(
    tag: str,
    *,
    name: str = "port",
    mean: float = 0.001,
    std: float = 0.01,
    n_days: int = 30,
    seed: int = 42,
    weights: tuple[float, ...] = (0.5, 0.5),
    columns: list[str] | None = None,
    start: str = "2025-01-02",
    risk_free_rate: float = 0.0,
) -> Portfolio:
    rng = np.random.default_rng(seed)
    n_assets = len(weights)
    cols = columns if columns is not None else [f"A{i}" for i in range(n_assets)]
    dates = pd.bdate_range(start, periods=n_days)
    X = pd.DataFrame(
        rng.normal(mean, std, size=(n_days, n_assets)),
        index=dates,
        columns=cols,
    )
    return Portfolio(
        X=X,
        weights=np.array(weights, dtype=float),
        name=name,
        tag=tag,
        risk_free_rate=risk_free_rate,
    )


def _make_multi_period(
    tag: str,
    *,
    name: str = "mpp",
    n_folds: int = 4,
    fold_len: int = 12,
    mean: float = 0.001,
    std: float = 0.01,
    seed: int = 42,
    weights: tuple[float, ...] = (0.6, 0.4),
    columns: list[str] | None = None,
) -> MultiPeriodPortfolio:
    rng = np.random.default_rng(seed)
    cols = columns if columns is not None else ["AAPL", "MSFT"]
    total = n_folds * fold_len
    dates = pd.bdate_range("2025-01-02", periods=total)
    X = pd.DataFrame(
        rng.normal(mean, std, size=(total, len(weights))),
        index=dates,
        columns=cols,
    )
    folds = [
        Portfolio(
            X=X.iloc[i * fold_len : (i + 1) * fold_len],
            weights=np.array(weights, dtype=float),
            name=f"{name}_f{i}",
        )
        for i in range(n_folds)
    ]
    return MultiPeriodPortfolio(portfolios=folds, name=name, tag=tag)


@pytest.fixture
def dummy_population() -> Population:
    """Fixture providing a real deterministic Population of Portfolio instances."""
    p1 = _make_portfolio(
        "Baseline",
        name="port1",
        mean=0.0005,
        std=0.01,
        seed=10,
        weights=(0.3, 0.3, 0.4),
        columns=["A", "B", "C"],
    )
    p2 = _make_portfolio(
        "StratA",
        name="port2",
        mean=0.003,
        std=0.008,
        seed=20,
        weights=(0.5, 0.5, 0.0),
        columns=["A", "B", "C"],
    )
    return Population([p1, p2])


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def test_init_valid(dummy_population: Population) -> None:
    """Reporter accepts a real Population instance."""
    reporter = Reporter(dummy_population)
    assert reporter._population is dummy_population


def test_init_wrong_type() -> None:
    """Reporter raises TypeError for non-Population argument."""
    with pytest.raises(TypeError, match="population must be a skfolio.Population"):
        Reporter(population="not_a_population")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Tearsheet & Headless Plotly Figures (Real skfolio objects)
# ---------------------------------------------------------------------------


def test_generate_tearsheet_renders_and_shows_all_three_figures(
    dummy_population: Population, monkeypatch: pytest.MonkeyPatch
) -> None:
    """generate_tearsheet builds all three Plotly figures and invokes .show() on each."""
    shown: list[go.Figure] = []
    monkeypatch.setattr(go.Figure, "show", lambda self, *a, **kw: shown.append(self))

    reporter = Reporter(dummy_population)
    reporter.generate_tearsheet(baseline_tag="Baseline")

    assert len(shown) == 3
    assert all(isinstance(fig, go.Figure) for fig in shown)


def test_tag_list_excludes_absent_baseline_and_honors_custom_baseline() -> None:
    """Baseline tag is only included when present; custom baseline tags are honored."""
    p_a = _make_portfolio("StratA", name="a", seed=1)
    reporter_no_base = Reporter(Population([p_a]))
    figs = reporter_no_base.get_plotly_figures(baseline_tag="Baseline")
    trace_names = [t.name for t in figs["cvar_boxplot"].data]
    assert "Baseline" not in trace_names
    assert "StratA" in trace_names

    p_custom = _make_portfolio("MyBaseline", name="mb", seed=2)
    reporter_custom = Reporter(Population([p_custom]))
    figs_custom = reporter_custom.get_plotly_figures(baseline_tag="MyBaseline")
    custom_names = [t.name for t in figs_custom["cvar_boxplot"].data]
    assert "Baseline" not in custom_names
    assert "MyBaseline" in custom_names


def test_best_strategy_selection_by_median_cvar_ratio() -> None:
    """Strategy with highest median CVAR_RATIO is selected for composition and markdown."""
    p_base = _make_portfolio(
        "Baseline", name="base", mean=-0.001, std=0.01, seed=1, weights=(0.5, 0.5)
    )
    pa1 = _make_portfolio(
        "StratA", name="a1", mean=0.0002, std=0.015, seed=2, weights=(0.8, 0.2)
    )
    pa2 = _make_portfolio(
        "StratA", name="a2", mean=0.0003, std=0.015, seed=3, weights=(0.8, 0.2)
    )
    pb1 = _make_portfolio(
        "StratB", name="b1", mean=0.003, std=0.010, seed=4, weights=(0.15, 0.85)
    )
    pb2 = _make_portfolio(
        "StratB", name="b2", mean=0.0035, std=0.010, seed=5, weights=(0.15, 0.85)
    )

    assert np.median([pb1.cvar_ratio, pb2.cvar_ratio]) > np.median(
        [pa1.cvar_ratio, pa2.cvar_ratio]
    )

    pop = Population([p_base, pa1, pa2, pb1, pb2])
    reporter = Reporter(pop)

    assert (
        reporter._best_tag_by_median_cvar(["Baseline", "StratA", "StratB"]) == "StratB"
    )
    report = reporter.to_markdown_artifact(baseline_tag="Baseline")
    assert "**Best Tag (by median CVaR Ratio):** StratB" in report
    assert "A1 (85.00%)" in report

    figs = reporter.get_plotly_figures(baseline_tag="Baseline")
    assert isinstance(figs["best_composition"], go.Figure)
    assert len(figs["best_composition"].data) > 0


def test_get_plotly_figures_returns_dict_without_calling_show(
    dummy_population: Population, monkeypatch: pytest.MonkeyPatch
) -> None:
    """get_plotly_figures returns exact keys without invoking .show()."""
    called = False

    def _fail_show(self, *args, **kwargs) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(go.Figure, "show", _fail_show)

    reporter = Reporter(dummy_population)
    figs = reporter.get_plotly_figures()

    assert set(figs.keys()) == {
        "cvar_boxplot",
        "drawdown_distribution",
        "best_composition",
    }
    assert all(isinstance(fig, go.Figure) for fig in figs.values())
    assert not called


def test_get_plotly_figures_empty() -> None:
    """get_plotly_figures raises ValueError on empty population."""
    reporter = Reporter(Population([]))
    with pytest.raises(ValueError, match="Population contains no tagged portfolios."):
        reporter.get_plotly_figures()


# ---------------------------------------------------------------------------
# Stress Impact
# ---------------------------------------------------------------------------


def test_plot_stress_impact_success_and_titles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """plot_stress_impact produces titled CVaR and Max Drawdown figures and calls .show()."""
    shown: list[go.Figure] = []
    monkeypatch.setattr(go.Figure, "show", lambda self, *a, **kw: shown.append(self))

    p_baseline = _make_portfolio("Baseline", name="b", mean=0.001, seed=11)
    p_stress = _make_portfolio("Stress", name="s", mean=-0.002, std=0.02, seed=12)
    reporter = Reporter(Population([p_baseline, p_stress]))

    reporter.plot_stress_impact(stress_tag="Stress", baseline_tag="Baseline")

    assert len(shown) == 2
    assert shown[0].layout.title.text == "CVaR Distribution: Stress vs Baseline"
    assert shown[1].layout.title.text == "Max Drawdown Distribution: Stress vs Baseline"


def test_plot_stress_impact_missing_tags() -> None:
    """plot_stress_impact raises ValueError if either tag is missing."""
    p_baseline = _make_portfolio("Baseline", name="b", seed=11)
    with pytest.raises(
        ValueError, match="Tag 'Stress' matched no portfolios in the population."
    ):
        Reporter(Population([p_baseline])).plot_stress_impact(
            stress_tag="Stress", baseline_tag="Baseline"
        )

    p_stress = _make_portfolio("Stress", name="s", seed=12)
    with pytest.raises(
        ValueError, match="Tag 'Baseline' matched no portfolios in the population."
    ):
        Reporter(Population([p_stress])).plot_stress_impact(
            stress_tag="Stress", baseline_tag="Baseline"
        )


# ---------------------------------------------------------------------------
# Markdown & Metrics DataFrame
# ---------------------------------------------------------------------------


def test_to_markdown_artifact_real_population(dummy_population: Population) -> None:
    """to_markdown_artifact formats metrics, best strategy, holdings, and baseline."""
    reporter = Reporter(dummy_population)
    report = reporter.to_markdown_artifact(baseline_tag="Baseline")

    assert "# Portfolio Population Report" in report
    assert "**Generated:**" in report
    assert "port1" in report
    assert "port2" in report
    assert "**Best Tag (by median CVaR Ratio):** StratA" in report
    assert "**Top 5 Holdings:**" in report
    assert "**Baseline Tag:** Baseline" in report


def test_to_markdown_artifact_empty() -> None:
    """to_markdown_artifact raises ValueError on empty population."""
    reporter = Reporter(Population([]))
    with pytest.raises(ValueError, match="Population is empty."):
        reporter.to_markdown_artifact()


def test_to_markdown_artifact_without_tabulate(dummy_population: Population) -> None:
    """Native Markdown table fallback works when tabulate is unavailable."""
    reporter = Reporter(dummy_population)
    with patch.dict(sys.modules, {"tabulate": None}):
        report = reporter.to_markdown_artifact()

    assert "| Strategy" in report
    assert "| Tag" in report
    assert "| Sharpe" in report
    assert "--- | ---" in report


def test_to_markdown_artifact_unlabelled_assets() -> None:
    """Portfolios without string column names format cleanly without crashing."""
    rng = np.random.default_rng(7)
    df = pd.DataFrame(rng.normal(0.001, 0.01, (20, 3)))
    df.index = pd.bdate_range("2025-01-02", periods=20)
    p = Portfolio(
        X=df, weights=np.array([0.3, 0.3, 0.4]), name="port_unlabelled", tag="StratB"
    )

    report = Reporter(Population([p])).to_markdown_artifact()
    assert "Asset_0" in report or "0 (" in report
    assert "port_unlabelled" in report


def test_to_markdown_artifact_multi_period_portfolio() -> None:
    """Formatting works when best portfolio is a MultiPeriodPortfolio."""
    mp = _make_multi_period("StratA", name="mp_port", weights=(0.6, 0.4))
    report = Reporter(Population([mp])).to_markdown_artifact()
    assert "AAPL (60.00%)" in report
    assert "MSFT (40.00%)" in report


def test_extract_metrics_dataframe(dummy_population: Population) -> None:
    """extract_metrics_dataframe returns exact columns and real annualized metrics."""
    reporter = Reporter(dummy_population)
    df = reporter.extract_metrics_dataframe()

    expected_cols = [
        "name",
        "tag",
        "sharpe",
        "sortino",
        "cvar",
        "max_drawdown",
        "cvar_ratio",
        "mean_return",
    ]
    assert list(df.columns) == expected_cols
    assert len(df) == 2
    assert df.iloc[0]["name"] == "port1"
    assert df.iloc[0]["sharpe"] == pytest.approx(
        float(dummy_population[0].annualized_sharpe_ratio)
    )
    assert df.iloc[0]["mean_return"] == pytest.approx(
        float(dummy_population[0].annualized_mean)
    )


def test_extract_metrics_dataframe_empty() -> None:
    """extract_metrics_dataframe raises ValueError on empty population."""
    reporter = Reporter(Population([]))
    with pytest.raises(ValueError, match="Population is empty. No metrics to extract."):
        reporter.extract_metrics_dataframe()


def test_reporter_consistent_trajectory_cvar_and_no_tag_mutation() -> None:
    """Unrolling clones children without mutating their original tags."""
    rng = np.random.default_rng(13)
    df = pd.DataFrame(
        rng.normal(0.001, 0.01, (24, 2)),
        columns=["SPY", "TLT"],
        index=pd.bdate_range("2025-01-02", periods=24),
    )
    child1 = Portfolio(X=df.iloc[:12], weights=np.array([0.5, 0.5]), name="c1")
    child2 = Portfolio(X=df.iloc[12:], weights=np.array([0.6, 0.4]), name="c2")
    assert getattr(child1, "tag", None) is None
    assert getattr(child2, "tag", None) is None

    mp = MultiPeriodPortfolio(portfolios=[child1, child2], tag="StratA")
    reporter = Reporter(Population([mp]))
    unrolled = reporter._unroll_portfolios(reporter._population)

    assert getattr(child1, "tag", None) is None
    assert getattr(child2, "tag", None) is None
    assert unrolled[0].tag == "StratA"
    assert unrolled[1].tag == "StratA"
    assert reporter._best_tag_by_median_cvar(["StratA"]) == "StratA"


def test_reporter_annualized_metrics_and_uncensored_drawdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Annualized metrics and uncensored full-trajectory drawdown work with zero RuntimeWarnings."""
    shown: list[go.Figure] = []
    monkeypatch.setattr(go.Figure, "show", lambda self, *a, **kw: shown.append(self))

    dates = pd.bdate_range("2026-01-01", periods=40)
    ret_df = pd.DataFrame({"A": [-0.01] * 40, "B": [-0.01] * 40}, index=dates)
    folds = [
        Portfolio(
            X=ret_df.iloc[i * 10 : (i + 1) * 10],
            weights=np.array([0.5, 0.5]),
            name=f"p{i}",
        )
        for i in range(4)
    ]
    mp = MultiPeriodPortfolio(portfolios=folds, name="MPP", tag="StratA")
    mp_baseline = MultiPeriodPortfolio(
        portfolios=folds[:2], name="BaseMPP", tag="Baseline"
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reporter = Reporter(Population([mp]))
        metrics_df = reporter.extract_metrics_dataframe()
        stress_reporter = Reporter(Population([mp, mp_baseline]))
        stress_reporter.plot_stress_impact("StratA", "Baseline")

    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert runtime_warnings == [], f"Unexpected RuntimeWarnings: {runtime_warnings}"

    with np.errstate(divide="ignore", invalid="ignore"):
        assert np.isclose(metrics_df["sharpe"].iloc[0], mp.annualized_sharpe_ratio)
        assert np.isclose(metrics_df["sortino"].iloc[0], mp.annualized_sortino_ratio)
        assert np.isclose(metrics_df["mean_return"].iloc[0], mp.annualized_mean)

    total_dd = float(mp.max_drawdown)
    child_max_dd = max(float(p.max_drawdown) for p in folds)
    assert total_dd > child_max_dd
    assert len(shown) == 2


# ---------------------------------------------------------------------------
# Cycle 11 (F-05, F-08, F-10): Unrolling, Finite Ranking, Live Weights & Slop
# ---------------------------------------------------------------------------


def test_get_plotly_figures_unrolls_real_multi_period_portfolio_folds() -> None:
    """F-08 (O5): Real 4-fold MultiPeriodPortfolio yields 4 points in cvar_boxplot (not 1)."""
    mp_a = _make_multi_period("StratA", name="A", n_folds=4, fold_len=12, seed=101)
    mp_b = _make_multi_period("Baseline", name="B", n_folds=4, fold_len=12, seed=202)
    pop = Population([mp_a, mp_b])

    reporter = Reporter(pop)
    figs = reporter.get_plotly_figures(baseline_tag="Baseline")

    # Default unroll_folds=True produces 4 fold-level observations per strategy
    traces_by_name = {t.name: t for t in figs["cvar_boxplot"].data}
    assert len(traces_by_name["StratA"].y) == 4
    assert len(traces_by_name["Baseline"].y) == 4

    # Explicit unroll_folds=False preserves aggregate single-trajectory points
    figs_agg = reporter.get_plotly_figures(baseline_tag="Baseline", unroll_folds=False)
    agg_by_name = {t.name: t for t in figs_agg["cvar_boxplot"].data}
    assert len(agg_by_name["StratA"].y) == 1
    assert len(agg_by_name["Baseline"].y) == 1


def test_best_strategy_ignores_infinite_cvar_ratio_and_fails_when_all_nonfinite() -> (
    None
):
    """F-08 (O2): Degenerate +inf CVaR ratio (underperforming risk-free rate) cannot beat finite strategies."""
    dates = pd.bdate_range("2025-01-02", periods=40)
    # 90% +0.002, 10% 0.0 -> CVaR = -0.0; with risk_free_rate=0.005 > mean, cvar_ratio == +inf!
    deg_df = pd.DataFrame(
        {"A": [0.0] * 4 + [0.002] * 36, "B": [0.0] * 4 + [0.002] * 36}, index=dates
    )
    p_inf = Portfolio(
        X=deg_df,
        weights=np.array([1.0, 0.0]),
        name="degenerate_inf",
        tag="Degenerate",
        risk_free_rate=0.005,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        assert np.isposinf(p_inf.cvar_ratio)

    p_real = _make_portfolio(
        "RealStrat",
        name="real_strat",
        mean=0.002,
        std=0.01,
        n_days=40,
        seed=55,
        weights=(0.7, 0.3),
        columns=["A", "B"],
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reporter = Reporter(Population([p_inf, p_real]))
        best_tag = reporter._best_tag_by_median_cvar(["Degenerate", "RealStrat"])
        report = reporter.to_markdown_artifact(baseline_tag="Degenerate")
        figs = reporter.get_plotly_figures(baseline_tag="Degenerate")

    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert runtime_warnings == []
    assert best_tag == "RealStrat"
    assert "**Best Tag (by median CVaR Ratio):** RealStrat" in report
    assert "**Baseline Median CVaR Ratio:** N/A" in report
    assert isinstance(figs["best_composition"], go.Figure)

    # All-non-finite population must raise ValueError instead of picking +inf
    with pytest.raises(ValueError, match="finite|matching"):
        Reporter(Population([p_inf]))._best_tag_by_median_cvar(["Degenerate"])


def test_reporter_preserves_live_weights_from_experiment_result_and_kwarg() -> None:
    """F-05 (O5): ExperimentResult and explicit live_weights override stale Fold-N weights after Population(list(pop))."""
    mp = _make_multi_period(
        "LiveStrat",
        name="live",
        n_folds=2,
        fold_len=12,
        weights=(0.9, 0.1),
        columns=["AAPL", "MSFT"],
    )
    rewrapped = Population(list(Population([mp])))
    live_weights = {"LiveStrat": {"AAPL": 0.15, "MSFT": 0.85}}

    # 1. Explicit live_weights kwarg
    report_kw = Reporter(rewrapped, live_weights=live_weights).to_markdown_artifact()
    assert "MSFT (85.00%)" in report_kw
    assert "AAPL (15.00%)" in report_kw

    # 2. Direct ExperimentResult input
    result = ExperimentResult(population=rewrapped, live_weights=live_weights)
    report_res = Reporter(result).to_markdown_artifact()
    assert "MSFT (85.00%)" in report_res
    assert "AAPL (15.00%)" in report_res

    # 3. Invalid live_weights raises TypeError
    with pytest.raises(TypeError, match="live_weights"):
        Reporter(rewrapped, live_weights=[("LiveStrat", {"AAPL": 1.0})])  # type: ignore[arg-type]


def test_generate_tearsheet_propagates_value_error_on_empty_population() -> None:
    """F-08: generate_tearsheet fails loudly on empty/untagged Population instead of swallowing ValueError."""
    reporter = Reporter(Population([]))
    with pytest.raises(ValueError, match="Population contains no tagged portfolios."):
        reporter.generate_tearsheet()


def test_reporting_module_free_of_mock_name_and_slots_monkey_patch() -> None:
    """F-08 & F-10: reporting.py contains zero _mock_name or dead latest_weights_dict references."""
    source = inspect.getsource(reporting_module)
    assert "_mock_name" not in source
    assert "latest_weights_dict" not in source
