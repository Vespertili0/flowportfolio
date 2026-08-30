"""Unit tests for the Reporter class.

This module verifies the visual tearsheet generation logic in
:class:`flowportfolio.core.reporting.Reporter`. All skfolio plotting
methods are mocked to prevent rendering during tests.
"""

from unittest.mock import MagicMock, patch
import sys

import numpy as np
import pandas as pd
import pytest
from skfolio import Population, Portfolio
from skfolio.portfolio import MultiPeriodPortfolio

from flowportfolio.core.reporting import Reporter


def _make_mock_portfolio(tag: str, cvar_ratio: float = 1.0) -> MagicMock:
    p = MagicMock(spec=MultiPeriodPortfolio)
    p.__class__ = MultiPeriodPortfolio
    p.tag = tag
    p.cvar_ratio = cvar_ratio
    return p


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def test_init_valid() -> None:
    """Test Reporter accepts a valid Population instance."""
    pop = MagicMock(spec=Population)
    reporter = Reporter(pop)
    assert reporter._population is pop


def test_init_wrong_type() -> None:
    """Test Reporter raises TypeError for non-Population argument."""
    with pytest.raises(TypeError, match="population must be a skfolio.Population"):
        Reporter(population="not_a_population")  # type: ignore


# ---------------------------------------------------------------------------
# Tearsheet Generation
# ---------------------------------------------------------------------------


@patch.object(Population, "plot_composition")
@patch.object(Population, "plot_distribution")
@patch.object(Population, "boxplot_measure")
def test_generate_tearsheet_calls_all_plots(
    mock_boxplot: MagicMock,
    mock_distribution: MagicMock,
    mock_composition: MagicMock,
) -> None:
    """Test generate_tearsheet calls the three required plot methods."""
    # Setup mocks to return dummy figures
    fig_mock = MagicMock()
    mock_boxplot.return_value = fig_mock
    mock_distribution.return_value = fig_mock
    mock_composition.return_value = fig_mock

    # Setup population with dummy portfolios
    p1 = _make_mock_portfolio("Baseline", 1.0)

    pop = MagicMock(spec=Population)
    pop.__iter__.return_value = [p1]
    reporter = Reporter(pop)
    reporter.generate_tearsheet()

    # Verify calls
    pop.boxplot_measure.assert_called_once()
    pop.plot_distribution.assert_called_once()
    mock_composition.assert_called_once()


@patch.object(Population, "plot_composition")
@patch.object(Population, "plot_distribution")
@patch.object(Population, "boxplot_measure")
def test_generate_tearsheet_calls_show(
    mock_boxplot: MagicMock,
    mock_distribution: MagicMock,
    mock_composition: MagicMock,
) -> None:
    """Test that .show() is called on all three returned figures."""
    fig1 = MagicMock()
    fig2 = MagicMock()
    fig3 = MagicMock()

    mock_boxplot.return_value = fig1
    mock_distribution.return_value = fig2
    mock_composition.return_value = fig3

    p1 = _make_mock_portfolio("Baseline", 1.0)

    pop = MagicMock(spec=Population)
    pop.__iter__.return_value = [p1]
    pop.boxplot_measure.return_value = fig1
    pop.plot_distribution.return_value = fig2

    reporter = Reporter(pop)
    reporter.generate_tearsheet()

    fig1.show.assert_called_once()
    fig2.show.assert_called_once()
    fig3.show.assert_called_once()


@patch.object(Population, "plot_composition")
@patch.object(Population, "plot_distribution")
@patch.object(Population, "boxplot_measure")
def test_tag_list_includes_baseline(
    mock_boxplot: MagicMock,
    mock_distribution: MagicMock,
    mock_composition: MagicMock,
) -> None:
    """Test baseline_tag is always in the tag_list."""
    fig_mock = MagicMock()
    mock_boxplot.return_value = fig_mock
    mock_distribution.return_value = fig_mock
    mock_composition.return_value = fig_mock

    # Population without Baseline
    p1 = _make_mock_portfolio("StratA", 1.0)

    pop = MagicMock(spec=Population)
    pop.__iter__.return_value = [p1]
    reporter = Reporter(pop)
    reporter.generate_tearsheet(baseline_tag="Baseline")

    # Verify tag_list passed to boxplot
    boxplot_kwargs = pop.boxplot_measure.call_args[1]
    assert "Baseline" in boxplot_kwargs["tag_list"]
    assert "StratA" in boxplot_kwargs["tag_list"]


@patch.object(Population, "plot_composition", autospec=True)
@patch.object(Population, "plot_distribution")
@patch.object(Population, "boxplot_measure")
def test_best_strategy_selection(
    mock_boxplot: MagicMock,
    mock_distribution: MagicMock,
    mock_composition: MagicMock,
) -> None:
    """Test strategy with highest median CVAR_RATIO is used for composition plot."""
    fig_mock = MagicMock()
    mock_boxplot.return_value = fig_mock
    mock_distribution.return_value = fig_mock
    mock_composition.return_value = fig_mock

    # StratA has lower CVAR_RATIO (0.5)
    pa1 = _make_mock_portfolio("StratA", 0.4)
    pa2 = _make_mock_portfolio("StratA", 0.6)

    # StratB has higher CVAR_RATIO (1.5)
    pb1 = _make_mock_portfolio("StratB", 1.4)
    pb2 = _make_mock_portfolio("StratB", 1.6)

    # Baseline (must exist to prevent np.median from returning NaN on empty lists)
    p_base = _make_mock_portfolio("Baseline", 0.0)

    pop = MagicMock(spec=Population)
    pop.__iter__.return_value = [p_base, pa1, pa2, pb1, pb2]
    pop.boxplot_measure.return_value = fig_mock
    pop.plot_distribution.return_value = fig_mock

    reporter = Reporter(pop)
    reporter.generate_tearsheet()

    # Verify the best_portfolios Population was constructed with only StratB items
    # Since we can't patch Population easily due to isinstance checks, we inspect
    # the instance passed to plot_composition.
    called_instance = mock_composition.call_args[0][0]
    passed_portfolios = list(called_instance)
    assert all(p.tag == "StratB" for p in passed_portfolios)
    assert len(passed_portfolios) == 2


@patch.object(Population, "plot_composition")
@patch.object(Population, "plot_distribution")
@patch.object(Population, "boxplot_measure")
def test_custom_baseline_tag(
    mock_boxplot: MagicMock,
    mock_distribution: MagicMock,
    mock_composition: MagicMock,
) -> None:
    """Test custom baseline_tag argument is honoured."""
    fig_mock = MagicMock()
    mock_boxplot.return_value = fig_mock
    mock_distribution.return_value = fig_mock
    mock_composition.return_value = fig_mock

    p1 = _make_mock_portfolio("MyBaseline", 1.0)

    pop = MagicMock(spec=Population)
    pop.__iter__.return_value = [p1]
    reporter = Reporter(pop)
    reporter.generate_tearsheet(baseline_tag="MyBaseline")

    boxplot_kwargs = pop.boxplot_measure.call_args[1]
    # It shouldn't add "Baseline" if "MyBaseline" is the designated baseline
    assert "Baseline" not in boxplot_kwargs["tag_list"]
    assert "MyBaseline" in boxplot_kwargs["tag_list"]


# ---------------------------------------------------------------------------
# Headless API Methods
# ---------------------------------------------------------------------------


@patch.object(Population, "plot_composition")
@patch.object(Population, "plot_distribution")
@patch.object(Population, "boxplot_measure")
def test_get_plotly_figures(
    mock_boxplot: MagicMock,
    mock_distribution: MagicMock,
    mock_composition: MagicMock,
) -> None:
    """Test get_plotly_figures returns dict of figures and does not call show()."""
    fig1 = MagicMock()
    fig2 = MagicMock()
    fig3 = MagicMock()
    mock_boxplot.return_value = fig1
    mock_distribution.return_value = fig2
    mock_composition.return_value = fig3

    p1 = _make_mock_portfolio("Baseline", 1.0)
    pop = MagicMock(spec=Population)
    pop.__iter__.return_value = [p1]
    pop.boxplot_measure.return_value = fig1
    pop.plot_distribution.return_value = fig2

    reporter = Reporter(pop)
    figs = reporter.get_plotly_figures()

    assert isinstance(figs, dict)
    assert set(figs.keys()) == {
        "cvar_boxplot",
        "drawdown_distribution",
        "best_composition",
    }
    assert figs["cvar_boxplot"] is fig1
    assert figs["drawdown_distribution"] is fig2
    assert figs["best_composition"] is fig3

    fig1.show.assert_not_called()
    fig2.show.assert_not_called()
    fig3.show.assert_not_called()


def test_get_plotly_figures_empty() -> None:
    """Test get_plotly_figures raises ValueError on empty population."""
    pop = MagicMock(spec=Population)
    pop.__iter__.return_value = []
    reporter = Reporter(pop)
    with pytest.raises(ValueError, match="Population contains no tagged portfolios."):
        reporter.get_plotly_figures()


def test_to_markdown_artifact() -> None:
    """Test to_markdown_artifact generates expected markdown format."""
    p1 = _make_mock_portfolio("Baseline", 1.0)
    p1.name = "TestStrat"
    p1.sharpe_ratio = 1.2345
    p1.cvar = 0.05
    p1.max_drawdown = 0.10
    p1.sortino_ratio = 1.5
    p1.weights = [0.4, 0.3, 0.2, 0.05, 0.05, 0.0]
    p1.assets = ["A", "B", "C", "D", "E", "F"]

    pop = MagicMock(spec=Population)
    pop.__iter__.return_value = [p1]
    pop.__len__.return_value = 1

    reporter = Reporter(pop)
    report = reporter.to_markdown_artifact()

    assert "Portfolio Population Report" in report
    assert "Generated:" in report
    assert "TestStrat" in report
    assert "1.2345" in report
    assert "**Best Tag (by median CVaR Ratio):** Baseline" in report
    assert "Top 5 Holdings:" in report
    assert "A (40.00%)" in report


def test_to_markdown_artifact_empty() -> None:
    """Test to_markdown_artifact raises ValueError on empty population."""
    pop = MagicMock(spec=Population)
    pop.__iter__.return_value = []
    pop.__len__.return_value = 0
    reporter = Reporter(pop)
    with pytest.raises(ValueError, match="Population is empty."):
        reporter.to_markdown_artifact()


def test_extract_metrics_dataframe() -> None:
    """Test extract_metrics_dataframe returns a pandas DataFrame with exact columns."""
    p1 = _make_mock_portfolio("Baseline", 1.0)
    p1.name = "StratA"
    p1.sharpe_ratio = 1.1
    p1.sortino_ratio = 1.2
    p1.cvar = 0.1
    p1.max_drawdown = 0.2
    p1.mean = 0.05

    pop = MagicMock(spec=Population)
    pop.__iter__.return_value = [p1]
    pop.__len__.return_value = 1

    reporter = Reporter(pop)
    df = reporter.extract_metrics_dataframe()

    import pandas as pd

    assert isinstance(df, pd.DataFrame)
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
    assert len(df) == 1
    assert df.iloc[0]["name"] == "StratA"
    assert df.iloc[0]["sharpe"] == 1.1


def test_extract_metrics_dataframe_empty() -> None:
    """Test extract_metrics_dataframe raises ValueError on empty population."""
    pop = MagicMock(spec=Population)
    pop.__iter__.return_value = []
    pop.__len__.return_value = 0
    reporter = Reporter(pop)
    with pytest.raises(ValueError, match="Population is empty. No metrics to extract."):
        reporter.extract_metrics_dataframe()


# ---------------------------------------------------------------------------
# Real Fixture Tests & Edge Cases
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_population() -> Population:
    """Fixture providing a real Population with real Portfolio instances."""
    df1 = pd.DataFrame(np.random.randn(10, 3) / 100, columns=["A", "B", "C"])
    df1.index = pd.date_range("2023-01-01", periods=10)
    p1 = Portfolio(X=df1, weights=np.array([0.3, 0.3, 0.4]), name="port1", tag="Baseline")
    
    df2 = pd.DataFrame(np.random.randn(10, 3) / 100, columns=["A", "B", "C"])
    df2.index = pd.date_range("2023-01-01", periods=10)
    p2 = Portfolio(X=df2, weights=np.array([0.5, 0.5, 0.0]), name="port2", tag="StratA")

    return Population([p1, p2])


def test_generate_tearsheet_delegates_to_get_plotly_figures(dummy_population: Population) -> None:
    """Test that generate_tearsheet delegates correctly when using a real population."""
    reporter = Reporter(dummy_population)
    
    with patch.object(reporter, "get_plotly_figures") as mock_get_figures:
        fig_mock = MagicMock()
        mock_get_figures.return_value = {"mock_fig": fig_mock}
        
        reporter.generate_tearsheet(baseline_tag="Baseline")
        
        mock_get_figures.assert_called_once_with(baseline_tag="Baseline")
        fig_mock.show.assert_called_once()


def test_to_markdown_artifact_real_population(dummy_population: Population) -> None:
    """Test to_markdown_artifact formatting with real objects."""
    reporter = Reporter(dummy_population)
    report = reporter.to_markdown_artifact(baseline_tag="Baseline")
    
    assert "Portfolio Population Report" in report
    assert "port1" in report
    assert "port2" in report
    # Either port1 or port2 will be best strategy depending on random returns
    assert "Top 5 Holdings" in report


def test_to_markdown_artifact_without_tabulate(dummy_population: Population) -> None:
    """Test the native Markdown table fallback when tabulate is missing."""
    reporter = Reporter(dummy_population)
    
    # Hide tabulate from sys.modules to simulate it being uninstalled
    with patch.dict(sys.modules, {"tabulate": None}):
        report = reporter.to_markdown_artifact()
        
    assert "| Strategy" in report
    assert "| Tag" in report
    assert "| Sharpe" in report
    assert "--- | ---" in report


def test_to_markdown_artifact_unlabelled_assets() -> None:
    """Test formatting when portfolios lack string column names."""
    # Dataframe without column names (default integer indexing)
    df = pd.DataFrame(np.random.randn(10, 3) / 100)
    df.index = pd.date_range("2023-01-01", periods=10)
    
    # Portfolio created without explicit names
    p = Portfolio(X=df, weights=np.array([0.3, 0.3, 0.4]), name="port_unlabelled", tag="StratB")
    
    pop = Population([p])
    reporter = Reporter(pop)
    report = reporter.to_markdown_artifact()
    
    assert "Asset_0" in report or "0" in report
    assert "port_unlabelled" in report
