"""Degradation reporting module.

This module provides the :class:`Reporter` class, which standardises visual
dominance reporting for candidate strategies using ``skfolio``'s built-in
plotting capabilities.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from skfolio import Population, RatioMeasure, RiskMeasure

logger = logging.getLogger(__name__)


class Reporter:
    """Visual tearsheet generator for evaluated strategies.

    Consumes a :class:`Population` of portfolios (typically output by
    :meth:`flowportfolio.core.experiment.PortfolioExperimentEngine.run_robustness_test`)
    and generates a standardised three-panel visual tearsheet comparing them
    against a baseline.

    Parameters
    ----------
    population : Population
        The population of portfolios to analyse.

    Raises
    ------
    TypeError
        If the provided ``population`` is not a ``skfolio.Population`` instance.
    """

    def __init__(self, population: Population) -> None:
        if not isinstance(population, Population):
            raise TypeError("population must be a skfolio.Population instance.")
        self._population = population

    @staticmethod
    def _unroll_portfolios(portfolios) -> Population:
        """Unroll MultiPeriodPortfolios into constituent single-period Portfolios with inherited tags."""
        unrolled = []
        for p in portfolios:
            parent_tag = getattr(p, "tag", None)
            children = getattr(p, "portfolios", None)
            if isinstance(children, list) and len(children) > 0:
                for child in children:
                    if getattr(child, "tag", None) is None and parent_tag is not None:
                        try:
                            child.tag = parent_tag
                        except AttributeError:
                            pass
                    unrolled.append(child)
            else:
                unrolled.append(p)
        return Population(unrolled)

    def generate_tearsheet(self, baseline_tag: str = "Baseline") -> None:
        """Generate and display the standardised tearsheet plots.

        Executes and displays three comparative plots:
        1. A boxplot of the CVaR Ratio across all strategies.
        2. A distribution plot of the Average Drawdown Ratio.
        3. A composition plot for the strategy with the highest median CVaR Ratio.

        Parameters
        ----------
        baseline_tag : str, default "Baseline"
            The tag representing the baseline strategy, ensuring it is always
            included in the visual comparisons even if its metric distribution
            is narrow.
        """
        try:
            figures = self.get_plotly_figures(baseline_tag=baseline_tag)
        except ValueError as e:
            logger.warning("Failed to generate plot in Tearsheet: %s", e)
            return

        for fig in figures.values():
            fig.show()

    def plot_stress_impact(
        self,
        stress_tag: str,
        baseline_tag: str,
    ) -> None:
        """Generate comparative risk plots for stress vs baseline portfolios.

        Filters the internal Population by tag and produces two plots:
        1. Boxplot of CVaR comparing stress scenarios against the baseline.
        2. Distribution plot of Max Drawdown comparing stress vs baseline.

        Tags are set by the PortfolioExperimentEngine when assembling the
        Population. The stress_tag typically corresponds to portfolios
        generated using a SyntheticData (VineCopula) prior.

        Parameters
        ----------
        stress_tag : str
            Tag value identifying stress-scenario portfolios in the Population.
        baseline_tag : str
            Tag value identifying baseline portfolios in the Population.

        Returns
        -------
        None
            Renders plots directly via .show() for interactive environments.

        Raises
        ------
        ValueError
            If stress_tag or baseline_tag matches zero portfolios in the
            Population, with message:
            "Tag '{tag}' matched no portfolios in the population."
        """
        stress_raw = [
            p for p in self._population if getattr(p, "tag", None) == stress_tag
        ]
        if not stress_raw:
            raise ValueError(
                f"Tag '{stress_tag}' matched no portfolios in the population."
            )

        baseline_raw = [
            p for p in self._population if getattr(p, "tag", None) == baseline_tag
        ]
        if not baseline_raw:
            raise ValueError(
                f"Tag '{baseline_tag}' matched no portfolios in the population."
            )

        combined = self._unroll_portfolios([*stress_raw, *baseline_raw])

        fig1 = combined.boxplot_measure(
            measure=RiskMeasure.CVAR,
            tag_list=[stress_tag, baseline_tag],
        )
        fig1.update_layout(title=f"CVaR Distribution: {stress_tag} vs {baseline_tag}")
        fig1.show()

        fig2 = combined.plot_distribution(
            measure_list=[RiskMeasure.MAX_DRAWDOWN],
            tag_list=[stress_tag, baseline_tag],
        )
        fig2.update_layout(
            title=f"Max Drawdown Distribution: {stress_tag} vs {baseline_tag}"
        )
        fig2.show()

    def get_plotly_figures(
        self, baseline_tag: str = "Baseline"
    ) -> dict[str, go.Figure]:
        """Generate Plotly figures for strategy comparison without displaying them.

        Parameters
        ----------
        baseline_tag : str, default "Baseline"
            The tag representing the baseline strategy, ensuring it is always
            included in the visual comparisons even if its metric distribution
            is narrow.

        Returns
        -------
        dict[str, plotly.graph_objects.Figure]
            Exact keys "cvar_boxplot", "drawdown_distribution", and
            "best_composition".

        Raises
        ------
        ValueError
            If the population contains no tagged portfolios.
        """
        tag_list = self._unique_tags()
        if not tag_list:
            raise ValueError("Population contains no tagged portfolios.")

        valid_population_tags = {
            p.tag for p in self._population if getattr(p, "tag", None) is not None
        }
        if baseline_tag not in tag_list and baseline_tag in valid_population_tags:
            tag_list.insert(0, baseline_tag)

        unrolled_population = self._unroll_portfolios(self._population)

        fig1 = unrolled_population.boxplot_measure(
            measure=RatioMeasure.CVAR_RATIO,
            tag_list=tag_list,
        )

        fig2 = unrolled_population.plot_distribution(
            measure_list=[RatioMeasure.AVERAGE_DRAWDOWN_RATIO],
            tag_list=tag_list,
        )

        best_tag = self._best_tag_by_median_cvar(tag_list)

        best_portfolios = self._unroll_portfolios(
            [p for p in self._population if getattr(p, "tag", None) == best_tag]
        )
        fig3 = best_portfolios.plot_composition()

        return {
            "cvar_boxplot": fig1,
            "drawdown_distribution": fig2,
            "best_composition": fig3,
        }

    def _generate_metrics_table(self) -> str:
        """Generate a Markdown table of performance metrics for the population."""

        def _fmt(val: object) -> str:
            if val is None:
                return "N/A"

            if isinstance(val, (int, float, np.number)):
                if np.isnan(val):
                    return "nan"
                return f"{float(val):.4f}"
            try:
                f_val = float(val)  # type: ignore[arg-type]
                if np.isnan(f_val):
                    return "nan"
                return f"{f_val:.4f}"
            except (ValueError, TypeError):
                return str(val)

        metrics = []
        for p in self._population:
            metrics.append(
                {
                    "Strategy": getattr(p, "name", "N/A") or "N/A",
                    "Tag": getattr(p, "tag", None)
                    if getattr(p, "tag", None)
                    else "N/A",
                    "Sharpe": _fmt(getattr(p, "sharpe_ratio", None)),
                    "CVaR": _fmt(getattr(p, "cvar", None)),
                    "Max Drawdown": _fmt(getattr(p, "max_drawdown", None)),
                    "Sortino": _fmt(getattr(p, "sortino_ratio", None)),
                }
            )

        df_metrics = pd.DataFrame(
            metrics,
            columns=["Strategy", "Tag", "Sharpe", "CVaR", "Max Drawdown", "Sortino"],
        )
        try:
            table_md = df_metrics.to_markdown(index=False)
        except (ImportError, ModuleNotFoundError):
            cols = list(df_metrics.columns)
            header = "| " + " | ".join(cols) + " |"
            separator = "| " + " | ".join(["---"] * len(cols)) + " |"
            rows = [
                "| " + " | ".join(str(val) for val in row) + " |"
                for row in df_metrics.itertuples(index=False)
            ]
            table_md = "\n".join([header, separator] + rows)

        return table_md

    def _get_best_strategy_and_holdings(self) -> tuple[str, str]:
        """Identify the best strategy by median CVaR Ratio and format its top 5 holdings."""
        tag_list = self._unique_tags()
        if not tag_list:
            return "N/A (No tagged portfolios)", "N/A"

        best_tag = self._best_tag_by_median_cvar(tag_list)

        best_portfolios = [
            p for p in self._population if getattr(p, "tag", None) == best_tag
        ]

        def _get_cvar_ratio(p: object) -> float:
            val = getattr(p, "cvar_ratio", None)
            if val is not None and not np.isnan(val):
                return float(val)
            return float("-inf")

        best_portfolio = max(best_portfolios, key=_get_cvar_ratio)

        # Resolve the terminal single-period portfolio of a MultiPeriodPortfolio
        target_port = best_portfolio
        if hasattr(best_portfolio, "portfolios"):
            portfolios = getattr(best_portfolio, "portfolios", None)
            if isinstance(portfolios, list) and len(portfolios) > 0:
                target_port = portfolios[-1]

        def _is_valid_weight(w: object) -> bool:
            if w is None:
                return False
            try:
                return not np.isnan(float(w))  # type: ignore[arg-type]
            except (ValueError, TypeError):
                return False

        weights_dict = getattr(target_port, "weights_dict", None)
        if isinstance(weights_dict, dict) and len(weights_dict) > 0:
            asset_weights = [
                (str(asset), float(w))
                for asset, w in weights_dict.items()
                if _is_valid_weight(w)
            ]
        else:
            weights = getattr(target_port, "weights", None)
            assets = getattr(
                target_port, "assets_names", getattr(target_port, "assets", None)
            )
            if weights is None or len(weights) == 0:
                return best_tag, "N/A (No weights available)"
            if assets is None or len(assets) != len(weights):
                assets = [f"Asset_{i}" for i in range(len(weights))]
            asset_weights = [
                (str(aw[0]), float(aw[1]))
                for aw in zip(assets, weights)
                if _is_valid_weight(aw[1])
            ]

        if not asset_weights:
            top_holdings = "N/A (All weights zero or NaN)"
        else:
            asset_weights.sort(key=lambda x: abs(x[1]), reverse=True)
            top_5 = asset_weights[:5]
            top_holdings = ", ".join(f"{asset!s} ({w:.2%})" for asset, w in top_5)

        return best_tag, top_holdings

    def to_markdown_artifact(self, baseline_tag: str = "Baseline") -> str:
        """Extract key population statistics into a formatted Markdown report.

        Parameters
        ----------
        baseline_tag : str, default "Baseline"
            The tag representing the baseline strategy, provided for interface
            consistency with other reporting methods.

        Returns
        -------
        str
            A formatted Markdown string containing an ISO8601 UTC timestamp,
            a metrics table (Sharpe, CVaR, Max Drawdown, Sortino), and analysis
            identifying the best strategy by median CVaR Ratio and its top 5
            holdings.

        Raises
        ------
        ValueError
            If the population is empty.
        """
        if not self._population or len(self._population) == 0:
            raise ValueError("Population is empty.")

        timestamp = datetime.now(UTC).isoformat()
        table_md = self._generate_metrics_table()
        best_strategy, top_holdings = self._get_best_strategy_and_holdings()

        baseline_section = ""
        baseline_portfolios = [
            p for p in self._population if getattr(p, "tag", None) == baseline_tag
        ]
        if baseline_portfolios:
            baseline_cvar_ratios: list[float] = [
                float(val)
                for p in baseline_portfolios
                if (val := getattr(p, "cvar_ratio", None)) is not None
                and not np.isnan(float(val))
            ]
            if baseline_cvar_ratios:
                baseline_median_cvar = float(np.median(baseline_cvar_ratios))
                baseline_section = (
                    f"\n\n## Baseline Strategy Analysis\n\n"
                    f"**Baseline Tag:** {baseline_tag}\n"
                    f"**Baseline Median CVaR Ratio:** {baseline_median_cvar:.4f}"
                )
            else:
                baseline_section = (
                    f"\n\n## Baseline Strategy Analysis\n\n"
                    f"**Baseline Tag:** {baseline_tag}\n"
                    f"**Baseline Median CVaR Ratio:** N/A"
                )

        report = (
            f"# Portfolio Population Report\n\n"
            f"**Generated:** {timestamp}\n\n"
            f"## Performance Metrics\n\n"
            f"{table_md}\n\n"
            f"## Best Strategy Analysis\n\n"
            f"**Best Tag (by median CVaR Ratio):** {best_strategy}\n"
            f"**Top 5 Holdings:** {top_holdings}"
            f"{baseline_section}\n"
        )
        return report

    def _unique_tags(self) -> list[str]:
        return list(
            dict.fromkeys(
                p.tag for p in self._population if getattr(p, "tag", None) is not None
            )
        )

    def _best_tag_by_median_cvar(self, tag_list: list[str]) -> str:
        import collections

        unrolled_pop = self._unroll_portfolios(self._population)
        tag_to_cvar = collections.defaultdict(list)
        for p in unrolled_pop:
            tag = getattr(p, "tag", None)
            if tag is not None and tag in tag_list:
                cvar_ratio = getattr(p, "cvar_ratio", None)
                if cvar_ratio is not None and not np.isnan(cvar_ratio):
                    tag_to_cvar[tag].append(float(cvar_ratio))

        valid_tags = list(tag_to_cvar.keys())
        if not valid_tags:
            raise ValueError("No portfolios with matching tags found in population.")

        def safe_median(tag: str) -> float:
            cvar_ratios = tag_to_cvar[tag]
            return float(np.median(cvar_ratios)) if cvar_ratios else float("-inf")

        return max(valid_tags, key=safe_median)

    def extract_metrics_dataframe(self) -> pd.DataFrame:
        """Extract per-portfolio performance metrics into a flat pandas DataFrame.

        Returns
        -------
        pandas.DataFrame
            Columns exactly "name", "tag", "sharpe", "sortino", "cvar",
            "max_drawdown", "cvar_ratio", "mean_return".

        Raises
        ------
        ValueError
            If the population is empty.
        """
        if not self._population or len(self._population) == 0:
            raise ValueError("Population is empty. No metrics to extract.")

        data = []
        for p in self._population:
            data.append(
                {
                    "name": getattr(p, "name", None),
                    "tag": getattr(p, "tag", None),
                    "sharpe": getattr(p, "sharpe_ratio", None),
                    "sortino": getattr(p, "sortino_ratio", None),
                    "cvar": getattr(p, "cvar", None),
                    "max_drawdown": getattr(p, "max_drawdown", None),
                    "cvar_ratio": getattr(p, "cvar_ratio", None),
                    "mean_return": getattr(p, "mean", None),
                }
            )

        return pd.DataFrame(
            data,
            columns=[
                "name",
                "tag",
                "sharpe",
                "sortino",
                "cvar",
                "max_drawdown",
                "cvar_ratio",
                "mean_return",
            ],
        )
