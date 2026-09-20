"""Degradation reporting module.

This module provides the :class:`Reporter` class, which standardises visual
dominance reporting for candidate strategies using ``skfolio``'s built-in
plotting capabilities.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from skfolio import Population, RatioMeasure, RiskMeasure


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
        except ValueError:
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
        stress_pop = Population([p for p in self._population if p.tag == stress_tag])
        if not stress_pop:
            raise ValueError(
                f"Tag '{stress_tag}' matched no portfolios in the population."
            )

        baseline_pop = Population(
            [p for p in self._population if p.tag == baseline_tag]
        )
        if not baseline_pop:
            raise ValueError(
                f"Tag '{baseline_tag}' matched no portfolios in the population."
            )

        combined = stress_pop + baseline_pop

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
        tag_list = list(
            dict.fromkeys(
                p.tag for p in self._population if getattr(p, "tag", None) is not None
            )
        )
        if not tag_list:
            raise ValueError("Population contains no tagged portfolios.")

        if baseline_tag not in tag_list:
            tag_list.insert(0, baseline_tag)

        fig1 = self._population.boxplot_measure(
            measure=RatioMeasure.CVAR_RATIO,
            tag_list=tag_list,
        )

        fig2 = self._population.plot_distribution(
            measure_list=[RatioMeasure.AVERAGE_DRAWDOWN_RATIO],
            tag_list=tag_list,
        )

        valid_tags = [
            t
            for t in set(tag_list)
            if any(getattr(p, "tag", None) == t for p in self._population)
        ]

        def safe_median(tag: str) -> float:
            cvar_ratios = [
                p.cvar_ratio
                for p in self._population
                if getattr(p, "tag", None) == tag
                and getattr(p, "cvar_ratio", None) is not None
                and not np.isnan(p.cvar_ratio)
            ]
            return float(np.median(cvar_ratios)) if cvar_ratios else float("-inf")

        best_tag = max(valid_tags, key=safe_median)

        best_portfolios = Population(
            [p for p in self._population if getattr(p, "tag", None) == best_tag]
        )
        fig3 = best_portfolios.plot_composition()

        return {
            "cvar_boxplot": fig1,
            "drawdown_distribution": fig2,
            "best_composition": fig3,
        }

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

        df_metrics = pd.DataFrame(metrics)
        try:
            table_md = df_metrics.to_markdown(index=False)
        except (ImportError, ModuleNotFoundError):
            cols = list(df_metrics.columns)
            header = "| " + " | ".join(cols) + " |"
            separator = "| " + " | ".join(["---"] * len(cols)) + " |"
            rows = [
                "| " + " | ".join(str(row[c]) for c in cols) + " |"
                for _, row in df_metrics.iterrows()
            ]
            table_md = "\n".join([header, separator] + rows)

        tag_list = list(
            dict.fromkeys(
                p.tag for p in self._population if getattr(p, "tag", None) is not None
            )
        )
        if not tag_list:
            best_strategy = "N/A (No tagged portfolios)"
            top_holdings = "N/A"
        else:
            valid_tags = [
                t
                for t in set(tag_list)
                if any(getattr(p, "tag", None) == t for p in self._population)
            ]

            def safe_median(tag: str) -> float:
                cvar_ratios = [
                    p.cvar_ratio
                    for p in self._population
                    if getattr(p, "tag", None) == tag
                    and getattr(p, "cvar_ratio", None) is not None
                    and not np.isnan(p.cvar_ratio)
                ]
                return float(np.median(cvar_ratios)) if cvar_ratios else float("-inf")

            best_tag = max(valid_tags, key=safe_median)

            best_portfolios = [
                p for p in self._population if getattr(p, "tag", None) == best_tag
            ]
            best_portfolio = max(
                best_portfolios,
                key=lambda p: (
                    p.cvar_ratio
                    if getattr(p, "cvar_ratio", None) is not None
                    and not np.isnan(p.cvar_ratio)
                    else float("-inf")
                ),
            )

            weights = getattr(best_portfolio, "weights", None)
            assets = getattr(best_portfolio, "assets", None)

            if weights is None or len(weights) == 0:
                top_holdings = "N/A (No weights available)"
            else:
                if assets is None or len(assets) != len(weights):
                    assets = [f"Asset_{i}" for i in range(len(weights))]

                asset_weights = list(zip(assets, weights))
                asset_weights = [
                    aw
                    for aw in asset_weights
                    if aw[1] is not None and not np.isnan(aw[1])
                ]
                asset_weights.sort(key=lambda x: abs(x[1]), reverse=True)
                top_5 = asset_weights[:5]

                if not top_5:
                    top_holdings = "N/A (All weights zero or NaN)"
                else:
                    top_holdings = ", ".join(
                        f"{asset!s} ({w:.2%})" for asset, w in top_5
                    )

            best_strategy = best_tag

        report = (
            f"# Portfolio Population Report\n\n"
            f"**Generated:** {timestamp}\n\n"
            f"## Performance Metrics\n\n"
            f"{table_md}\n\n"
            f"## Best Strategy Analysis\n\n"
            f"**Best Tag (by median CVaR Ratio):** {best_strategy}\n"
            f"**Top 5 Holdings:** {top_holdings}\n"
        )
        return report

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
