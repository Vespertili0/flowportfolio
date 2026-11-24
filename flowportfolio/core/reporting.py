"""Degradation reporting module.

This module provides the :class:`Reporter` class, which standardises visual
dominance reporting for candidate strategies using ``skfolio``'s built-in
plotting capabilities.
"""

from __future__ import annotations

import collections
import copy
import math
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from skfolio import Population, RatioMeasure, RiskMeasure

from flowportfolio.core.experiment import ExperimentResult


def _unpack_population_and_live_weights(
    population: Population | ExperimentResult,
    live_weights: dict[str, dict[str, float]] | None,
) -> tuple[Population, dict[str, dict[str, float]]]:
    """Resolve a ``Population`` and merged live-weights mapping from inputs."""
    result_lw: dict[str, dict[str, float]] | None = None
    if isinstance(population, ExperimentResult):
        pop = population.population
        result_lw = population.live_weights
    else:
        pop = population

    if not isinstance(pop, Population):
        raise TypeError("population must be a skfolio.Population instance.")
    if live_weights is not None and not isinstance(live_weights, dict):
        raise TypeError(
            "live_weights must be a dict mapping strategy tags to weight dicts or None."
        )

    merged: dict[str, dict[str, float]] = {}
    pop_latest = getattr(pop, "latest_weights_by_tag", None)
    if isinstance(pop_latest, dict):
        merged.update(pop_latest)
    if isinstance(result_lw, dict):
        merged.update(result_lw)
    if live_weights is not None:
        merged.update(live_weights)
    return pop, merged


def _get_portfolio_metric(p: object, annualized_attr: str, raw_attr: str) -> object:
    """Return annualized metric when available, falling back to raw_attr without warnings."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ann_val = getattr(p, annualized_attr, None)
        if isinstance(ann_val, (int, float, np.number)) and not isinstance(
            ann_val, bool
        ):
            return float(ann_val)
        return getattr(p, raw_attr, None)


def _safe_getattr(p: object, attr: str) -> Any:
    """Read a portfolio attribute while suppressing divide-by-zero numpy warnings."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return getattr(p, attr, None)


def _finite_cvar_ratio(p: object) -> float | None:
    """Extract a portfolio's CVaR ratio only when it is a finite real number."""
    val = _safe_getattr(p, "cvar_ratio")
    if val is None or isinstance(val, bool):
        return None
    try:
        f_val = float(val)
    except (ValueError, TypeError):
        return None
    return f_val if math.isfinite(f_val) else None


class Reporter:
    """Visual tearsheet generator for evaluated strategies.

    Consumes a :class:`Population` or :class:`ExperimentResult` of portfolios
    (typically output by
    :meth:`flowportfolio.core.experiment.PortfolioExperimentEngine.run_robustness_test`)
    and generates a standardised three-panel visual tearsheet comparing them
    against a baseline.

    Parameters
    ----------
    population : Population or ExperimentResult
        The population of portfolios (or immutable :class:`ExperimentResult`) to
        analyse.
    live_weights : dict[str, dict[str, float]] or None, optional
        Optional explicit mapping of strategy tag to live terminal weights.
        Takes precedence over ``ExperimentResult.live_weights`` and historical
        Fold-N OOS weights.

    Raises
    ------
    TypeError
        If ``population`` is not a ``skfolio.Population`` or
        ``ExperimentResult``, or if ``live_weights`` is not a dict or None.
    """

    def __init__(
        self,
        population: Population | ExperimentResult,
        *,
        live_weights: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self._population, self._live_weights = _unpack_population_and_live_weights(
            population, live_weights
        )

    @staticmethod
    def _unroll_portfolios(portfolios: Any) -> Population:
        """Unroll MultiPeriodPortfolios into constituent single-period Portfolios with inherited tags."""
        unrolled = []
        for p in portfolios:
            parent_tag = getattr(p, "tag", None)
            children = getattr(p, "portfolios", None)
            if isinstance(children, list) and len(children) > 0:
                for child in children:
                    child_copy = copy.copy(child)
                    if (
                        getattr(child_copy, "tag", None) is None
                        and parent_tag is not None
                    ):
                        try:
                            child_copy.tag = parent_tag
                        except AttributeError:
                            pass
                    unrolled.append(child_copy)
            else:
                unrolled.append(p)
        return Population(unrolled)

    def generate_tearsheet(
        self,
        baseline_tag: str = "Baseline",
        *,
        unroll_folds: bool = True,
    ) -> None:
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
        unroll_folds : bool, default True
            When ``True``, unrolls ``MultiPeriodPortfolio`` folds into
            per-fold observations so boxplots and distributions reflect the
            full out-of-sample fold distribution.

        Raises
        ------
        ValueError
            If the population contains no tagged portfolios or no finite CVaR
            ratios.
        """
        figures = self.get_plotly_figures(
            baseline_tag=baseline_tag, unroll_folds=unroll_folds
        )
        for fig in figures.values():
            fig.show()

    def plot_stress_impact(
        self,
        stress_tag: str,
        baseline_tag: str,
        *,
        unroll_folds: bool = False,
    ) -> None:
        """Generate comparative risk plots for stress vs baseline portfolios.

        Filters the internal Population by tag and produces two plots:
        1. Boxplot of CVaR comparing stress scenarios against the baseline.
        2. Distribution plot of Max Drawdown comparing stress vs baseline.

        Parameters
        ----------
        stress_tag : str
            Tag value identifying stress-scenario portfolios in the Population.
        baseline_tag : str
            Tag value identifying baseline portfolios in the Population.
        unroll_folds : bool, default False
            When ``False`` (default), evaluates full uncensored multi-period
            trajectories. When ``True``, unrolls constituent fold portfolios.

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

        combined_items = [*stress_raw, *baseline_raw]
        combined = (
            self._unroll_portfolios(combined_items)
            if unroll_folds
            else Population(combined_items)
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            fig1 = combined.boxplot_measure(
                measure=RiskMeasure.CVAR,
                tag_list=[stress_tag, baseline_tag],
            )
            fig1.update_layout(
                title=f"CVaR Distribution: {stress_tag} vs {baseline_tag}"
            )
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
        self,
        baseline_tag: str = "Baseline",
        *,
        unroll_folds: bool = True,
    ) -> dict[str, go.Figure]:
        """Generate Plotly figures for strategy comparison without displaying them.

        Parameters
        ----------
        baseline_tag : str, default "Baseline"
            The tag representing the baseline strategy, ensuring it is always
            included in the visual comparisons even if its metric distribution
            is narrow.
        unroll_folds : bool, default True
            When ``True`` (default), unrolls ``MultiPeriodPortfolio`` folds
            into single-period fold portfolios so boxplots and distributions
            display the complete fold distribution. Set ``False`` to plot
            aggregate multi-period trajectory metrics.

        Returns
        -------
        dict[str, plotly.graph_objects.Figure]
            Exact keys "cvar_boxplot", "drawdown_distribution", and
            "best_composition".

        Raises
        ------
        ValueError
            If the population contains no tagged portfolios or no finite CVaR
            ratios.
        """
        tag_list = self._unique_tags()
        if not tag_list:
            raise ValueError("Population contains no tagged portfolios.")

        valid_population_tags = {
            p.tag for p in self._population if getattr(p, "tag", None) is not None
        }
        if baseline_tag not in tag_list and baseline_tag in valid_population_tags:
            tag_list.insert(0, baseline_tag)

        plot_population = (
            self._unroll_portfolios(self._population)
            if unroll_folds
            else Population(list(self._population))
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            fig1 = plot_population.boxplot_measure(
                measure=RatioMeasure.CVAR_RATIO,
                tag_list=tag_list,
            )

            fig2 = plot_population.plot_distribution(
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

    @staticmethod
    def _format_metric(val: object) -> str:
        """Format a numeric or nullable portfolio metric for Markdown tables."""
        if val is None or isinstance(val, bool):
            return "N/A"
        try:
            f_val = float(val)  # type: ignore[arg-type]
        except (ValueError, TypeError):
            return str(val)
        if math.isnan(f_val):
            return "nan"
        if not math.isfinite(f_val):
            return "N/A"
        return f"{f_val:.4f}"

    @staticmethod
    def _extract_asset_weights(
        target_port: object,
        live_tag_weights: dict[str, float] | None,
    ) -> list[tuple[str, float]] | None:
        """Extract finite ``(asset, weight)`` pairs from live weights or a portfolio."""

        def _is_valid_weight(w: object) -> bool:
            if w is None or isinstance(w, bool):
                return False
            try:
                return math.isfinite(float(w))  # type: ignore[arg-type]
            except (ValueError, TypeError):
                return False

        weights_dict: object = (
            live_tag_weights
            if isinstance(live_tag_weights, dict) and len(live_tag_weights) > 0
            else getattr(target_port, "weights_dict", None)
        )
        if isinstance(weights_dict, dict) and len(weights_dict) > 0:
            return [
                (str(asset), float(w))
                for asset, w in weights_dict.items()
                if _is_valid_weight(w)
            ]

        weights = getattr(target_port, "weights", None)
        if weights is None or len(weights) == 0:
            return None

        assets_candidate = getattr(target_port, "assets_names", None)
        if not isinstance(assets_candidate, (list, tuple, np.ndarray, pd.Index)):
            assets_candidate = getattr(target_port, "assets", None)
        assets = (
            assets_candidate
            if assets_candidate is not None and len(assets_candidate) == len(weights)
            else [f"Asset_{i}" for i in range(len(weights))]
        )
        return [
            (str(asset), float(w))
            for asset, w in zip(assets, weights, strict=True)
            if _is_valid_weight(w)
        ]

    def _generate_metrics_table(self) -> str:
        """Generate a Markdown table of performance metrics for the population."""
        metrics = []
        for p in self._population:
            tag = getattr(p, "tag", None)
            metrics.append(
                {
                    "Strategy": getattr(p, "name", "N/A") or "N/A",
                    "Tag": tag if tag else "N/A",
                    "Sharpe": self._format_metric(
                        _get_portfolio_metric(
                            p, "annualized_sharpe_ratio", "sharpe_ratio"
                        )
                    ),
                    "CVaR": self._format_metric(_safe_getattr(p, "cvar")),
                    "Max Drawdown": self._format_metric(
                        _safe_getattr(p, "max_drawdown")
                    ),
                    "Sortino": self._format_metric(
                        _get_portfolio_metric(
                            p, "annualized_sortino_ratio", "sortino_ratio"
                        )
                    ),
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
        """Identify the best strategy by median finite CVaR Ratio and format its top 5 holdings."""
        tag_list = self._unique_tags()
        if not tag_list:
            return "N/A (No tagged portfolios)", "N/A"

        best_tag = self._best_tag_by_median_cvar(tag_list)

        best_portfolios = [
            p for p in self._population if getattr(p, "tag", None) == best_tag
        ]

        def _score_portfolio(p: object) -> float:
            ratio = _finite_cvar_ratio(p)
            return ratio if ratio is not None else float("-inf")

        best_portfolio = max(best_portfolios, key=_score_portfolio)

        # Resolve the terminal single-period portfolio of a MultiPeriodPortfolio
        target_port = best_portfolio
        if hasattr(best_portfolio, "portfolios"):
            portfolios = getattr(best_portfolio, "portfolios", None)
            if isinstance(portfolios, list) and len(portfolios) > 0:
                target_port = portfolios[-1]

        asset_weights = self._extract_asset_weights(
            target_port, self._live_weights.get(best_tag)
        )
        if asset_weights is None:
            return best_tag, "N/A (No weights available)"
        if not asset_weights:
            return best_tag, "N/A (All weights zero or NaN)"

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
            identifying the best strategy by median finite CVaR Ratio and its
            top 5 holdings.

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
                ratio
                for p in baseline_portfolios
                if (ratio := _finite_cvar_ratio(p)) is not None
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
        def _collect_cvar_by_tag(iterable: Any) -> dict[str, list[float]]:
            mapping: dict[str, list[float]] = collections.defaultdict(list)
            for p in iterable:
                tag = getattr(p, "tag", None)
                if tag is not None and tag in tag_list:
                    ratio = _finite_cvar_ratio(p)
                    if ratio is not None:
                        mapping[tag].append(ratio)
            return mapping

        tag_to_cvar = _collect_cvar_by_tag(self._population)
        if not tag_to_cvar:
            tag_to_cvar = _collect_cvar_by_tag(
                self._unroll_portfolios(self._population)
            )

        valid_tags = list(tag_to_cvar.keys())
        if not valid_tags:
            raise ValueError(
                "No portfolios with finite CVaR ratio and matching tags found in population."
            )

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
                    "sharpe": _get_portfolio_metric(
                        p, "annualized_sharpe_ratio", "sharpe_ratio"
                    ),
                    "sortino": _get_portfolio_metric(
                        p, "annualized_sortino_ratio", "sortino_ratio"
                    ),
                    "cvar": _safe_getattr(p, "cvar"),
                    "max_drawdown": _safe_getattr(p, "max_drawdown"),
                    "cvar_ratio": _safe_getattr(p, "cvar_ratio"),
                    "mean_return": _get_portfolio_metric(p, "annualized_mean", "mean"),
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
