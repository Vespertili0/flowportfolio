"""Degradation reporting module.

This module provides the :class:`Reporter` class, which standardises visual
dominance reporting for candidate strategies using ``skfolio``'s built-in
plotting capabilities.
"""

from __future__ import annotations

import numpy as np
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
        # 1. Derive tag_list ensuring baseline is included
        tag_list = list(dict.fromkeys(
            p.tag for p in self._population if p.tag is not None
        ))
        
        if baseline_tag not in tag_list:
            tag_list.insert(0, baseline_tag)

        # 2. Plot 1: CVaR Ratio boxplot
        fig1 = self._population.boxplot_measure(
            measure=RatioMeasure.CVAR_RATIO,
            tag_list=tag_list,
        )
        fig1.show()

        # 3. Plot 2: Average Drawdown Ratio distribution
        fig2 = self._population.plot_distribution(
            measure_list=[RatioMeasure.AVERAGE_DRAWDOWN_RATIO],
            tag_list=tag_list,
        )
        fig2.show()

        # 4. Plot 3: Composition plot for the best strategy
        # Find the strategy (tag) with the highest median CVAR_RATIO
        best_tag = max(
            set(tag_list),
            key=lambda t: np.median([
                p.cvar_ratio for p in self._population if p.tag == t
            ])
        )
        
        best_portfolios = Population([
            p for p in self._population if p.tag == best_tag
        ])
        fig3 = best_portfolios.plot_composition()
        fig3.show()

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
            raise ValueError(f"Tag '{stress_tag}' matched no portfolios in the population.")

        baseline_pop = Population([p for p in self._population if p.tag == baseline_tag])
        if not baseline_pop:
            raise ValueError(f"Tag '{baseline_tag}' matched no portfolios in the population.")

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
        fig2.update_layout(title=f"Max Drawdown Distribution: {stress_tag} vs {baseline_tag}")
        fig2.show()
