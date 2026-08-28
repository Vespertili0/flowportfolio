"""Prior synthesiser module for flowportfolio."""

from __future__ import annotations

import re
from skfolio.prior import EmpiricalPrior, EntropyPooling, SyntheticData
from skfolio.moments import DenoiseCovariance, ShrunkMu
from skfolio.distribution import VineCopula

from flowportfolio.core.universe import Universe


class PriorSynthesiser:
    """Builds skfolio prior objects for injection into portfolio optimizers.

    This class acts as the forward-looking prior layer in the flowportfolio
    pipeline. It translates raw Universe returns and user-defined market views
    into fully configured skfolio prior objects. It does not run optimizations
    itself.

    Prefect Compatibility: All build_*() methods are pure functions of
    internal state and are safe to wrap as Prefect @task decorators.

    Parameters
    ----------
    universe : Universe
        A fully initialised Universe object from flowportfolio.core.universe.
        Must have had fetch_data() and anchor_history() called prior to
        being passed here.

    Raises
    ------
    TypeError
        If universe is not an instance of flowportfolio.core.universe.Universe.
    """

    def __init__(self, universe: Universe) -> None:
        if not isinstance(universe, Universe):
            raise TypeError(
                "universe must be an instance of flowportfolio.core.universe.Universe"
            )
        self._universe = universe
        self._views: list[dict] = []

    def add_market_view(self, view_str: str, confidence: float) -> PriorSynthesiser:
        """Register a forward-looking market view for use in Entropy Pooling.

        Parameters
        ----------
        view_str : str
            A view expression referencing asset tickers present in the universe.
            Example: "SPY > 0.05" (SPY expected to return >5% annualised).
        confidence : float
            Confidence in this view, between 0.0 (no confidence) and 1.0
            (certainty). Maps to the tau parameter in Entropy Pooling.

        Returns
        -------
        PriorSynthesiser
            Returns self to enable fluent method chaining.

        Raises
        ------
        ValueError
            If confidence is not in [0.0, 1.0].
        ValueError
            If view_str references a ticker not in universe.tickers.
        """
        if not (0.0 <= confidence <= 1.0):
            raise ValueError("confidence must be between 0.0 and 1.0")

        # Extract uppercase alphanumeric identifiers resembling tickers.
        extracted_tickers = re.findall(r"[A-Z][A-Z0-9_.]*", view_str)
        valid_tickers = set(self._universe.tickers)

        # Check that the view references at least one known ticker — catches
        # purely numeric expressions or views with no ticker-like tokens at all.
        if not any(t in valid_tickers for t in extracted_tickers):
            raise ValueError(
                f"View '{view_str}' does not reference any valid tickers. "
                f"Universe tickers: {sorted(valid_tickers)}"
            )

        # Reject any extracted token that looks like a ticker but is absent from
        # the universe — prevents silently ignoring mis-typed tickers.
        for ticker in extracted_tickers:
            if ticker not in valid_tickers:
                raise ValueError(
                    f"View '{view_str}' references ticker '{ticker}' which is "
                    f"not present in the universe."
                )

        self._views.append({"view": view_str, "confidence": confidence})
        return self

    def build_empirical_prior(
        self,
        covariance_estimator=None,
        mu_estimator=None,
    ) -> EmpiricalPrior:
        """Build a denoised empirical prior from historical returns.

        Note: Denoising is applied INSIDE the prior estimator, not as a
        pipeline transformer step. This is the correct skfolio architecture.

        Parameters
        ----------
        covariance_estimator : skfolio covariance estimator, optional
            Defaults to DenoiseCovariance().
        mu_estimator : skfolio mu estimator, optional
            Defaults to ShrunkMu().

        Returns
        -------
        skfolio.prior.EmpiricalPrior
        """
        if covariance_estimator is None:
            covariance_estimator = DenoiseCovariance()
        if mu_estimator is None:
            mu_estimator = ShrunkMu()

        return EmpiricalPrior(
            covariance_estimator=covariance_estimator, mu_estimator=mu_estimator
        )

    def build_entropy_prior(self) -> EntropyPooling:
        """Build an Entropy Pooling prior from registered market views.

        Returns
        -------
        skfolio.prior.EntropyPooling

        Raises
        ------
        RuntimeError
            If no market views have been registered via add_market_view().
        """
        if not self._views:
            raise RuntimeError(
                "No market views registered. Call add_market_view() before building an entropy prior."
            )

        views = [v["view"] for v in self._views]
        tau = [v["confidence"] for v in self._views]

        return EntropyPooling(views=views, tau=tau)

    def build_synthetic_prior(self, n_samples: int = 5000) -> SyntheticData:
        """Generate synthetic return paths using a VineCopula model.

        Fits a VineCopula to universe.returns and generates n_samples
        synthetic paths that preserve empirical tail dependencies.
        Used for stress-testing portfolio allocations.

        Parameters
        ----------
        n_samples : int, optional
            Number of synthetic return paths to generate. Default is 5000.

        Returns
        -------
        skfolio.prior.SyntheticData

        Raises
        ------
        ValueError
            If n_samples < 100.
        """
        if n_samples < 100:
            raise ValueError(f"n_samples must be at least 100, got {n_samples}")

        return SyntheticData(distribution_estimator=VineCopula(), n_samples=n_samples)
