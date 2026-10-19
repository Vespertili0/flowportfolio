"""Prior synthesiser module for flowportfolio."""

from __future__ import annotations

import re

from skfolio.distribution import VineCopula
from skfolio.moments import DenoiseCovariance, ShrunkMu
from skfolio.prior import (
    BasePrior,
    BlackLitterman,
    EmpiricalPrior,
    EntropyPooling,
    SyntheticData,
)

from flowportfolio.core.protocols import UniverseProtocol


class PriorSynthesiser:
    """Builds skfolio prior objects for injection into portfolio optimisers.

    This class acts as the forward-looking prior layer in the flowportfolio
    pipeline. It translates raw Universe returns and user-defined market views
    into fully configured skfolio prior objects. It does not run optimisations
    itself.

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

    def __init__(self, universe: UniverseProtocol) -> None:
        from flowportfolio.core.protocols import is_universe_protocol

        if not is_universe_protocol(universe):
            raise TypeError("universe must implement UniverseProtocol")
        self._universe = universe
        self._views: list[dict] = []

    @staticmethod
    def _normalise_view_operators(view_str: str) -> str:
        """Normalise strict inequality operators (> or <) to (>= or <=)."""
        normalised = re.sub(r"(?<![<>=!])>(?!=)", ">=", view_str)
        normalised = re.sub(r"(?<![<>=!])<(?!=)", "<=", normalised)
        return normalised

    def add_market_view(self, view_str: str, confidence: float) -> PriorSynthesiser:
        """Register a forward-looking market view for use in prior estimation.

        Parameters
        ----------
        view_str : str
            A view expression referencing asset tickers or metadata group names
            present in the universe. Example: "SPY >= 0.05" (SPY expected to
            return >=5% annualised). Only the right-hand-side scalar constant is
            converted to a daily return (divided by 252.0); left-hand-side
            coefficients are left untouched.
        confidence : float
            Confidence in this view, between 0.0 (no confidence) and 1.0
            (certainty). Maps to view confidences in Black-Litterman and
            Entropy Pooling.

        Returns
        -------
        PriorSynthesiser
            Returns self to enable fluent method chaining.

        Raises
        ------
        ValueError
            If confidence is not in [0.0, 1.0].
        ValueError
            If view_str references an identifier that is neither a ticker in
            universe.tickers nor a group in universe.metadata.
        ValueError
            If view_str has no valid comparison operator or its right-hand
            side is not a scalar numeric return.
        """
        if not (0.0 <= confidence <= 1.0):
            raise ValueError("confidence must be between 0.0 and 1.0")

        valid_tickers = set(self._universe.tickers)
        valid_groups = (
            set(self._universe.metadata.values())
            if isinstance(getattr(self._universe, "metadata", None), dict)
            else set()
        )
        valid_identifiers = valid_tickers | valid_groups

        # Tokenize by non-word chars (keeping dots and hyphens if possible) to find identifiers
        words = re.findall(r"[\w.-]+", view_str)
        extracted_ids = [w for w in words if w in valid_identifiers]

        # Find tokens that look like identifiers but are not valid
        potential_ids = [
            w
            for w in words
            if re.match(r"^[A-Za-z][A-Za-z0-9_.-]*$", w) and w not in valid_identifiers
        ]

        if not extracted_ids:
            raise ValueError(
                f"View '{view_str}' does not reference any valid tickers or groups. "
                f"Valid identifiers: {sorted(valid_identifiers)}"
            )

        for unknown_id in potential_ids:
            raise ValueError(
                f"View '{view_str}' references identifier '{unknown_id}' which is "
                f"not present in universe tickers or metadata groups."
            )

        normalised_view = self._normalise_view_operators(view_str)
        parts = re.split(r"(>=|<=|==)", normalised_view, maxsplit=1)
        if len(parts) != 3:
            raise ValueError(
                f"View '{view_str}' must contain a valid comparison operator (==, >=, <=)."
            )

        lhs, op, rhs = parts[0].strip(), parts[1], parts[2].strip()
        try:
            rhs_annual = float(rhs)
        except ValueError as exc:
            raise ValueError(
                f"Right-hand side of view '{view_str}' must be a scalar numeric return, got '{rhs}'."
            ) from exc

        # Scale only the annualised RHS constant to a daily return (divide by 252)
        scaled_rhs = rhs_annual / 252.0
        scaled_view = f"{lhs} {op} {scaled_rhs}"

        self._views.append({"view": scaled_view, "confidence": confidence})
        return self

    @staticmethod
    def build_empirical_prior(
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

        raw_metadata = (
            self._universe.metadata
            if isinstance(getattr(self._universe, "metadata", None), dict)
            else {}
        )
        groups: dict[str, list[str]] = {}
        for ticker in getattr(self._universe, "tickers", []):
            if ticker in raw_metadata:
                val = raw_metadata[ticker]
                groups[ticker] = [val] if isinstance(val, str) else list(val)
        groups_arg = groups if groups else None

        return EntropyPooling(mean_views=views, groups=groups_arg)

    def build_black_litterman_prior(
        self,
        tau: float = 0.05,
        prior_estimator: BasePrior | None = None,
        risk_free_rate: float = 0.0,
    ) -> BlackLitterman:
        """Build a Black-Litterman prior from registered market views.

        Parameters
        ----------
        tau : float, default 0.05
            Degree of uncertainty given to the analyst views.
        prior_estimator : BasePrior or None, optional
            Prior distribution estimator. If None, skfolio default is used.
        risk_free_rate : float, default 0.0
            Annualised risk-free rate; converted internally to a daily rate
            (divided by 252.0) to match the frequency of the returns and views.

        Returns
        -------
        skfolio.prior.BlackLitterman

        Raises
        ------
        RuntimeError
            If no market views have been registered via add_market_view().
        """
        if not self._views:
            raise RuntimeError(
                "No market views registered. Call add_market_view() before building a Black-Litterman prior."
            )

        views = [v["view"] for v in self._views]
        confidences = [v["confidence"] for v in self._views]

        raw_metadata = (
            self._universe.metadata
            if isinstance(getattr(self._universe, "metadata", None), dict)
            else {}
        )
        groups: dict[str, list[str]] = {}
        for ticker in getattr(self._universe, "tickers", []):
            if ticker in raw_metadata:
                val = raw_metadata[ticker]
                groups[ticker] = [val] if isinstance(val, str) else list(val)
        groups_arg = groups if groups else None

        kwargs: dict[str, object] = {
            "views": views,
            "tau": tau,
            "view_confidences": confidences,
            "groups": groups_arg,
            "risk_free_rate": risk_free_rate / 252.0,
        }
        if prior_estimator is not None:
            kwargs["prior_estimator"] = prior_estimator

        return BlackLitterman(**kwargs)

    def build_synthetic_prior(
        self, n_samples: int = 5000, random_state: int | None = None
    ) -> SyntheticData:
        """Build a synthetic prior estimator using a VineCopula model.

        Builds a SyntheticData estimator that, when fitted in the pipeline,
        will fit a VineCopula to universe.returns and generate n_samples
        synthetic paths preserving empirical tail dependencies.
        Used for stress-testing portfolio allocations.

        Parameters
        ----------
        n_samples : int, optional
            Number of synthetic return paths to generate. Default is 5000.
        random_state : int or None, optional
            Seed for the random number generator used by the VineCopula.

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

        return SyntheticData(
            distribution_estimator=VineCopula(random_state=random_state),
            n_samples=n_samples,
        )
