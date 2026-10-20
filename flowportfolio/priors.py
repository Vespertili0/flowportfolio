"""Prior synthesiser module for flowportfolio."""

from __future__ import annotations

import logging
import math
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

logger = logging.getLogger(__name__)


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

    @staticmethod
    def _tokenize_identifiers(
        expr: str, valid_identifiers: set[str]
    ) -> tuple[list[str], list[str]]:
        """Extract known and unknown identifiers from a view expression.

        Tokens containing a hyphen that are not themselves valid identifiers
        (e.g. ``"SPY-TLT"``) are split on the hyphen so that relative views
        written without whitespace are parsed correctly.

        Parameters
        ----------
        expr : str
            The (operator-normalised) view expression.
        valid_identifiers : set[str]
            Tickers and group names present in the universe.

        Returns
        -------
        tuple[list[str], list[str]]
            The valid identifiers found, followed by the unknown identifiers.
        """
        raw_tokens = re.findall(r"[\w.-]+", expr)
        expanded_tokens: list[str] = []
        for tok in raw_tokens:
            if tok in valid_identifiers or "-" not in tok:
                expanded_tokens.append(tok)
            else:
                expanded_tokens.extend(part for part in tok.split("-") if part)

        extracted = [w for w in expanded_tokens if w in valid_identifiers]
        unknown = [
            w
            for w in expanded_tokens
            if re.match(r"^[A-Za-z_][A-Za-z0-9_.-]*$", w)
            and not re.match(r"^[eE][+-]?\d+$", w)
            and w not in valid_identifiers
        ]
        return extracted, unknown

    @staticmethod
    def _average_group_identifiers(
        expr: str, valid_tickers: set[str], metadata: dict
    ) -> str:
        """Scale each multi-member group identifier in ``expr`` by ``1 / N``.

        An existing ``coef *`` prefix is multiplied by ``1 / N`` rather than
        duplicated. Identifiers that are also tickers are left untouched.
        """
        counts: dict[str, int] = {}
        for value in metadata.values():
            if isinstance(value, str):
                counts[value] = counts.get(value, 0) + 1
        scalable = {g: n for g, n in counts.items() if n > 1 and g not in valid_tickers}
        if not scalable:
            return expr

        alternation = "|".join(
            re.escape(g) for g in sorted(scalable, key=len, reverse=True)
        )
        pattern = re.compile(
            r"(?:(?P<coef>\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)\s*\*\s*)?"
            r"(?<![\w.])(?P<ident>" + alternation + r")(?![\w.])"
        )

        def _replace(match: re.Match) -> str:
            ident = match.group("ident")
            coef = float(match.group("coef")) if match.group("coef") else 1.0
            return f"{coef / scalable[ident]:.8g} * {ident}"

        return pattern.sub(_replace, expr)

    def add_market_view(
        self, view_str: str, confidence: float, *, average_groups: bool = False
    ) -> PriorSynthesiser:
        """Register a forward-looking market view for use in prior estimation.

        Parameters
        ----------
        view_str : str
            A view expression referencing asset tickers or metadata group names
            present in the universe. Example: "SPY >= 0.05" (SPY expected to
            return >=5% annualised). A scalar right-hand side is converted to a
            daily return (divided by 252.0); left-hand-side coefficients are
            left untouched. A right-hand side that is a pure linear expression
            of identifiers (e.g. "SPY >= TLT") is treated as a ranking view,
            which is only supported by Entropy Pooling.
        confidence : float
            Confidence in this view, between 0.0 (no confidence) and 1.0
            (certainty). Maps to view confidences in Black-Litterman; Entropy
            Pooling enforces views as hard constraints.
        average_groups : bool, default False
            If True, each group identifier with more than one member is scaled
            by ``1 / N`` so that the view constrains the group's average return
            rather than the sum of its members' returns (skfolio's default).

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
            side is neither a scalar numeric return nor a pure ranking
            expression.
        """
        if (
            not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or math.isnan(confidence)
            or not (0.0 <= float(confidence) <= 1.0)
        ):
            raise ValueError("confidence must be between 0.0 and 1.0")

        valid_tickers = set(self._universe.tickers)
        raw_metadata = (
            self._universe.metadata
            if isinstance(getattr(self._universe, "metadata", None), dict)
            else {}
        )
        valid_groups = set(raw_metadata.values())
        valid_identifiers = valid_tickers | valid_groups

        normalised_view = self._normalise_view_operators(view_str)
        parts = re.split(r"(>=|<=|==)", normalised_view, maxsplit=1)
        if len(parts) != 3:
            raise ValueError(
                f"View '{view_str}' must contain a valid comparison operator (==, >=, <=)."
            )

        extracted_ids, unknown_ids = self._tokenize_identifiers(
            normalised_view, valid_identifiers
        )
        if not extracted_ids:
            raise ValueError(
                f"View '{view_str}' does not reference any valid tickers or groups. "
                f"Valid identifiers: {sorted(valid_identifiers)}"
            )
        if unknown_ids:
            raise ValueError(
                f"View '{view_str}' references identifier '{unknown_ids[0]}' which is "
                f"not present in the universe (tickers or metadata groups)."
            )

        lhs, op, rhs = parts[0].strip(), parts[1], parts[2].strip()
        if re.search(r"[<>=!]", lhs) or re.search(r"[<>=!]", rhs):
            raise ValueError(
                f"View '{view_str}' must contain exactly one comparison operator."
            )

        # Determine whether the right-hand side is a scalar return or a ranking.
        try:
            rhs_annual: float | None = float(rhs)
        except ValueError:
            rhs_annual = None

        if rhs_annual is not None:
            if not math.isfinite(rhs_annual):
                raise ValueError(
                    f"Right-hand side of view '{view_str}' must be a finite number, got '{rhs}'."
                )
            # Scale only the annualised RHS constant to a daily return (divide by 252)
            scaled_rhs = f"{rhs_annual / 252.0}"
            is_ranking = False
        else:
            id_alternation = "|".join(
                re.escape(i) for i in sorted(valid_identifiers, key=len, reverse=True)
            )
            term = (
                r"(?:\d+(?:\.\d*)?(?:[eE][+-]?\d+)?\s*\*\s*)?(?:" + id_alternation + r")"
            )
            ranking_pattern = rf"^[+-]?\s*{term}(?:\s*[+-]\s*{term})*$"
            if not re.match(ranking_pattern, rhs):
                raise ValueError(
                    f"Right-hand side of view '{view_str}' must be a scalar numeric "
                    f"return or a pure ranking expression, got '{rhs}'."
                )
            scaled_rhs = rhs
            is_ranking = True

        if average_groups:
            lhs = self._average_group_identifiers(lhs, valid_tickers, raw_metadata)
            if is_ranking:
                scaled_rhs = self._average_group_identifiers(
                    scaled_rhs, valid_tickers, raw_metadata
                )

        self._views.append(
            {
                "view": f"{lhs} {op} {scaled_rhs}",
                "operator": op,
                "confidence": float(confidence),
                "is_ranking": is_ranking,
                "raw_view": view_str,
            }
        )
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

        if any(v["confidence"] < 1.0 for v in self._views):
            logger.warning(
                "EntropyPooling enforces views as hard constraints and ignores "
                "fractional confidence values (< 1.0)."
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

        for v in self._views:
            if v["operator"] in (">=", "<=") or v.get("is_ranking", False):
                raise ValueError(
                    "BlackLitterman only supports linear equality views ('==') "
                    f"with scalar returns; got '{v['raw_view']}'. Use "
                    "build_entropy_prior() for inequality or ranking views."
                )
            if v["confidence"] == 0.0:
                raise ValueError(
                    "BlackLitterman view confidence must be strictly greater than "
                    f"0.0; got 0.0 for view '{v['raw_view']}'."
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
