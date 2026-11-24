"""Prior synthesiser module for flowportfolio."""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from skfolio.distribution import VineCopula
from skfolio.moments import (
    EquilibriumMu,
    LedoitWolf,
    ShrunkMu,
    ShrunkMuMethods,
)
from skfolio.prior import (
    BasePrior,
    BlackLitterman,
    EmpiricalPrior,
    EntropyPooling,
    SyntheticData,
)

from flowportfolio.core.protocols import UniverseProtocol

logger = logging.getLogger(__name__)


class _FlowPortfolioBlackLitterman(BlackLitterman):
    """BlackLitterman prior retaining the economic per-period risk-free rate."""

    def __init__(
        self,
        views: Any,
        groups: Any = None,
        prior_estimator: BasePrior | None = None,
        tau: float = 0.05,
        view_confidences: Any = None,
        risk_free_rate: float = 0.0,
        portfolio_risk_free_rate: float = 0.0,
    ) -> None:
        super().__init__(
            views=views,
            groups=groups,
            prior_estimator=prior_estimator,
            tau=tau,
            view_confidences=view_confidences,
            risk_free_rate=risk_free_rate,
        )
        self.portfolio_risk_free_rate = portfolio_risk_free_rate


_NUM_PAT = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_NUM_RE = re.compile(_NUM_PAT)
_UNKNOWN_ID_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*")


@dataclass(frozen=True)
class _ViewTerm:
    """Single linear term in a parsed market view side."""

    sign: str  # "+" or "-"
    coef: float  # factor loading if ident is not None, else annualized scalar
    ident: str | None  # ticker or group name, or None for standalone scalar
    raw_coef: str | None = None  # original literal for exact non-scaled formatting


def _resolve_periods_per_year(universe: object) -> float:
    """Return a positive finite periods_per_year from universe, defaulting to 252.0."""
    raw = getattr(universe, "periods_per_year", 252)
    if (
        isinstance(raw, (int, float))
        and not isinstance(raw, bool)
        and math.isfinite(raw)
        and raw > 0
    ):
        return float(raw)
    return 252.0


def _build_ident_pattern(valid_identifiers: set[str]) -> str | None:
    """Build a longest-match (maximal munch) regex pattern for valid tickers and groups."""
    if not valid_identifiers:
        return None
    alternation = "|".join(
        re.escape(i) for i in sorted(valid_identifiers, key=len, reverse=True)
    )
    # Boundary prevents ticker 'A' from matching the prefix of unknown 'APPLE'
    # while still allowing subtraction operators like 'BRK-B-SPY'.
    return rf"(?:{alternation})(?![A-Za-z0-9_.])"


def _build_ident_regex(valid_identifiers: set[str]) -> re.Pattern[str] | None:
    """Compile a longest-match (maximal munch) regex for valid tickers and groups."""
    pattern = _build_ident_pattern(valid_identifiers)
    return re.compile(pattern) if pattern is not None else None


def _scan_identifiers(
    expr: str, valid_identifiers: set[str]
) -> tuple[list[str], list[str]]:
    """Scan an operator-normalised expression left-to-right using maximal munch."""
    cleaned = re.sub(r">=|<=|==", " ", expr)
    id_re = _build_ident_regex(valid_identifiers)
    extracted: list[str] = []
    unknown: list[str] = []

    pos = 0
    n = len(cleaned)
    while pos < n:
        ch = cleaned[pos]
        if ch.isspace() or ch in "+-*":
            pos += 1
            continue

        if id_re is not None:
            m_id = id_re.match(cleaned, pos)
            if m_id:
                extracted.append(m_id.group(0))
                pos = m_id.end()
                continue

        m_num = _NUM_RE.match(cleaned, pos)
        if m_num:
            pos = m_num.end()
            continue

        m_unk = _UNKNOWN_ID_RE.match(cleaned, pos)
        if m_unk:
            unknown.append(m_unk.group(0))
            pos = m_unk.end()
            continue

        pos += 1

    return extracted, unknown


def _parse_view_side(
    side_str: str,
    valid_identifiers: set[str],
    raw_view: str,
    side_label: str = "Right-hand side",
) -> list[_ViewTerm]:
    """Parse one side of a view equation into typed _ViewTerm objects."""
    s = side_str.strip()
    if not s:
        raise ValueError(
            f"View '{raw_view}' has an empty expression on one side of the operator."
        )

    # Pure scalar fast path (e.g. "0.05", "-0.02", ".05")
    try:
        scalar_val = float(s)
    except ValueError:
        scalar_val = None

    if scalar_val is not None:
        if not math.isfinite(scalar_val):
            raise ValueError(
                f"{side_label} of view '{raw_view}' must be a finite number, got '{s}'."
            )
        sign = "-" if scalar_val < 0 else "+"
        return [_ViewTerm(sign=sign, coef=abs(scalar_val), ident=None, raw_coef=s)]

    id_pat = _build_ident_pattern(valid_identifiers)
    if id_pat is not None:
        term_body = (
            rf"(?P<coef1>{_NUM_PAT})\s*\*\s*(?P<id1>{id_pat})"
            rf"|(?P<id2>{id_pat})\s*\*\s*(?P<coef2>{_NUM_PAT})"
            rf"|(?P<id3>{id_pat})"
            rf"|(?P<const>{_NUM_PAT})"
        )
    else:
        term_body = rf"(?P<const>{_NUM_PAT})"

    term_re = re.compile(rf"\s*(?P<sign>[+-])?\s*(?:{term_body})\s*")

    terms: list[_ViewTerm] = []
    pos = 0
    while pos < len(s):
        m = term_re.match(s, pos)
        if not m or m.end() == pos:
            raise ValueError(
                f"{side_label} of view '{raw_view}' must be a scalar numeric "
                f"return or a valid linear view expression, got '{side_str}'."
            )
        raw_sign = m.group("sign")
        if terms and raw_sign is None:
            raise ValueError(
                f"{side_label} of view '{raw_view}' must be a scalar numeric "
                f"return or a valid linear view expression, got '{side_str}'."
            )
        sign = raw_sign if raw_sign is not None else "+"
        groups = m.groupdict()
        ident = groups.get("id1") or groups.get("id2") or groups.get("id3")
        coef_str = groups.get("coef1") or groups.get("coef2")
        const_str = groups.get("const")

        if ident is not None:
            if coef_str is not None:
                coef_val = float(coef_str)
                if not math.isfinite(coef_val):
                    raise ValueError(
                        f"Non-finite coefficient '{coef_str}' in view '{raw_view}'."
                    )
                terms.append(
                    _ViewTerm(sign=sign, coef=coef_val, ident=ident, raw_coef=coef_str)
                )
            else:
                terms.append(_ViewTerm(sign=sign, coef=1.0, ident=ident, raw_coef=None))
        else:
            assert const_str is not None
            const_val = float(const_str)
            if not math.isfinite(const_val):
                raise ValueError(
                    f"{side_label} of view '{raw_view}' must be a finite number, got '{const_str}'."
                )
            terms.append(
                _ViewTerm(sign=sign, coef=const_val, ident=None, raw_coef=const_str)
            )

        pos = m.end()

    return terms


def _format_view_side(
    terms: Sequence[_ViewTerm],
    *,
    periods_per_year: float = 252.0,
    group_divisors: dict[str, int] | None = None,
) -> str:
    """Serialize typed _ViewTerm objects into a skfolio equation side string."""
    out: list[str] = []
    for idx, term in enumerate(terms):
        if term.ident is not None:
            divisor = (group_divisors or {}).get(term.ident)
            if divisor is not None:
                scaled_coef = term.coef / divisor
                core = f"{scaled_coef:.12g} * {term.ident}"
            elif term.raw_coef is not None:
                # Normalize leading-dot literals (".5" -> "0.5") while preserving "1.0"
                literal = (
                    f"0{term.raw_coef}"
                    if term.raw_coef.startswith(".")
                    else term.raw_coef
                )
                core = f"{literal} * {term.ident}"
            else:
                core = term.ident
        else:
            core = f"{term.coef / periods_per_year}"

        if idx == 0:
            out.append(f"-{core}" if term.sign == "-" else core)
        else:
            out.append(f"{term.sign} {core}")

    return " ".join(out)


def _compute_group_divisors(
    valid_tickers: set[str], raw_metadata: dict[str, object]
) -> dict[str, int]:
    """Return member counts for multi-member metadata groups that are not ticker names."""
    counts: dict[str, int] = {}
    for value in raw_metadata.values():
        if isinstance(value, str):
            counts[value] = counts.get(value, 0) + 1
    return {g: n for g, n in counts.items() if n > 1 and g not in valid_tickers}


def _prior_uses_excess_returns(
    prior_estimator: BasePrior | None,
    _visited: set[int] | None = None,
) -> bool:
    """Return True if prior_estimator estimates excess returns (skfolio EquilibriumMu convention)."""
    if prior_estimator is None:
        return True
    visited = _visited if _visited is not None else set()
    obj_id = id(prior_estimator)
    if obj_id in visited:
        return False
    visited.add(obj_id)

    if isinstance(prior_estimator, BlackLitterman):
        return False

    mu_est = getattr(prior_estimator, "mu_estimator", None)
    if mu_est is not None:
        return isinstance(mu_est, EquilibriumMu)

    nested_prior = getattr(prior_estimator, "prior_estimator", None)
    if nested_prior is not None:
        return _prior_uses_excess_returns(nested_prior, _visited=visited)
    return False


def _format_bl_equality_view(
    lhs_terms: Sequence[_ViewTerm],
    rhs_terms: Sequence[_ViewTerm],
    *,
    periods_per_year: float,
    risk_free_rate_annual: float,
) -> str:
    """Compile typed equality view terms into a BlackLitterman equation string.

    When risk_free_rate_annual != 0.0 (excess-return prior mode), transforms
    p^T r = q_total into p^T r_excess = q_total - (sum p_j) * risk_free_rate_annual
    before scaling scalar constants by 1 / periods_per_year.
    """
    if risk_free_rate_annual == 0.0:
        lhs_str = _format_view_side(lhs_terms, periods_per_year=periods_per_year)
        rhs_str = _format_view_side(rhs_terms, periods_per_year=periods_per_year)
        return f"{lhs_str} == {rhs_str}"

    # Canonical form: all identifier terms on LHS, net scalar constant on RHS
    id_terms: list[_ViewTerm] = []
    net_loading = 0.0
    rhs_const_ann = 0.0

    for term in lhs_terms:
        signed_val = -term.coef if term.sign == "-" else term.coef
        if term.ident is not None:
            id_terms.append(term)
            net_loading += signed_val
        else:
            rhs_const_ann -= signed_val

    for term in rhs_terms:
        signed_val = -term.coef if term.sign == "-" else term.coef
        if term.ident is not None:
            flipped_sign = "+" if term.sign == "-" else "-"
            id_terms.append(
                _ViewTerm(
                    sign=flipped_sign,
                    coef=term.coef,
                    ident=term.ident,
                    raw_coef=term.raw_coef,
                )
            )
            net_loading -= signed_val
        else:
            rhs_const_ann += signed_val

    excess_rhs_ann = rhs_const_ann - net_loading * risk_free_rate_annual
    lhs_str = _format_view_side(id_terms, periods_per_year=periods_per_year)
    rhs_per_period = excess_rhs_ann / periods_per_year
    if rhs_per_period == 0.0:
        rhs_per_period = 0.0
    return f"{lhs_str} == {rhs_per_period}"


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

    def _raw_metadata(self) -> dict[str, object]:
        """Return universe metadata as a dict, or an empty dict if absent."""
        metadata = getattr(self._universe, "metadata", None)
        return metadata if isinstance(metadata, dict) else {}

    def _build_groups_mapping(self) -> dict[str, list[str]] | None:
        """Build the skfolio ``groups`` mapping from universe tickers and metadata."""
        raw_metadata = self._raw_metadata()
        groups: dict[str, list[str]] = {}
        for ticker in getattr(self._universe, "tickers", []):
            if ticker in raw_metadata:
                val = raw_metadata[ticker]
                if isinstance(val, str):
                    groups[ticker] = [val]
                elif isinstance(val, Sequence):
                    groups[ticker] = [str(item) for item in val]
        return groups if groups else None

    def add_market_view(
        self, view_str: str, confidence: float, *, average_groups: bool = True
    ) -> PriorSynthesiser:
        """Register a forward-looking market view for use in prior estimation.

        Parameters
        ----------
        view_str : str
            A view expression referencing asset tickers or metadata group names
            present in the universe. Example: "SPY >= 0.05" (SPY expected to
            return >=5% annualised). Standalone scalar return constants are
            scaled by ``1 / universe.periods_per_year`` (default 252.0); factor
            coefficients multiplying identifiers are left untouched. Relative
            views with or without scalar spreads (e.g. "SPY >= TLT" or
            "SPY >= TLT + 0.02") are supported.
        confidence : float
            Confidence in this view, between 0.0 (no confidence) and 1.0
            (certainty). Maps to view confidences in Black-Litterman; Entropy
            Pooling enforces views as hard constraints.
        average_groups : bool, default True
            If True, each group identifier with more than one member is scaled
            by ``1 / N`` in Entropy Pooling so that the view constrains the
            group's average return rather than the sum of its members' returns.
            (Black-Litterman always averages group members via ``sum_to_one=True``
            in ``skfolio`` and is not double-scaled.)

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
            If view_str has no valid comparison operator or contains invalid
            syntax.
        """
        if (
            not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or math.isnan(confidence)
            or not (0.0 <= float(confidence) <= 1.0)
        ):
            raise ValueError("confidence must be between 0.0 and 1.0")

        valid_tickers = set(self._universe.tickers)
        raw_metadata = self._raw_metadata()
        valid_groups = {v for v in raw_metadata.values() if isinstance(v, str)}
        valid_identifiers = valid_tickers | valid_groups

        normalised_view = self._normalise_view_operators(view_str)
        parts = re.split(r"(>=|<=|==)", normalised_view, maxsplit=1)
        if len(parts) != 3:
            raise ValueError(
                f"View '{view_str}' must contain a valid comparison operator (==, >=, <=)."
            )

        extracted_ids, unknown_ids = _scan_identifiers(
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

        group_divisors = _compute_group_divisors(valid_tickers, raw_metadata)

        lhs_terms = _parse_view_side(
            lhs, valid_identifiers, view_str, side_label="Left-hand side"
        )
        rhs_terms = _parse_view_side(
            rhs, valid_identifiers, view_str, side_label="Right-hand side"
        )

        lhs_has_id = any(t.ident is not None for t in lhs_terms)
        rhs_has_id = any(t.ident is not None for t in rhs_terms)
        rhs_has_const = any(t.ident is None for t in rhs_terms)
        if not (lhs_has_id or rhs_has_id):
            raise ValueError(
                f"View '{view_str}' does not reference any valid tickers or groups."
            )

        # A pure ranking view has identifiers on the RHS, no scalar constant, and is an inequality
        is_ranking = rhs_has_id and not rhs_has_const and op != "=="
        ppy = _resolve_periods_per_year(self._universe)

        bl_lhs = _format_view_side(lhs_terms, periods_per_year=ppy)
        bl_rhs = _format_view_side(rhs_terms, periods_per_year=ppy)
        base_view = f"{bl_lhs} {op} {bl_rhs}"

        if average_groups and group_divisors:
            ep_lhs = _format_view_side(
                lhs_terms, periods_per_year=ppy, group_divisors=group_divisors
            )
            ep_rhs = _format_view_side(
                rhs_terms, periods_per_year=ppy, group_divisors=group_divisors
            )
            ep_view = f"{ep_lhs} {op} {ep_rhs}"
        else:
            ep_view = base_view

        self._views.append(
            {
                "view": ep_view,
                "bl_view": base_view,
                "operator": op,
                "confidence": float(confidence),
                "is_ranking": is_ranking,
                "raw_view": view_str,
                "lhs_terms": tuple(lhs_terms),
                "rhs_terms": tuple(rhs_terms),
                "average_groups": average_groups,
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
            Defaults to LedoitWolf().
        mu_estimator : skfolio mu estimator, optional
            Defaults to ShrunkMu(method=ShrunkMuMethods.BAYES_STEIN).

        Returns
        -------
        skfolio.prior.EmpiricalPrior
        """
        if covariance_estimator is None:
            covariance_estimator = LedoitWolf()
        if mu_estimator is None:
            mu_estimator = ShrunkMu(method=ShrunkMuMethods.BAYES_STEIN)

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

        raw_metadata = self._raw_metadata()
        valid_tickers = set(getattr(self._universe, "tickers", []))
        group_divisors = _compute_group_divisors(valid_tickers, raw_metadata)
        ppy = _resolve_periods_per_year(self._universe)

        views: list[str] = []
        for v in self._views:
            if "lhs_terms" in v and "rhs_terms" in v:
                divisors = (
                    group_divisors
                    if v.get("average_groups", True) and group_divisors
                    else None
                )
                ep_lhs = _format_view_side(
                    v["lhs_terms"], periods_per_year=ppy, group_divisors=divisors
                )
                ep_rhs = _format_view_side(
                    v["rhs_terms"], periods_per_year=ppy, group_divisors=divisors
                )
                views.append(f"{ep_lhs} {v['operator']} {ep_rhs}")
            else:
                views.append(v["view"])

        return EntropyPooling(mean_views=views, groups=self._build_groups_mapping())

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
            Degree of uncertainty given to the analyst views. Must be > 0.
        prior_estimator : BasePrior or None, optional
            Prior distribution estimator. If None, skfolio default is used.
        risk_free_rate : float, default 0.0
            Annualised risk-free rate; converted internally to a per-period rate
            (divided by ``universe.periods_per_year``) when using excess-return
            priors (``EquilibriumMu``). Suppressed for total-return priors to
            prevent double-counting.

        Returns
        -------
        skfolio.prior.BlackLitterman

        Raises
        ------
        RuntimeError
            If no market views have been registered via add_market_view().
        ValueError
            If tau <= 0, risk_free_rate is non-finite, or any registered view
            is an inequality or pure ranking view.
        """
        if not self._views:
            raise RuntimeError(
                "No market views registered. Call add_market_view() before building a Black-Litterman prior."
            )

        if (
            not isinstance(tau, (int, float))
            or isinstance(tau, bool)
            or not math.isfinite(float(tau))
            or float(tau) <= 0.0
        ):
            raise ValueError(f"tau must be a positive finite number, got {tau}.")

        if (
            not isinstance(risk_free_rate, (int, float))
            or isinstance(risk_free_rate, bool)
            or not math.isfinite(float(risk_free_rate))
        ):
            raise ValueError(
                f"risk_free_rate must be a finite number, got {risk_free_rate}."
            )

        for v in self._views:
            if v["operator"] in (">=", "<=") or v.get("is_ranking", False):
                raise ValueError(
                    "BlackLitterman only supports linear equality views ('=='); "
                    f"got '{v['raw_view']}'. Use "
                    "build_entropy_prior() for inequality or ranking views."
                )
            if v["confidence"] == 0.0:
                raise ValueError(
                    "BlackLitterman view confidence must be strictly greater than "
                    f"0.0; got 0.0 for view '{v['raw_view']}'."
                )

        ppy = _resolve_periods_per_year(self._universe)
        rfr_annual = float(risk_free_rate)
        uses_excess = _prior_uses_excess_returns(prior_estimator)

        if uses_excess:
            effective_view_rfr_ann = rfr_annual
            bl_rfr_per_period = rfr_annual / ppy
        else:
            if rfr_annual != 0.0:
                logger.warning(
                    "prior_estimator '%s' estimates total returns directly (already including "
                    "the risk-free rate). Suppressing BlackLitterman's additive risk_free_rate "
                    "offset to prevent double-counting.",
                    type(prior_estimator).__name__,
                )
            effective_view_rfr_ann = 0.0
            bl_rfr_per_period = 0.0

        # BlackLitterman already sets sum_to_one=True in equations_to_matrix,
        # so format without group_divisors to avoid 1/N^2 scaling.
        views = [
            _format_bl_equality_view(
                v["lhs_terms"],
                v["rhs_terms"],
                periods_per_year=ppy,
                risk_free_rate_annual=effective_view_rfr_ann,
            )
            if "lhs_terms" in v and "rhs_terms" in v
            else v.get("bl_view", v["view"])
            for v in self._views
        ]
        confidences = [v["confidence"] for v in self._views]

        kwargs: dict[str, object] = {
            "views": views,
            "tau": float(tau),
            "view_confidences": confidences,
            "groups": self._build_groups_mapping(),
            "risk_free_rate": bl_rfr_per_period,
            "portfolio_risk_free_rate": rfr_annual / ppy,
        }
        if prior_estimator is not None:
            kwargs["prior_estimator"] = prior_estimator

        return _FlowPortfolioBlackLitterman(**kwargs)  # type: ignore[arg-type]

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
            Non-negative integer seed for the VineCopula random number
            generator. Pass an integer for reproducible sampling.

        Returns
        -------
        skfolio.prior.SyntheticData

        Raises
        ------
        TypeError
            If n_samples is not an integer.
        ValueError
            If n_samples < 100 or random_state is not None or a non-negative integer.
        """
        if not isinstance(n_samples, int) or isinstance(n_samples, bool):
            raise TypeError("n_samples must be an integer.")
        if n_samples < 100:
            raise ValueError(f"n_samples must be at least 100, got {n_samples}")
        if random_state is not None and (
            not isinstance(random_state, int)
            or isinstance(random_state, bool)
            or random_state < 0
        ):
            raise ValueError(
                f"random_state must be a non-negative integer or None, got {random_state!r}."
            )
        if random_state is None:
            logger.warning(
                "build_synthetic_prior() called with random_state=None; VineCopula "
                "sampling will be non-deterministic across runs."
            )

        return SyntheticData(
            distribution_estimator=VineCopula(random_state=random_state),
            n_samples=n_samples,
        )
