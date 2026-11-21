"""Constraint builder module.

This module provides the :class:`ConstraintBuilder` class, which offers a fluent
API for generating ``skfolio``-compatible linear constraints based on the
asset universe structure.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator
from dataclasses import dataclass, field

from flowportfolio.core.protocols import UniverseProtocol
from flowportfolio.core.validation import validate_fee, validate_weight

_SIMPLEX_TOL: float = 1e-6
_GROUP_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


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


def _validate_holding_period(holding_period: float) -> float:
    """Validate that holding_period is a strictly positive finite number."""
    if not isinstance(holding_period, (int, float)) or isinstance(holding_period, bool):
        raise TypeError("holding_period must be numeric.")
    hp = float(holding_period)
    if not math.isfinite(hp) or hp <= 0.0:
        raise ValueError(
            f"holding_period must be strictly positive and finite, got {holding_period}."
        )
    return hp


@dataclass(frozen=True)
class ConstraintSpec:
    """Structured container holding linear constraints and asset group mappings.

    This specification satisfies both the ``linear_constraints`` string
    grammar and the asset grouping dictionary contract expected by ``skfolio``
    optimisers. It implements sequence dunder methods so that existing code and
    tests expecting a ``list[str]`` continue to operate seamlessly.

    Parameters
    ----------
    linear_constraints : list[str], optional
        List of formatted linear constraint expressions (e.g. ``["core >= 0.40"]``).
    groups : dict[str, list[str]], optional
        Mapping of asset ticker to list of group labels
        (e.g. ``{"AAPL": ["tech"]}``).
    previous_weights : dict[str, float] | None, optional
        Previous weights for turnover constraints, keyed by ticker.
    max_turnover : float | None, optional
        Maximum per-asset absolute weight deviation ``|w_i - w_prev_i| <= limit``
        enforced by ``skfolio`` (not portfolio-level L1 half-turnover).
    management_fees : dict[str, float] | None, optional
        Period-scaled management fees (annual fee divided by
        ``universe.periods_per_year``), keyed by ticker so that ``skfolio`` can
        align them by column name.
    transaction_costs : dict[str, float] | float | None, optional
        Period-amortized proportional transaction costs (``cost / holding_period``),
        either uniform (scalar float) or keyed by ticker.
    """

    linear_constraints: list[str] = field(default_factory=list)
    groups: dict[str, list[str]] = field(default_factory=dict)
    previous_weights: dict[str, float] | None = None
    max_turnover: float | None = None
    management_fees: dict[str, float] | None = None
    transaction_costs: dict[str, float] | float | None = None
    holding_period: float | None = None

    def __iter__(self) -> Iterator[str]:
        return iter(self.linear_constraints)

    def __len__(self) -> int:
        return len(self.linear_constraints)

    def __getitem__(self, index: int) -> str:
        return self.linear_constraints[index]

    def __contains__(self, item: object) -> bool:
        return item in self.linear_constraints

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ConstraintSpec):
            return (
                self.linear_constraints == other.linear_constraints
                and self.groups == other.groups
                and self.previous_weights == other.previous_weights
                and self.max_turnover == other.max_turnover
                and self.management_fees == other.management_fees
                and self.transaction_costs == other.transaction_costs
                and self.holding_period == other.holding_period
            )
        if isinstance(other, list):
            return self.linear_constraints == other
        return NotImplemented


class ConstraintBuilder:
    """Fluent builder for generating skfolio linear constraints.

    Generates strings compatible with ``skfolio``'s ``linear_constraints``
    parser by leveraging the group metadata defined in the :class:`Universe`.

    Parameters
    ----------
    universe : UniverseProtocol
        The asset universe satisfying :class:`~flowportfolio.core.protocols.UniverseProtocol`
        from which group metadata and ticker lists are drawn.
    holding_period : float | None, optional
        Expected holding horizon expressed in return periods matching
        ``universe.returns`` (e.g. ``21`` for monthly rebalancing on daily
        returns, ``63`` for quarterly rebalancing). When provided, proportional
        transaction costs in :meth:`transaction_costs` are divided by
        ``holding_period``. When ``None`` (default), costs remain unamortized
        in :class:`ConstraintSpec` so :class:`PortfolioExperimentEngine` can
        reconcile them automatically to the outer evaluation ``test_size``.

    Raises
    ------
    TypeError
        If the ``universe`` argument does not implement :class:`UniverseProtocol`,
        if its ``metadata`` is not a dict or ``tickers`` is not a list, or if
        ``holding_period`` is not numeric.
    ValueError
        If ``holding_period`` is not strictly positive and finite.
    """

    def __init__(
        self, universe: UniverseProtocol, *, holding_period: float | None = None
    ) -> None:
        if not hasattr(universe, "tickers") or not hasattr(universe, "metadata"):
            raise TypeError(
                "universe must implement UniverseProtocol "
                "(requires 'tickers: list[str]' and 'metadata: dict[str, str]' properties)."
            )
        if not isinstance(universe.metadata, dict):
            raise TypeError("universe.metadata must be a dictionary.")
        if not isinstance(universe.tickers, list):
            raise TypeError("universe.tickers must be a list.")
        self._holding_period: float | None = (
            _validate_holding_period(holding_period)
            if holding_period is not None
            else None
        )
        self._universe = universe
        self._valid_groups: set[str] = {
            str(universe.metadata[t])
            for t in universe.tickers
            if t in universe.metadata
        }
        self._constraints: list[str] = []
        self._max_turnover: float | None = None
        self._previous_weights: dict[str, float] | None = None
        self._transaction_costs: dict[str, float] | float | None = None
        self._tc_holding_period: float | None = None

    def _validate_group_name(self, group_name: str) -> None:
        """Validate group identifier syntax and universe membership."""
        if not _GROUP_NAME_RE.match(group_name):
            raise ValueError(f"Invalid group name: '{group_name}'")
        if group_name not in self._valid_groups:
            raise ValueError(f"Group '{group_name}' not found in universe metadata.")

    def min_group(self, group_name: str, weight: float) -> ConstraintBuilder:
        """Add a minimum weight constraint for a specific group.

        Parameters
        ----------
        group_name : str
            The name of the group as defined in the universe metadata.
        weight : float
            The minimum combined weight for all assets in the group.

        Returns
        -------
        ConstraintBuilder
            The builder instance for method chaining.

        Raises
        ------
        ValueError
            If the group is not found in the universe metadata, weight is out of bounds,
            or group_name is invalid.
        """
        self._validate_group_name(group_name)
        validate_weight(weight)
        self._constraints.append(f"{group_name} >= {weight:.6g}")
        return self

    def max_group(self, group_name: str, weight: float) -> ConstraintBuilder:
        """Add a maximum weight constraint for a specific group.

        Parameters
        ----------
        group_name : str
            The name of the group as defined in the universe metadata.
        weight : float
            The maximum combined weight for all assets in the group.

        Returns
        -------
        ConstraintBuilder
            The builder instance for method chaining.

        Raises
        ------
        ValueError
            If the group is not found in the universe metadata, weight is out of bounds,
            or group_name is invalid.
        """
        self._validate_group_name(group_name)
        validate_weight(weight)
        self._constraints.append(f"{group_name} <= {weight:.6g}")
        return self

    def max_combined_groups(
        self, group_names: list[str], weight: float
    ) -> ConstraintBuilder:
        """Add a maximum weight constraint for a combination of groups.

        Parameters
        ----------
        group_names : list[str]
            A non-empty list of unique group names defined in the universe metadata.
        weight : float
            The maximum combined weight for all assets in all specified groups.

        Returns
        -------
        ConstraintBuilder
            The builder instance for method chaining.

        Raises
        ------
        ValueError
            If ``group_names`` is empty or contains duplicates, if any of the
            groups are not found in the universe metadata, weight is out of bounds,
            or group names are invalid.
        """
        if not group_names:
            raise ValueError("group_names must contain at least one group.")
        if len(set(group_names)) != len(group_names):
            raise ValueError(
                f"Duplicate group names in max_combined_groups: {group_names}"
            )
        for group in group_names:
            self._validate_group_name(group)

        validate_weight(weight)

        combined_str = " + ".join(group_names)
        self._constraints.append(f"{combined_str} <= {weight:.6g}")
        return self

    def transaction_costs(
        self,
        costs: dict[str, float] | float,
        *,
        holding_period: float | None = None,
    ) -> ConstraintBuilder:
        """Add per-asset or uniform proportional transaction costs.

        One-off proportional costs are divided by ``holding_period`` (defaulting
        to the builder's ``holding_period``) so that ``skfolio`` subtracts costs
        on the same per-period scale as single-period expected returns ``mu``.

        Parameters
        ----------
        costs : dict[str, float] | float
            Uniform transaction cost (as a non-negative float) or mapping of
            universe ticker symbols to per-asset transaction costs.
        holding_period : float | None, optional
            Optional override for the expected holding horizon in return periods.
            If ``None``, uses the builder's ``holding_period`` default.

        Returns
        -------
        ConstraintBuilder
            The builder instance for method chaining.

        Raises
        ------
        TypeError
            If ``costs`` is not a float or a dictionary of floats, or if
            ``holding_period`` is not numeric.
        ValueError
            If any cost is negative or non-finite, if ``costs`` contains
            tickers not in the universe, or if ``holding_period`` is not
            strictly positive and finite.
        """
        explicit_hp = (
            self._holding_period
            if holding_period is None
            else _validate_holding_period(holding_period)
        )
        hp = explicit_hp if explicit_hp is not None else 1.0

        if isinstance(costs, (int, float)) and not isinstance(costs, bool):
            cost_val = float(costs)
            if not math.isfinite(cost_val) or cost_val < 0.0:
                raise ValueError(
                    "Transaction costs must be a non-negative finite number."
                )
            self._transaction_costs = cost_val / hp
            self._tc_holding_period = explicit_hp
        elif isinstance(costs, dict):
            unknown_tickers = set(costs.keys()) - set(self._universe.tickers)
            if unknown_tickers:
                raise ValueError(
                    f"Transaction costs contain unknown tickers not in universe: "
                    f"{sorted(unknown_tickers)}."
                )
            cost_dict: dict[str, float] = {}
            for ticker, c in costs.items():
                if not isinstance(c, (int, float)) or isinstance(c, bool):
                    raise TypeError(
                        f"Transaction cost for ticker '{ticker}' must be a number."
                    )
                c_val = float(c)
                if not math.isfinite(c_val) or c_val < 0.0:
                    raise ValueError(
                        f"Transaction cost for ticker '{ticker}' must be non-negative and finite."
                    )
                cost_dict[ticker] = c_val / hp
            self._transaction_costs = cost_dict
            self._tc_holding_period = explicit_hp
        else:
            raise TypeError(
                "Transaction costs must be a float or a dictionary of floats."
            )

        return self

    def max_turnover(
        self, limit: float, current_weights: dict[str, float]
    ) -> ConstraintBuilder:
        """Add per-asset turnover constraints to limit deviation from current weights.

        Configures ``skfolio``'s per-asset turnover bound ``|w_i - w_prev_i| <= limit``
        for every investable asset (note: this is a per-asset L-infinity bound,
        not a portfolio-level L1 half-turnover cap).

        Parameters
        ----------
        limit : float
            The maximum allowed absolute deviation per asset (must be > 0.0
            and <= 1.0).
        current_weights : dict[str, float]
            A dictionary mapping each ticker in the universe to its current
            weight (0.0 to 1.0), summing to 1.0 within ``1e-6`` tolerance.

        Returns
        -------
        ConstraintBuilder
            The builder instance for method chaining.

        Raises
        ------
        TypeError
            If ``current_weights`` is not a dictionary.
        ValueError
            If the limit is not strictly between 0.0 and 1.0, if any ticker
            in the universe is missing from ``current_weights``, if
            ``current_weights`` contains unknown tickers, or if
            ``current_weights`` does not sum to 1.0 within ``1e-6`` tolerance.
        """
        if (
            not isinstance(limit, (int, float))
            or isinstance(limit, bool)
            or not (0.0 < float(limit) <= 1.0)
        ):
            raise ValueError("Turnover limit must be between 0.0 (exclusive) and 1.0.")

        if not isinstance(current_weights, dict):
            raise TypeError(
                "current_weights must be a dict mapping tickers to weights."
            )

        universe_tickers = set(self._universe.tickers)
        unknown_tickers = set(current_weights.keys()) - universe_tickers
        if unknown_tickers:
            raise ValueError(
                f"current_weights contains unknown tickers not in universe: "
                f"{sorted(unknown_tickers)}."
            )

        missing_tickers = universe_tickers - set(current_weights.keys())
        if missing_tickers:
            raise ValueError(f"Missing current weights for tickers: {missing_tickers}")

        for ticker in self._universe.tickers:
            validate_weight(current_weights[ticker])

        total_w = sum(float(current_weights[t]) for t in self._universe.tickers)
        if abs(total_w - 1.0) > _SIMPLEX_TOL:
            raise ValueError(
                f"current_weights must sum to 1.0 (within 1e-6 tolerance); got {total_w}."
            )

        self._max_turnover = float(limit)
        self._previous_weights = {
            ticker: float(current_weights[ticker]) / total_w
            for ticker in self._universe.tickers
        }

        return self

    def build(self) -> ConstraintSpec:
        """Compile the accumulated constraints and asset group bindings.

        Returns
        -------
        ConstraintSpec
            A structured constraint specification containing the string
            constraints and asset group bindings from the universe, suitable
            for passing to ``skfolio`` optimisers.
        """
        raw_metadata = (
            self._universe.metadata
            if isinstance(getattr(self._universe, "metadata", None), dict)
            else {}
        )
        groups: dict[str, list[str]] = {}

        # Include all universe tickers
        tickers = (
            self._universe.tickers
            if isinstance(getattr(self._universe, "tickers", None), list)
            else []
        )
        for ticker in tickers:
            if ticker in raw_metadata:
                val = raw_metadata[ticker]
                groups[ticker] = [val] if isinstance(val, str) else list(val)
            else:
                groups[ticker] = []

        management_fees: dict[str, float] | None = None
        fees = getattr(self._universe, "fees", None)
        if isinstance(fees, dict):
            # Convert annual decimal TER to the live return frequency of universe.returns
            periods_per_year = _resolve_periods_per_year(self._universe)
            management_fees = {}
            for ticker in tickers:
                raw_fee = fees.get(ticker, 0.0)
                validate_fee(raw_fee)
                management_fees[ticker] = float(raw_fee) / periods_per_year

        return ConstraintSpec(
            linear_constraints=list(self._constraints),
            groups=groups,
            previous_weights=(
                dict(self._previous_weights)
                if self._previous_weights is not None
                else None
            ),
            max_turnover=self._max_turnover,
            management_fees=management_fees,
            transaction_costs=(
                dict(self._transaction_costs)
                if isinstance(self._transaction_costs, dict)
                else self._transaction_costs
            ),
            holding_period=(
                self._tc_holding_period if self._transaction_costs is not None else None
            ),
        )

    def reset(self) -> ConstraintBuilder:
        """Clear all accumulated constraints and return self.

        Resets the builder to its initial empty state (preserving the configured
        default ``holding_period``), ready for a new constraint-building chain.
        This is the only way to discard previously accumulated constraints;
        :meth:`build` is idempotent and does not clear internal state.

        Returns
        -------
        ConstraintBuilder
            The builder instance for method chaining.
        """
        self._constraints = []
        self._max_turnover = None
        self._previous_weights = None
        self._transaction_costs = None
        self._tc_holding_period = None
        return self
