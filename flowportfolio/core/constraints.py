"""Constraint builder module.

This module provides the :class:`ConstraintBuilder` class, which offers a fluent
API for generating ``skfolio``-compatible linear constraints based on the
asset universe structure.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field

from flowportfolio.core.protocols import UniverseProtocol


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
    previous_weights : list[float] | None, optional
        Previous weights for turnover constraints.
    max_turnover : float | None, optional
        Maximum turnover limit.
    management_fees : list[float] | None, optional
        Management fees for each asset.
    """

    linear_constraints: list[str] = field(default_factory=list)
    groups: dict[str, list[str]] = field(default_factory=dict)
    previous_weights: list[float] | None = None
    max_turnover: float | None = None
    management_fees: list[float] | None = None

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
            )
        if isinstance(other, list):
            return self.linear_constraints == other
        return False


class ConstraintBuilder:
    """Fluent builder for generating skfolio linear constraints.

    Generates strings compatible with ``skfolio``'s ``linear_constraints``
    parser by leveraging the group metadata defined in the :class:`Universe`.

    Parameters
    ----------
    universe : UniverseProtocol
        The asset universe satisfying :class:`~flowportfolio.core.protocols.UniverseProtocol`
        from which group metadata and ticker lists are drawn.

    Raises
    ------
    TypeError
        If the ``universe`` argument does not implement :class:`UniverseProtocol`,
        or if its ``metadata`` is not a dict or ``tickers`` is not a list.
    """

    def __init__(self, universe: UniverseProtocol) -> None:
        if not hasattr(universe, "tickers") or not hasattr(universe, "metadata"):
            raise TypeError(
                "universe must implement UniverseProtocol "
                "(requires 'tickers: list[str]' and 'metadata: dict[str, str]' properties)."
            )
        if not isinstance(universe.metadata, dict):
            raise TypeError("universe.metadata must be a dictionary.")
        if not isinstance(universe.tickers, list):
            raise TypeError("universe.tickers must be a list.")
        self._universe = universe
        self._valid_groups: set[str] = set(universe.metadata.values())
        self._constraints: list[str] = []
        self._max_turnover: float | None = None
        self._previous_weights: list[float] | None = None

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
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", group_name):
            raise ValueError(f"Invalid group name: '{group_name}'")
        if group_name not in self._valid_groups:
            raise ValueError(f"Group '{group_name}' not found in universe metadata.")
        if not (0.0 <= weight <= 1.0):
            raise ValueError(f"Weight must be between 0.0 and 1.0, got {weight}")
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
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", group_name):
            raise ValueError(f"Invalid group name: '{group_name}'")
        if group_name not in self._valid_groups:
            raise ValueError(f"Group '{group_name}' not found in universe metadata.")
        if not (0.0 <= weight <= 1.0):
            raise ValueError(f"Weight must be between 0.0 and 1.0, got {weight}")
        self._constraints.append(f"{group_name} <= {weight:.6g}")
        return self

    def max_combined_groups(
        self, group_names: list[str], weight: float
    ) -> ConstraintBuilder:
        """Add a maximum weight constraint for a combination of groups.

        Parameters
        ----------
        group_names : list[str]
            A list of group names defined in the universe metadata.
        weight : float
            The maximum combined weight for all assets in all specified groups.

        Returns
        -------
        ConstraintBuilder
            The builder instance for method chaining.

        Raises
        ------
        ValueError
            If any of the groups are not found in the universe metadata, weight is out of bounds,
            or group names are invalid.
        """
        for group in group_names:
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", group):
                raise ValueError(f"Invalid group name: '{group}'")
            if group not in self._valid_groups:
                raise ValueError(f"Group '{group}' not found in universe metadata.")

        if not (0.0 <= weight <= 1.0):
            raise ValueError(f"Weight must be between 0.0 and 1.0, got {weight}")

        combined_str = " + ".join(group_names)
        self._constraints.append(f"{combined_str} <= {weight:.6g}")
        return self

    def max_turnover(
        self, limit: float, current_weights: dict[str, float]
    ) -> ConstraintBuilder:
        """Add turnover constraints to limit deviation from current weights.

        Calculates per-asset upper and lower bounds based on the current
        weights and the specified turnover limit.

        Parameters
        ----------
        limit : float
            The maximum allowed absolute deviation per asset (must be > 0.0
            and <= 1.0).
        current_weights : dict[str, float]
            A dictionary mapping each ticker in the universe to its current
            weight (0.0 to 1.0).

        Returns
        -------
        ConstraintBuilder
            The builder instance for method chaining.

        Raises
        ------
        ValueError
            If the limit is not strictly between 0.0 and 1.0, or if any ticker
            in the universe is missing from current_weights.
        """
        if not (0.0 < limit <= 1.0):
            raise ValueError("Turnover limit must be between 0.0 (exclusive) and 1.0.")

        missing_tickers = set(self._universe.tickers) - set(current_weights.keys())
        if missing_tickers:
            raise ValueError(f"Missing current weights for tickers: {missing_tickers}")

        self._max_turnover = limit
        self._previous_weights = [
            current_weights[ticker] for ticker in self._universe.tickers
        ]

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

        # Include any remaining metadata tickers
        for ticker, val in raw_metadata.items():
            if ticker not in groups:
                groups[ticker] = [val] if isinstance(val, str) else list(val)

        management_fees = None
        fees = getattr(self._universe, "fees", None)
        if isinstance(fees, dict):
            management_fees = [float(fees.get(ticker, 0.0)) for ticker in tickers]

        return ConstraintSpec(
            linear_constraints=list(self._constraints),
            groups=groups,
            previous_weights=self._previous_weights,
            max_turnover=self._max_turnover,
            management_fees=management_fees,
        )

    def reset(self) -> ConstraintBuilder:
        """Clear all accumulated constraints and return self.

        Resets the builder to its initial empty state, ready for a new
        constraint-building chain. This is the only way to discard
        previously accumulated constraints; :meth:`build` is idempotent
        and does not clear internal state.

        Returns
        -------
        ConstraintBuilder
            The builder instance for method chaining.
        """
        self._constraints = []
        self._max_turnover = None
        self._previous_weights = None
        return self
