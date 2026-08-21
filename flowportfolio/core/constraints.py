"""Constraint builder module.

This module provides the :class:`ConstraintBuilder` class, which offers a fluent
API for generating ``skfolio``-compatible linear constraints based on the
asset universe structure.
"""

from __future__ import annotations

from flowportfolio.core.universe import Universe


class ConstraintBuilder:
    """Fluent builder for generating skfolio linear constraints.

    Generates strings compatible with ``skfolio``'s ``linear_constraints``
    parser by leveraging the group metadata defined in the :class:`Universe`.

    Parameters
    ----------
    universe : Universe
        The asset universe from which group metadata and ticker lists are
        drawn.

    Raises
    ------
    TypeError
        If the ``universe`` argument is not a :class:`Universe` instance.
    """

    def __init__(self, universe: Universe) -> None:
        if not isinstance(universe, Universe):
            raise TypeError("universe must be a Universe instance.")
        self._universe = universe
        self._constraints: list[str] = []

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
            If the group is not found in the universe metadata.
        """
        if group_name not in self._universe.metadata.values():
            raise ValueError(f"Group '{group_name}' not found in universe metadata.")
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
            If the group is not found in the universe metadata.
        """
        if group_name not in self._universe.metadata.values():
            raise ValueError(f"Group '{group_name}' not found in universe metadata.")
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
            If any of the groups are not found in the universe metadata.
        """
        valid_groups = set(self._universe.metadata.values())
        for group in group_names:
            if group not in valid_groups:
                raise ValueError(f"Group '{group}' not found in universe metadata.")
        
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

        for ticker in self._universe.tickers:
            current_weight = current_weights[ticker]
            
            # Calculate bounds and clamp to [0.0, 1.0]
            lower_bound = max(0.0, current_weight - limit)
            upper_bound = min(1.0, current_weight + limit)
            
            self._constraints.append(f"{ticker} >= {lower_bound:.6g}")
            self._constraints.append(f"{ticker} <= {upper_bound:.6g}")

        return self

    def build(self) -> list[str]:
        """Compile the accumulated constraints.

        Returns
        -------
        list[str]
            A new list containing the string constraints, suitable for passing
            to ``skfolio`` optimisers.
        """
        return list(self._constraints)
