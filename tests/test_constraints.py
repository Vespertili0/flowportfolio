"""Unit tests for the ConstraintBuilder class.

This module verifies the generation of skfolio-compatible linear constraints
using the fluent builder pattern in :class:`flowportfolio.core.constraints.ConstraintBuilder`.
"""

import pytest

from flowportfolio.core.constraints import ConstraintBuilder
from flowportfolio.core.protocols import UniverseProtocol
from flowportfolio.core.universe import Universe

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_universe() -> Universe:
    """Provide a minimal Universe stub without fetching data.

    Using a 3-ticker universe: A (core), B (core), C (satellite).
    """
    tickers = ["A", "B", "C"]
    metadata = {"A": "core", "B": "core", "C": "satellite"}
    fees = {"A": 0.001, "B": 0.001, "C": 0.002}
    # Instantiate without calling fetch_data()
    return Universe(tickers=tickers, metadata=metadata, fees=fees)


@pytest.fixture
def stub_protocol_universe() -> UniverseProtocol:
    """Provide a minimal duck-typed stub that satisfies UniverseProtocol.

    This stub does NOT subclass Universe, verifying that ConstraintBuilder
    accepts any conforming object rather than requiring the concrete class.
    """

    class _MinimalStub:
        @property
        def tickers(self) -> list[str]:
            return ["X", "Y"]

        @property
        def metadata(self) -> dict[str, str]:
            return {"X": "growth", "Y": "value"}

    return _MinimalStub()  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def test_init_valid(stub_universe: Universe) -> None:
    """Test ConstraintBuilder accepts a valid Universe instance."""
    builder = ConstraintBuilder(stub_universe)
    assert builder._universe is stub_universe
    assert builder._constraints == []


def test_init_wrong_type() -> None:
    """Test ConstraintBuilder raises TypeError for non-Universe argument."""
    with pytest.raises(TypeError, match="universe must implement UniverseProtocol"):
        ConstraintBuilder(universe="not_a_universe")  # type: ignore


# ---------------------------------------------------------------------------
# Group constraints
# ---------------------------------------------------------------------------


def test_min_group_output(stub_universe: Universe) -> None:
    """Test min_group correctly formats the skfolio string."""
    builder = ConstraintBuilder(stub_universe)
    result = builder.min_group("core", 0.5).build()
    assert result == ["core >= 0.5"]


def test_max_group_output(stub_universe: Universe) -> None:
    """Test max_group correctly formats the skfolio string."""
    builder = ConstraintBuilder(stub_universe)
    result = builder.max_group("satellite", 0.30).build()
    assert result == ["satellite <= 0.3"]


def test_max_combined_groups_output(stub_universe: Universe) -> None:
    """Test max_combined_groups correctly formats the compound skfolio string."""
    builder = ConstraintBuilder(stub_universe)
    result = builder.max_combined_groups(["core", "satellite"], 0.8).build()
    assert result == ["core + satellite <= 0.8"]


def test_chaining(stub_universe: Universe) -> None:
    """Test that methods can be chained and maintain ordering."""
    builder = ConstraintBuilder(stub_universe)
    result = builder.min_group("core", 0.5).max_group("satellite", 0.3).build()
    assert result == ["core >= 0.5", "satellite <= 0.3"]


def test_unknown_group_raises(stub_universe: Universe) -> None:
    """Test that specifying an unknown group raises ValueError."""
    builder = ConstraintBuilder(stub_universe)
    with pytest.raises(ValueError, match="Group 'unknown' not found"):
        builder.min_group("unknown", 0.5)

    with pytest.raises(ValueError, match="Group 'unknown' not found"):
        builder.max_group("unknown", 0.5)

    with pytest.raises(ValueError, match="Group 'unknown' not found"):
        builder.max_combined_groups(["core", "unknown"], 0.8)


# ---------------------------------------------------------------------------
# Turnover constraints
# ---------------------------------------------------------------------------


def test_max_turnover_output(stub_universe: Universe) -> None:
    """Test max_turnover correctly computes upper and lower bounds."""
    builder = ConstraintBuilder(stub_universe)
    current_weights = {"A": 0.4, "B": 0.4, "C": 0.2}
    result = builder.max_turnover(limit=0.1, current_weights=current_weights).build()

    # 3 tickers * 2 bounds each = 6 constraints
    assert len(result) == 6
    assert "A >= 0.3" in result
    assert "A <= 0.5" in result
    assert "B >= 0.3" in result
    assert "B <= 0.5" in result
    assert "C >= 0.1" in result
    assert "C <= 0.3" in result


def test_max_turnover_clamping(stub_universe: Universe) -> None:
    """Test max_turnover clamps bounds to [0.0, 1.0]."""
    builder = ConstraintBuilder(stub_universe)
    # C weight is 0.05, limit is 0.1 -> lower bound clamped to 0.0
    # A weight is 0.95, limit is 0.1 -> upper bound clamped to 1.0
    current_weights = {"A": 0.95, "B": 0.0, "C": 0.05}
    result = builder.max_turnover(limit=0.1, current_weights=current_weights).build()

    assert "A <= 1" in result
    assert "C >= 0" in result


def test_max_turnover_invalid_limit(stub_universe: Universe) -> None:
    """Test max_turnover rejects invalid limit values."""
    builder = ConstraintBuilder(stub_universe)
    current_weights = {"A": 0.4, "B": 0.4, "C": 0.2}

    with pytest.raises(ValueError, match="Turnover limit must be between 0.0"):
        builder.max_turnover(limit=0.0, current_weights=current_weights)

    with pytest.raises(ValueError, match="Turnover limit must be between 0.0"):
        builder.max_turnover(limit=1.1, current_weights=current_weights)


def test_max_turnover_missing_ticker(stub_universe: Universe) -> None:
    """Test max_turnover raises ValueError if a ticker is missing from weights."""
    builder = ConstraintBuilder(stub_universe)
    current_weights = {"A": 0.5, "B": 0.5}  # Missing 'C'

    with pytest.raises(ValueError, match="Missing current weights for tickers"):
        builder.max_turnover(limit=0.1, current_weights=current_weights)


# ---------------------------------------------------------------------------
# Build behaviour
# ---------------------------------------------------------------------------


def test_build_does_not_mutate(stub_universe: Universe) -> None:
    """Test build() returns a copy, so repeated calls are safe."""
    builder = ConstraintBuilder(stub_universe)
    builder.min_group("core", 0.5)

    result1 = builder.build()
    result2 = builder.build()

    assert result1 == ["core >= 0.5"]
    assert result1 is not result2
    assert result1 == result2


# ---------------------------------------------------------------------------
# Protocol decoupling
# ---------------------------------------------------------------------------


def test_init_accepts_protocol_stub(stub_protocol_universe: UniverseProtocol) -> None:
    """Test ConstraintBuilder accepts any object satisfying UniverseProtocol."""
    builder = ConstraintBuilder(stub_protocol_universe)
    assert builder._universe is stub_protocol_universe
    assert builder._constraints == []


def test_protocol_stub_builds_valid_constraints(
    stub_protocol_universe: UniverseProtocol,
) -> None:
    """Test that constraints can be compiled from a protocol stub."""
    builder = ConstraintBuilder(stub_protocol_universe)
    result = builder.min_group("growth", 0.5).build()
    assert result == ["growth >= 0.5"]


# ---------------------------------------------------------------------------
# Reset lifecycle
# ---------------------------------------------------------------------------


def test_reset_clears_constraints(stub_universe: Universe) -> None:
    """Test reset() clears accumulated constraints."""
    builder = ConstraintBuilder(stub_universe)
    builder.min_group("core", 0.5)
    assert builder.build() == ["core >= 0.5"]

    builder.reset()
    assert builder.build() == []


def test_reset_returns_self_for_chaining(stub_universe: Universe) -> None:
    """Test reset() returns self, enabling fluent chaining."""
    builder = ConstraintBuilder(stub_universe)
    result = builder.min_group("core", 0.5).reset().max_group("satellite", 0.3).build()
    assert result == ["satellite <= 0.3"]


def test_build_is_idempotent_without_reset(stub_universe: Universe) -> None:
    """Test build() does not clear state; calling it twice yields same result."""
    builder = ConstraintBuilder(stub_universe)
    builder.min_group("core", 0.5)
    first = builder.build()
    second = builder.build()
    assert first == second == ["core >= 0.5"]
    assert first is not second  # distinct list objects


def test_init_raises_if_metadata_not_dict() -> None:
    """Test ConstraintBuilder raises TypeError if universe.metadata is not a dict."""

    class _BadMetadata:
        def __init__(self) -> None:
            self.tickers = ["A"]
            self.metadata = "not-a-dict"

    with pytest.raises(TypeError, match="universe.metadata must be a dictionary"):
        ConstraintBuilder(_BadMetadata())  # type: ignore[arg-type]


def test_init_raises_if_tickers_not_list() -> None:
    """Test ConstraintBuilder raises TypeError if universe.tickers is not a list."""

    class _BadTickers:
        def __init__(self) -> None:
            self.tickers = "not-a-list"
            self.metadata = {"A": "growth"}

    with pytest.raises(TypeError, match="universe.tickers must be a list"):
        ConstraintBuilder(_BadTickers())  # type: ignore[arg-type]
