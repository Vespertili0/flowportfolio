"""Unit tests for the ConstraintBuilder class.

This module verifies the generation of skfolio-compatible linear constraints
using the fluent builder pattern in :class:`flowportfolio.core.constraints.ConstraintBuilder`.
"""

import pytest

from flowportfolio.core.constraints import ConstraintBuilder
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
    with pytest.raises(TypeError, match="universe must be a Universe instance"):
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
