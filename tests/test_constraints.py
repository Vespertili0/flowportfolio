"""Unit tests for the ConstraintBuilder class.

This module verifies the generation of skfolio-compatible linear constraints
using the fluent builder pattern in :class:`flowportfolio.core.constraints.ConstraintBuilder`.
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from skfolio.optimization import MeanRisk, ObjectiveFunction

from flowportfolio.core.constraints import ConstraintBuilder, ConstraintSpec
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
    """Test max_turnover correctly stores parameters."""
    builder = ConstraintBuilder(stub_universe)
    current_weights = {"A": 0.4, "B": 0.4, "C": 0.2}
    result = builder.max_turnover(limit=0.1, current_weights=current_weights).build()

    assert result.max_turnover == 0.1
    assert result.previous_weights == {"A": 0.4, "B": 0.4, "C": 0.2}


def test_max_turnover_clamping(stub_universe: Universe) -> None:
    """Test max_turnover stores parameters correctly without clamping."""
    builder = ConstraintBuilder(stub_universe)
    current_weights = {"A": 0.95, "B": 0.0, "C": 0.05}
    result = builder.max_turnover(limit=0.1, current_weights=current_weights).build()

    assert result.max_turnover == 0.1
    assert result.previous_weights == {"A": 0.95, "B": 0.0, "C": 0.05}


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


def test_build_returns_constraint_spec(stub_universe: Universe) -> None:
    """Test build() returns a ConstraintSpec with both constraints and groups."""
    builder = ConstraintBuilder(stub_universe)
    builder.min_group("core", 0.5)
    spec = builder.build()

    assert isinstance(spec, ConstraintSpec)
    assert spec.linear_constraints == ["core >= 0.5"]
    assert spec.groups == {
        "A": ["core"],
        "B": ["core"],
        "C": ["satellite"],
    }


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

    with pytest.raises(
        TypeError,
        match="universe.(metadata must be a dictionary|tickers must be a list)",
    ):
        ConstraintBuilder(_BadMetadata())  # type: ignore[arg-type]


def test_init_raises_if_tickers_not_list() -> None:
    """Test ConstraintBuilder raises TypeError if universe.tickers is not a list."""

    class _BadTickers:
        def __init__(self) -> None:
            self.tickers = "not-a-list"
            self.metadata = {"A": "growth"}

    with pytest.raises(
        TypeError,
        match="universe.(metadata must be a dictionary|tickers must be a list)",
    ):
        ConstraintBuilder(_BadTickers())  # type: ignore[arg-type]


def test_constraint_spec_equality_all_fields() -> None:
    """Test ConstraintSpec equality checks all fields when comparing with another spec."""
    s1 = ConstraintSpec(linear_constraints=["A >= 0.5"], max_turnover=0.1)
    s2 = ConstraintSpec(linear_constraints=["A >= 0.5"], max_turnover=0.2)
    s3 = ConstraintSpec(linear_constraints=["A >= 0.5"], max_turnover=0.1)
    assert s1 != s2
    assert s1 == s3
    assert s1 == ["A >= 0.5"]


def test_constraint_spec_equals_list_and_validates_turnover_weights() -> None:
    """Verify ConstraintSpec list equality and negative/invalid weight rejection in turnover."""
    from flowportfolio.core.universe import Universe

    spec = ConstraintSpec(linear_constraints=["core >= 0.4"])
    assert spec == ["core >= 0.4"]

    u = MagicMock(spec=Universe)
    u.tickers = ["SPY", "TLT"]
    u.metadata = {"SPY": "core", "TLT": "satellite"}
    u.fees = {"SPY": 0.0003, "TLT": 0.0015}
    builder = ConstraintBuilder(u)

    with pytest.raises(ValueError, match="Weight must be between 0.0 and 1.0"):
        builder.max_turnover(0.2, {"SPY": -0.5, "TLT": 1.5})


def test_max_turnover_validates_sum_to_one(stub_universe: Universe) -> None:
    """Verify max_turnover rejects weights not summing to 1.0 and normalizes valid weights."""
    builder = ConstraintBuilder(stub_universe)
    # stub_universe has tickers ['A', 'B', 'C']
    bad_weights = {"A": 0.1, "B": 0.1, "C": 0.1}
    with pytest.raises(
        ValueError, match=r"current_weights must sum to 1\.0 \(within 1e-6 tolerance\)"
    ):
        builder.max_turnover(0.05, bad_weights)

    good_weights = {"A": 0.501, "B": 0.299, "C": 0.200}
    builder.max_turnover(0.05, good_weights)
    spec = builder.build()
    assert spec.max_turnover == 0.05
    assert spec.previous_weights is not None
    assert abs(sum(spec.previous_weights.values()) - 1.0) < 1e-12


def test_transaction_costs_configuration_and_reset(stub_universe: Universe) -> None:
    """Verify ConstraintBuilder.transaction_costs accepts float/dict and resets."""
    builder = ConstraintBuilder(stub_universe)

    # Float cost
    builder.transaction_costs(0.001)
    spec = builder.build()
    assert spec.transaction_costs == 0.001

    # Dict cost
    builder.transaction_costs({"A": 0.001, "B": 0.002, "C": 0.0015})
    spec = builder.build()
    assert spec.transaction_costs == {"A": 0.001, "B": 0.002, "C": 0.0015}

    # Reset
    builder.reset()
    assert builder.build().transaction_costs is None

    # Invalid costs
    with pytest.raises(
        ValueError, match="Transaction costs must be a non-negative finite number"
    ):
        builder.transaction_costs(-0.01)
    with pytest.raises(
        ValueError,
        match="Transaction cost for ticker 'A' must be non-negative and finite",
    ):
        builder.transaction_costs({"A": -0.01, "B": 0.01, "C": 0.01})
    with pytest.raises(
        TypeError, match="Transaction costs must be a float or a dictionary"
    ):
        builder.transaction_costs("invalid")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Cycle 5 (F-06): Strict Simplex, Period Fee Scaling, Holding Period & Oracles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_weights",
    [
        {"A": 0.99, "B": 0.0, "C": 0.0},
        {"A": 0.505, "B": 0.30, "C": 0.20},
        {"A": 0.5 - 2e-6, "B": 0.3, "C": 0.2},
        {"A": 0.5 + 2e-6, "B": 0.3, "C": 0.2},
    ],
)
def test_max_turnover_rejects_simplex_drift_beyond_1e6(
    stub_universe: Universe, bad_weights: dict[str, float]
) -> None:
    """F-06: Reject weight vectors drifting from 1.0 by more than 1e-6 (including 0.99)."""
    builder = ConstraintBuilder(stub_universe)
    with pytest.raises(ValueError, match=r"1e-6"):
        builder.max_turnover(0.05, bad_weights)


def test_max_turnover_accepts_sub_1e6_rounding_dust_and_normalizes(
    stub_universe: Universe,
) -> None:
    """F-06: Tiny floating-point dust within 1e-6 is accepted and normalized to 1.0."""
    builder = ConstraintBuilder(stub_universe)
    dust_weights = {"A": 0.5 + 5e-7, "B": 0.3, "C": 0.2}
    spec = builder.max_turnover(0.05, dust_weights).build()
    assert spec.previous_weights is not None
    assert abs(sum(spec.previous_weights.values()) - 1.0) < 1e-12


def test_max_turnover_rejects_unknown_tickers_and_non_dict(
    stub_universe: Universe,
) -> None:
    """F-06: Reject unknown extra tickers and non-dict current_weights."""
    builder = ConstraintBuilder(stub_universe)
    with pytest.raises(ValueError, match="unknown"):
        builder.max_turnover(0.05, {"A": 0.5, "B": 0.3, "C": 0.2, "GHOST": 0.0})

    with pytest.raises(TypeError, match="current_weights must be a dict"):
        builder.max_turnover(0.05, [("A", 0.5), ("B", 0.3), ("C", 0.2)])  # type: ignore[arg-type]


class _FreqPriceProvider:
    """Synthetic DataProvider producing weekday prices at a specified frequency."""

    def __init__(self, freq: str, periods: int = 30) -> None:
        self._freq = freq
        self._periods = periods

    def fetch(
        self, tickers: list[str], start: str | None = None, end: str | None = None
    ) -> pd.DataFrame:
        idx = pd.date_range("2000-01-03", periods=self._periods, freq=self._freq)
        data = {
            ("Close", t): np.linspace(100.0, 120.0 + i, len(idx))
            for i, t in enumerate(tickers)
        }
        return pd.DataFrame(data, index=idx)


@pytest.mark.parametrize(
    ("freq", "expected_ppy"),
    [
        ("B", 252),
        ("W-FRI", 52),
        ("BME", 12),
        ("BQE", 4),
        ("BYE", 1),
    ],
)
def test_build_scales_management_fees_by_live_periods_per_year(
    freq: str, expected_ppy: int
) -> None:
    """F-06: build() divides annual management fees by live universe.periods_per_year."""
    fees = {"A": 0.0052, "B": 0.0104, "C": 0.0}
    u = Universe(
        tickers=["A", "B", "C"],
        metadata={"A": "core", "B": "core", "C": "satellite"},
        fees=fees,
    )
    # Construct builder BEFORE fetch_data() to prove live resolution at build() time
    builder = ConstraintBuilder(u)
    u.fetch_data(data_provider=_FreqPriceProvider(freq=freq, periods=30))
    assert u.periods_per_year == expected_ppy

    spec = builder.build()
    assert spec.management_fees is not None
    for ticker, annual_fee in fees.items():
        assert spec.management_fees[ticker] == pytest.approx(
            annual_fee / float(expected_ppy), rel=0, abs=1e-15
        )


def test_build_management_fees_magicmock_periods_per_year_fallback() -> None:
    """F-06: MagicMock(spec=Universe) with unset periods_per_year falls back to 252.0, not 1.0."""
    u = MagicMock(spec=Universe)
    u.tickers = ["A", "B"]
    u.metadata = {"A": "core", "B": "satellite"}
    u.fees = {"A": 0.00252, "B": 0.00504}

    spec = ConstraintBuilder(u).build()
    assert spec.management_fees is not None
    assert spec.management_fees["A"] == pytest.approx(0.00252 / 252.0, rel=0, abs=1e-15)
    assert spec.management_fees["B"] == pytest.approx(0.00504 / 252.0, rel=0, abs=1e-15)


@pytest.mark.parametrize("bad_fee", [-0.01, float("nan"), float("inf")])
def test_build_validates_fees_from_protocol_stub(bad_fee: float) -> None:
    """F-06: Duck-typed UniverseProtocol stubs with invalid fees raise ValueError."""

    class _BadFeeStub:
        @property
        def tickers(self) -> list[str]:
            return ["A", "B"]

        @property
        def metadata(self) -> dict[str, str]:
            return {"A": "core", "B": "satellite"}

        @property
        def fees(self) -> dict[str, float]:
            return {"A": bad_fee, "B": 0.001}

    builder = ConstraintBuilder(_BadFeeStub())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Fee must be between 0.0 and 1.0"):
        builder.build()


def test_transaction_costs_holding_period_amortization_and_reset(
    stub_universe: Universe,
) -> None:
    """F-06: Builder-level and per-call holding_period amortize transaction costs cleanly."""
    builder = ConstraintBuilder(stub_universe, holding_period=21)

    # Scalar cost amortized by builder default holding_period=21
    spec_scalar = builder.transaction_costs(0.0021).build()
    assert spec_scalar.transaction_costs == pytest.approx(0.0001, rel=0, abs=1e-15)

    # Per-call holding_period override with partial valid-ticker dict
    spec_override = builder.transaction_costs(
        {"A": 0.0021, "B": 0.0042}, holding_period=7
    ).build()
    assert spec_override.transaction_costs == {
        "A": pytest.approx(0.0003, rel=0, abs=1e-15),
        "B": pytest.approx(0.0006, rel=0, abs=1e-15),
    }

    # reset() clears transaction_costs while preserving builder default holding_period=21
    builder.reset()
    assert builder.build().transaction_costs is None
    spec_after_reset = builder.transaction_costs(0.0042).build()
    assert spec_after_reset.transaction_costs == pytest.approx(0.0002, rel=0, abs=1e-15)


@pytest.mark.parametrize("bad_hp", [0, 0.0, -5, float("nan"), float("inf")])
def test_holding_period_rejects_non_positive_or_non_finite(
    stub_universe: Universe, bad_hp: float
) -> None:
    """F-06: holding_period must be strictly positive and finite."""
    with pytest.raises(ValueError, match="holding_period"):
        ConstraintBuilder(stub_universe, holding_period=bad_hp)

    builder = ConstraintBuilder(stub_universe)
    with pytest.raises(ValueError, match="holding_period"):
        builder.transaction_costs(0.001, holding_period=bad_hp)


@pytest.mark.parametrize("bad_hp", [True, False, "21"])
def test_holding_period_rejects_bool_and_non_numeric(
    stub_universe: Universe, bad_hp: object
) -> None:
    """F-06: holding_period rejects booleans and non-numeric types with TypeError."""
    with pytest.raises(TypeError, match="holding_period must be numeric"):
        ConstraintBuilder(stub_universe, holding_period=bad_hp)  # type: ignore[arg-type]

    builder = ConstraintBuilder(stub_universe)
    with pytest.raises(TypeError, match="holding_period must be numeric"):
        builder.transaction_costs(0.001, holding_period=bad_hp)  # type: ignore[arg-type]


def test_transaction_costs_rejects_unknown_tickers_nan_inf_and_bool(
    stub_universe: Universe,
) -> None:
    """F-06: transaction_costs validates unknown tickers, bools, and non-finite numbers."""
    builder = ConstraintBuilder(stub_universe)

    with pytest.raises(ValueError, match="unknown"):
        builder.transaction_costs({"A": 0.001, "TYPO": 0.002})

    for bad_val in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="non-negative"):
            builder.transaction_costs(bad_val)
        with pytest.raises(ValueError, match="non-negative"):
            builder.transaction_costs({"A": bad_val})

    with pytest.raises(TypeError, match="Transaction costs"):
        builder.transaction_costs(True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Transaction cost for ticker 'A'"):
        builder.transaction_costs({"A": True})  # type: ignore[arg-type]


def test_max_combined_groups_rejects_empty_and_duplicate_groups(
    stub_universe: Universe,
) -> None:
    """Edge case: max_combined_groups rejects empty lists and duplicate group names."""
    builder = ConstraintBuilder(stub_universe)

    with pytest.raises(ValueError, match="at least one group"):
        builder.max_combined_groups([], 0.5)

    with pytest.raises(ValueError, match="Duplicate"):
        builder.max_combined_groups(["core", "core"], 0.5)


def test_skfolio_mean_risk_per_asset_max_turnover_oracle() -> None:
    """F-06 (O3_CLONE_TURNOVER): Lock down skfolio's per-asset |w_i - w_prev_i| <= limit contract."""
    tickers = ["A", "B", "C", "D"]
    u = Universe(
        tickers=tickers,
        metadata={t: "core" for t in tickers},
        fees={t: 0.0 for t in tickers},
    )
    prev = {t: 0.25 for t in tickers}
    spec = ConstraintBuilder(u).max_turnover(0.05, prev).build()

    rng = np.random.default_rng(42)
    returns = pd.DataFrame(rng.normal(0.0005, 0.01, (200, 4)), columns=tickers)
    returns["A"] += 0.004
    returns["C"] += 0.004

    model = MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_RETURN,
        max_turnover=spec.max_turnover,
        previous_weights=spec.previous_weights,
    )
    model.fit(returns)

    w = pd.Series(model.weights_, index=tickers)
    diffs = (w - pd.Series(prev)).abs()
    assert float(diffs.max()) <= 0.05 + 1e-6
    # On 4 assets with two winners and two losers, portfolio L1 half-turnover reaches ~0.10 > 0.05
    l1_half_turnover = float(0.5 * diffs.sum())
    assert l1_half_turnover > 0.05 + 1e-3


def test_skfolio_mean_risk_amortized_transaction_costs_oracle() -> None:
    """F-06: Verify holding_period amortization prevents daily return scale over-penalization."""
    tickers = ["WIN", "LOSE"]
    u = Universe(
        tickers=tickers,
        metadata={"WIN": "core", "LOSE": "core"},
        fees={"WIN": 0.0, "LOSE": 0.0},
    )
    prev = {"WIN": 0.5, "LOSE": 0.5}

    # Daily returns where WIN beats LOSE by 3 bps/day (0.0003), while one-off cost is 10 bps (0.001)
    n_days = 252
    rng = np.random.default_rng(7)
    noise = rng.normal(0.0, 0.005, (n_days, 2))
    noise -= noise.mean(axis=0)
    returns = pd.DataFrame(
        {
            "WIN": 0.0005 + noise[:, 0],
            "LOSE": 0.0002 + noise[:, 1],
        }
    )

    # Unamortized (holding_period=1): 10 bps > 3 bps daily edge -> optimizer freezes at previous_weights
    spec_unamortized = (
        ConstraintBuilder(u, holding_period=1)
        .max_turnover(1.0, prev)
        .transaction_costs(0.001)
        .build()
    )
    m_frozen = MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_RETURN,
        previous_weights=spec_unamortized.previous_weights,
        transaction_costs=spec_unamortized.transaction_costs,
    ).fit(returns)
    assert abs(float(m_frozen.weights_[0]) - 0.5) < 1e-4

    # Annual horizon (holding_period=252): 10 bps / 252 ~= 0.04 bps < 3 bps -> optimizer tilts into WIN
    spec_amortized = (
        ConstraintBuilder(u, holding_period=252)
        .max_turnover(1.0, prev)
        .transaction_costs(0.001)
        .build()
    )
    m_rebalanced = MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_RETURN,
        previous_weights=spec_amortized.previous_weights,
        transaction_costs=spec_amortized.transaction_costs,
    ).fit(returns)
    assert float(m_rebalanced.weights_[0]) > 0.99
