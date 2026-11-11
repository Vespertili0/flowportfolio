"""Property-based tests for quantitative and algebraic invariants in flowportfolio.

Uses ``hypothesis`` to verify domain invariants across:
1. ``PriorSynthesiser`` linear view parsing, daily return constant scaling, and
   group averaging across ``EntropyPooling`` (``sum_to_one=False``) vs
   ``BlackLitterman`` (``sum_to_one=True``).
2. ``PortfolioDeltaEngine`` and ``ConstraintBuilder`` simplex normalisation,
   zero-sum group drift, bounded one-way turnover, and fee/friction accounting.
3. ``Universe.anchor_history`` inception alignment and bounded forward-fill
   invariants.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from skfolio.utils.equations import equations_to_matrix

from flowportfolio.core.constraints import ConstraintBuilder
from flowportfolio.core.universe import Universe
from flowportfolio.delta import PortfolioDeltaEngine
from flowportfolio.priors import PriorSynthesiser

# ---------------------------------------------------------------------------
# 1. PriorSynthesiser View Scaling & Group Averaging Invariants
# ---------------------------------------------------------------------------


@settings(max_examples=50, deadline=None)
@given(
    annual_spread=st.floats(
        min_value=-0.50, max_value=0.50, allow_nan=False, allow_infinity=False
    ).filter(lambda x: abs(x) >= 1e-4),
    c1=st.floats(
        min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False
    ),
    c2=st.floats(
        min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False
    ),
    confidence=st.floats(
        min_value=0.05, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
)
def test_property_prior_view_scaling_and_bl_ep_consistency(
    annual_spread: float, c1: float, c2: float, confidence: float
) -> None:
    """Verify standalone scalar constants are scaled by 1/252, factor multipliers
    are untouched, and group averaging sums to 1.0 in both EP and BL.
    """
    tickers = ["AAPL", "MSFT", "GOOG", "TLT"]
    metadata = {"AAPL": "tech", "MSFT": "tech", "GOOG": "tech", "TLT": "bonds"}
    fees = {t: 0.001 for t in tickers}
    u = Universe(tickers=tickers, metadata=metadata, fees=fees)

    ps = PriorSynthesiser(u)
    sign_str = "+" if annual_spread >= 0 else "-"
    abs_spread = abs(annual_spread)

    # View 1: Relative view with RHS scalar constant: AAPL == TLT +/- spread
    view_rel = f"AAPL == TLT {sign_str} {abs_spread:.10g}"
    # View 2: Factor-weighted linear equality view: c1 * AAPL - c2 * TLT == spread
    view_lin = f"{c1:.10g} * AAPL - {c2:.10g} * TLT == {annual_spread:.10g}"
    # View 3: Group view with average_groups=True: tech == spread
    view_grp = f"tech == {annual_spread:.10g}"

    ps.add_market_view(view_rel, confidence=confidence)
    ps.add_market_view(view_lin, confidence=confidence)
    ps.add_market_view(view_grp, confidence=confidence, average_groups=True)

    ep = ps.build_entropy_prior()
    bl = ps.build_black_litterman_prior()

    groups_arr = np.array(
        [["AAPL", "MSFT", "GOOG", "TLT"], ["tech", "tech", "tech", "bonds"]],
        dtype=object,
    )

    # EntropyPooling uses sum_to_one=False in equations_to_matrix
    a_ep, b_ep, _, _ = equations_to_matrix(
        groups=groups_arr, equations=ep.mean_views, sum_to_one=False
    )
    # BlackLitterman uses sum_to_one=True in equations_to_matrix
    a_bl, b_bl, _, _ = equations_to_matrix(
        groups=groups_arr, equations=bl.views, sum_to_one=True
    )

    # 1. Standalone scalar constants are scaled by 1 / 252.0
    np.testing.assert_allclose(b_ep[0], annual_spread / 252.0, rtol=1e-4, atol=1e-7)
    np.testing.assert_allclose(b_bl[0], annual_spread / 252.0, rtol=1e-4, atol=1e-7)

    # 2. Factor multipliers c1 and -c2 are preserved untouched
    np.testing.assert_allclose(a_ep[1, 0], c1, rtol=1e-4, atol=1e-7)
    np.testing.assert_allclose(a_ep[1, 3], -c2, rtol=1e-4, atol=1e-7)
    np.testing.assert_allclose(
        b_ep[1], annual_spread / 252.0, rtol=1e-4, atol=1e-7
    )

    # 3. Group 'tech' (3 members) weights sum to 1.0 in BOTH EP and BL (never 1/N in BL)
    np.testing.assert_allclose(a_ep[2, :3], [1 / 3, 1 / 3, 1 / 3], rtol=1e-6)
    np.testing.assert_allclose(a_bl[2, :3], [1 / 3, 1 / 3, 1 / 3], rtol=1e-6)
    assert np.isclose(a_ep[2, :3].sum(), 1.0)
    assert np.isclose(a_bl[2, :3].sum(), 1.0)

    # 4. BlackLitterman rejects inequality views while EntropyPooling accepts them
    ps_ineq = PriorSynthesiser(u)
    ps_ineq.add_market_view(
        f"{c1:.4f} * AAPL - {c2:.4f} * TLT >= {annual_spread:.6f}", confidence=1.0
    )
    ep_ineq = ps_ineq.build_entropy_prior()
    assert len(ep_ineq.mean_views) == 1
    with pytest.raises(ValueError, match="BlackLitterman only supports linear equality"):
        ps_ineq.build_black_litterman_prior()


# ---------------------------------------------------------------------------
# 2. PortfolioDeltaEngine & ConstraintBuilder Algebraic Invariants
# ---------------------------------------------------------------------------


@st.composite
def simplex_weights_strategy(draw: st.DrawFn, n_assets: int = 4) -> list[float]:
    raw = draw(
        st.lists(
            st.floats(
                min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False
            ),
            min_size=n_assets,
            max_size=n_assets,
        ).filter(lambda xs: sum(xs) > 1e-3)
    )
    total = sum(raw)
    weights = [x / total for x in raw]
    # Adjust last weight to guarantee exact float sum near 1.0
    weights[-1] = max(0.0, 1.0 - sum(weights[:-1]))
    total2 = sum(weights)
    return [w / total2 for w in weights]


@settings(max_examples=50, deadline=None)
@given(
    w_curr_list=simplex_weights_strategy(4),
    w_targ_list=simplex_weights_strategy(4),
    fees_list=st.lists(
        st.floats(
            min_value=0.0, max_value=0.02, allow_nan=False, allow_infinity=False
        ),
        min_size=4,
        max_size=4,
    ),
    brokerage_bps=st.floats(
        min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False
    ),
    slippage_bps=st.floats(
        min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False
    ),
)
def test_property_delta_engine_zero_sum_drift_and_friction_identities(
    w_curr_list: list[float],
    w_targ_list: list[float],
    fees_list: list[float],
    brokerage_bps: float,
    slippage_bps: float,
) -> None:
    """Verify group drift sums to 0.0, allocations sum to 1.0, and net friction
    identities hold across arbitrary simplex portfolios and fee schedules.
    """
    tickers = ["A", "B", "C", "D"]
    metadata = {"A": "eq", "B": "eq", "C": "fi", "D": "alt"}
    fees = dict(zip(tickers, fees_list))
    u = Universe(tickers=tickers, metadata=metadata, fees=fees)

    w_curr = dict(zip(tickers, w_curr_list))
    w_targ = dict(zip(tickers, w_targ_list))

    engine = PortfolioDeltaEngine(
        universe=u, current_weights=w_curr, target_weights=w_targ
    )

    # 1. Group drift invariants
    drift_df = engine.calculate_group_drift()
    assert np.isclose(drift_df["current_allocation"].sum(), 1.0, atol=1e-9)
    assert np.isclose(drift_df["target_allocation"].sum(), 1.0, atol=1e-9)
    assert np.isclose(drift_df["drift"].sum(), 0.0, atol=1e-12)
    np.testing.assert_allclose(
        drift_df["abs_drift"].to_numpy(),
        np.abs(drift_df["drift"].to_numpy()),
        atol=1e-12,
    )

    # 2. Net friction accounting identities
    fric_df = engine.calculate_net_friction(
        brokerage_bps=brokerage_bps, slippage_bps=slippage_bps
    )
    asset_rows = fric_df[fric_df["ticker"] != "TOTAL"]
    total_row = fric_df[fric_df["ticker"] == "TOTAL"].iloc[0]

    # One-way turnover is half of gross weight traded and bounded in [0, 1]
    np.testing.assert_allclose(
        asset_rows["one_way_turnover"].to_numpy(),
        0.5 * asset_rows["gross_weight_traded"].to_numpy(),
        atol=1e-12,
    )
    assert -1e-12 <= float(total_row["one_way_turnover"]) <= 1.0 + 1e-9

    # Transaction cost identity
    expected_tc = (
        float(total_row["gross_weight_traded"])
        * (brokerage_bps + slippage_bps)
        / 10_000.0
    )
    assert np.isclose(float(total_row["transaction_cost"]), expected_tc, atol=1e-12)

    # Annual TER savings identity: sum((w_curr - w_targ) * fee)
    expected_ter_savings = sum(
        (w_curr[t] - w_targ[t]) * fees[t] for t in tickers
    )
    assert np.isclose(
        float(total_row["annual_ter_savings"]), expected_ter_savings, atol=1e-12
    )


@settings(max_examples=40, deadline=None)
@given(
    base_weights=simplex_weights_strategy(3),
    limit=st.floats(
        min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
)
def test_property_constraint_builder_turnover_and_daily_fee_conversion(
    base_weights: list[float], limit: float
) -> None:
    """Verify ConstraintBuilder.max_turnover and build() preserve simplex
    previous_weights and convert annual decimal TER fees to daily frequency (/ 252).
    """
    tickers = ["A", "B", "C"]
    fees = {"A": 0.00252, "B": 0.00504, "C": 0.0}
    u = Universe(
        tickers=tickers,
        metadata={"A": "g", "B": "g", "C": "g"},
        fees=fees,
    )
    weights_dict = dict(zip(tickers, base_weights))

    spec = ConstraintBuilder(u).max_turnover(limit, weights_dict).build()
    assert spec.previous_weights is not None
    assert np.isclose(sum(spec.previous_weights.values()), 1.0, atol=1e-12)
    assert all(0.0 <= v <= 1.0 for v in spec.previous_weights.values())
    assert spec.max_turnover == limit
    assert spec.management_fees is not None
    for t in tickers:
        assert np.isclose(spec.management_fees[t], fees[t] / 252.0, atol=1e-15)


# ---------------------------------------------------------------------------
# 3. Universe.anchor_history Inception & Holiday Gap Alignment Invariants
# ---------------------------------------------------------------------------


class _SyntheticProvider:
    def __init__(self, close_df: pd.DataFrame) -> None:
        self._close_df = close_df

    def fetch(
        self, tickers: list[str], start: str, end: str | None = None
    ) -> pd.DataFrame:
        arrays = [["Close"] * len(tickers), list(tickers)]
        cols = pd.MultiIndex.from_arrays(arrays)
        df = pd.DataFrame(
            self._close_df[tickers].to_numpy(),
            index=self._close_df.index,
            columns=cols,
        )
        return df


@settings(max_examples=40, deadline=None)
@given(
    late_start_offset=st.integers(min_value=1, max_value=15),
    gap_len=st.integers(min_value=1, max_value=8),
    max_ffill=st.integers(min_value=1, max_value=5),
)
def test_property_universe_anchor_history_truncate_and_gap_limits(
    late_start_offset: int, gap_len: int, max_ffill: int
) -> None:
    """Verify anchor_history() never forward-fills pre-inception NaNs and
    enforces max_ffill_days strictly on internal holiday gaps.
    """
    n_days = 40
    idx = pd.date_range("2024-01-01", periods=n_days, freq="B")
    prices = pd.DataFrame(
        {
            "EARLY": np.linspace(100.0, 120.0, n_days),
            "LATE": np.linspace(50.0, 60.0, n_days),
        },
        index=idx,
    )
    # Asset LATE has not IPO'd for the first `late_start_offset` business days
    prices.iloc[:late_start_offset, 1] = np.nan

    # Inject an internal holiday gap in EARLY after LATE's inception
    gap_start = late_start_offset + 5
    gap_end = min(n_days - 2, gap_start + gap_len)
    actual_gap = max(0, gap_end - gap_start)
    if actual_gap > 0:
        prices.iloc[gap_start:gap_end, 0] = np.nan

    u = Universe(
        tickers=["EARLY", "LATE"],
        metadata={"EARLY": "eq", "LATE": "eq"},
        fees={"EARLY": 0.0, "LATE": 0.0},
    )
    u.fetch_data(
        start="2024-01-01",
        end="2024-03-01",
        data_provider=_SyntheticProvider(prices),
        max_ffill_days=max_ffill,
    )

    if actual_gap > max_ffill:
        with pytest.raises(ValueError, match="contains NaN return observations"):
            u.anchor_history()
    else:
        u.anchor_history()
        # First return observation must be strictly after LATE's first valid price date
        assert u.returns.index[0] == idx[late_start_offset + 1]
        assert not u.returns.isna().any().any()
