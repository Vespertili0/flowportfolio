"""Unit tests for the Universe class.

This module contains comprehensive unit tests verifying the initialisation,
validation, data retrieval, returns calculation, and anchoring behaviour
of the Universe manager in :mod:`flowportfolio.core.universe`.

All tests mock ``yfinance.download`` so that the suite runs fully offline
without any network dependency.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from flowportfolio.core.universe import DataFetchError, Universe

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

TICKERS = ["SPY", "IVV"]
METADATA = {"SPY": "core", "IVV": "core"}
FEES = {"SPY": 0.0004, "IVV": 0.0004}


def _make_multiindex_prices(
    tickers: list[str],
    price_arrays: dict[str, list[float]],
    start: str = "2026-01-01",
) -> pd.DataFrame:
    """Build a yfinance-style MultiIndex price DataFrame for mocking.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols that appear as the second level of the MultiIndex.
    price_arrays : dict[str, list[float]]
        Mapping of ticker → list of close prices. NaN is accepted.
    start : str
        ISO date string for the first row of the DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        DataFrame with a ``(Attribute, Ticker)`` MultiIndex on columns and
        a ``DatetimeIndex`` as the row index, matching yfinance output.
    """
    n = len(next(iter(price_arrays.values())))
    dates = pd.bdate_range(start, periods=n)
    data = {("Close", t): price_arrays[t] for t in tickers}
    df = pd.DataFrame(data, index=dates)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["Attributes", "Ticker"])
    return df


class _InMemoryProvider:
    """Minimal DataProviderProtocol stub for deterministic hygiene tests."""

    def __init__(self, close_df: pd.DataFrame) -> None:
        self._close_df = close_df

    def fetch(
        self,
        tickers: list[str],
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        return pd.concat({"Close": self._close_df[tickers]}, axis=1)


# ---------------------------------------------------------------------------
# 1. Initialisation & validation
# ---------------------------------------------------------------------------


def test_universe_initialisation_valid() -> None:
    """Test that a valid set of arguments initialises without error."""
    univ = Universe(TICKERS, METADATA, FEES)
    assert univ.metadata == METADATA
    assert univ.fees == FEES
    assert univ.tickers == TICKERS


def test_universe_tickers_property_returns_copy() -> None:
    """Mutating the returned tickers list must not affect the Universe state."""
    univ = Universe(TICKERS, METADATA, FEES)
    returned = univ.tickers
    returned.append("XYZ")
    assert univ.tickers == TICKERS


def test_universe_init_tickers_not_a_list() -> None:
    """Passing a tuple for tickers must raise TypeError."""
    with pytest.raises(TypeError):
        Universe(("SPY", "IVV"), METADATA, FEES)  # type: ignore[arg-type]


def test_universe_init_tickers_empty() -> None:
    """Passing an empty tickers list must raise ValueError."""
    with pytest.raises(ValueError):
        Universe([], METADATA, FEES)


def test_universe_init_tickers_non_string_element() -> None:
    """Tickers containing non-string elements must raise ValueError."""
    with pytest.raises(ValueError):
        Universe(["SPY", 123], METADATA, FEES)  # type: ignore[list-item]


def test_universe_init_metadata_not_dict() -> None:
    """Passing a non-dict for metadata must raise TypeError."""
    with pytest.raises(TypeError):
        Universe(TICKERS, ["core", "core"], FEES)  # type: ignore[arg-type]


def test_universe_init_metadata_non_string_value() -> None:
    """Metadata with non-string values must raise TypeError."""
    with pytest.raises(TypeError):
        Universe(TICKERS, {"SPY": 1.0, "IVV": "core"}, FEES)  # type: ignore[dict-item]


def test_universe_init_fees_not_dict() -> None:
    """Passing a non-dict for fees must raise TypeError."""
    with pytest.raises(TypeError):
        Universe(TICKERS, METADATA, [0.0004, 0.0004])  # type: ignore[arg-type]


def test_universe_init_fees_non_numeric_value() -> None:
    """Fees with non-numeric values must raise TypeError."""
    with pytest.raises(TypeError):
        Universe(TICKERS, METADATA, {"SPY": "free", "IVV": 0.0004})  # type: ignore[dict-item]


def test_universe_init_missing_metadata_key() -> None:
    """Tickers without a metadata entry must raise ValueError."""
    with pytest.raises(ValueError):
        Universe(TICKERS, {"SPY": "core"}, FEES)


def test_universe_init_missing_fees_key() -> None:
    """Tickers without a fees entry must raise ValueError."""
    with pytest.raises(ValueError):
        Universe(TICKERS, METADATA, {"SPY": 0.0004})


# ---------------------------------------------------------------------------
# 2. fetch_data — success paths
# ---------------------------------------------------------------------------


@patch("yfinance.download")
def test_universe_fetch_data_multiindex(mock_download) -> None:
    """Test fetch_data with a standard multi-ticker MultiIndex response."""
    mock_download.return_value = _make_multiindex_prices(
        TICKERS,
        {
            "SPY": [100.0, 101.0, 102.0, 103.0, 104.0],
            "IVV": [200.0, 202.0, 204.0, 206.0, 208.0],
        },
    )

    univ = Universe(TICKERS, METADATA, FEES)

    # Accessing returns before fetch_data must raise ValueError.
    with pytest.raises(ValueError):
        _ = univ.returns

    univ.fetch_data(start="2026-01-01")

    mock_download.assert_called_once_with(
        tickers=TICKERS,
        start="2026-01-01",
        end=None,
        auto_adjust=True,
        progress=False,
    )

    # skfolio prices_to_returns drops the first price row, yielding 4 returns
    # from 5 prices.
    assert len(univ.returns) == 4
    assert list(univ.returns.columns) == TICKERS

    # Verify SPY return values match simple pct_change arithmetic.
    expected_spy = [
        0.01,
        102.0 / 101.0 - 1.0,
        103.0 / 102.0 - 1.0,
        104.0 / 103.0 - 1.0,
    ]
    np.testing.assert_allclose(univ.returns["SPY"], expected_spy, rtol=1e-5)


@patch("yfinance.download")
def test_universe_fetch_data_single_ticker_flat_columns(mock_download) -> None:
    """Test fetch_data with a flat-column response (single ticker, older yfinance)."""
    dates = pd.bdate_range("2026-01-01", periods=5)
    mock_download.return_value = pd.DataFrame(
        {
            "Open": [99.0, 100.0, 101.0, 102.0, 103.0],
            "Close": [100.0, 101.0, 102.0, 103.0, 104.0],
        },
        index=dates,
    )

    univ = Universe(["SPY"], {"SPY": "core"}, {"SPY": 0.0004})
    univ.fetch_data(start="2026-01-01")

    assert len(univ.returns) == 4
    assert list(univ.returns.columns) == ["SPY"]
    np.testing.assert_allclose(
        univ.returns["SPY"],
        [0.01, 102.0 / 101.0 - 1.0, 103.0 / 102.0 - 1.0, 104.0 / 103.0 - 1.0],
        rtol=1e-5,
    )


# ---------------------------------------------------------------------------
# 3. fetch_data — failure paths
# ---------------------------------------------------------------------------


@patch("yfinance.download")
def test_universe_fetch_data_empty_response(mock_download) -> None:
    """An empty DataFrame from yfinance must raise ValueError."""
    mock_download.return_value = pd.DataFrame()
    univ = Universe(TICKERS, METADATA, FEES)
    with pytest.raises(ValueError):
        univ.fetch_data()


@patch("yfinance.download")
def test_universe_fetch_data_no_close_column(mock_download) -> None:
    """Missing 'Close' level in a MultiIndex response must raise ValueError."""
    dates = pd.bdate_range("2026-01-01", periods=3)
    mock_df = pd.DataFrame(
        {
            ("Open", "SPY"): [100.0, 101.0, 102.0],
            ("Open", "IVV"): [200.0, 202.0, 204.0],
        },
        index=dates,
    )
    mock_df.columns = pd.MultiIndex.from_tuples(mock_df.columns)
    mock_download.return_value = mock_df

    univ = Universe(TICKERS, METADATA, FEES)
    with pytest.raises(ValueError):
        univ.fetch_data()


@patch("yfinance.download")
def test_universe_fetch_data_missing_ticker(mock_download) -> None:
    """A response missing one of the requested tickers must raise ValueError."""
    mock_download.return_value = _make_multiindex_prices(
        ["SPY"],
        {"SPY": [100.0, 101.0, 102.0]},
    )
    univ = Universe(TICKERS, METADATA, FEES)
    with pytest.raises(ValueError):
        univ.fetch_data()


@patch("yfinance.download")
def test_universe_fetch_data_download_exception(mock_download) -> None:
    """An exception inside yfinance.download must be wrapped as DataFetchError."""
    mock_download.side_effect = Exception("Connection timeout")
    univ = Universe(TICKERS, METADATA, FEES)
    with pytest.raises(DataFetchError):
        univ.fetch_data()


# ---------------------------------------------------------------------------
# 4. anchor_history — success paths
# ---------------------------------------------------------------------------


@patch("yfinance.download")
def test_universe_anchor_history_aligns_to_latest_start(mock_download) -> None:
    """anchor_history must slice returns to the latest first-valid-index.

    Design
    ------
    SPY prices are valid from the first date.
    IVV prices are NaN for the first three dates, becoming valid at date 4.

    After prices_to_returns (drops first price row), IVV returns are NaN for
    the first three return rows (dates 2–4) because pct_change on NaN prices
    propagates NaN:

        IVV return at date 2 = NaN / NaN = NaN
        IVV return at date 3 = NaN / NaN = NaN
        IVV return at date 4 = 200 / NaN  = NaN  ← transition row
        IVV return at date 5 = 198 / 200 - 1 = valid

    The latest first_valid_index is therefore date 5 (2026-01-07), and
    anchor_history slices from there, leaving 3 rows.
    """
    mock_download.return_value = _make_multiindex_prices(
        TICKERS,
        {
            "SPY": [100.0, 101.0, 103.02, 104.05, 103.01, 105.07, 107.17],
            "IVV": [np.nan, np.nan, np.nan, 200.0, 198.0, 201.0, 203.0],
        },
    )

    univ = Universe(TICKERS, METADATA, FEES)
    # anchor_history before fetch_data must raise ValueError.
    with pytest.raises(ValueError):
        univ.anchor_history()

    univ.fetch_data(start="2026-01-01")
    univ.anchor_history()

    assert len(univ.returns) == 3
    assert univ.returns.index[0] == pd.Timestamp("2026-01-07")
    assert not univ.returns.isna().any().any()


@patch("yfinance.download")
def test_universe_anchor_history_equal_length_histories(mock_download) -> None:
    """anchor_history on a fully valid returns DataFrame must be a no-op."""
    mock_download.return_value = _make_multiindex_prices(
        TICKERS,
        {
            "SPY": [100.0, 101.0, 102.0, 103.0, 104.0],
            "IVV": [200.0, 202.0, 204.0, 206.0, 208.0],
        },
    )
    univ = Universe(TICKERS, METADATA, FEES)
    univ.fetch_data()
    original_len = len(univ.returns)
    univ.anchor_history()
    # No rows should be removed when histories are already aligned.
    assert len(univ.returns) == original_len


# ---------------------------------------------------------------------------
# 5. anchor_history — failure paths
# ---------------------------------------------------------------------------


@patch("yfinance.download")
def test_universe_anchor_history_all_nan_column(mock_download) -> None:
    """An asset with entirely NaN return history must raise ValueError."""
    mock_download.return_value = _make_multiindex_prices(
        TICKERS,
        {
            "SPY": [100.0, 101.0, 102.0, 103.0, 104.0],
            "IVV": [np.nan, np.nan, np.nan, np.nan, np.nan],
        },
    )
    univ = Universe(TICKERS, METADATA, FEES)
    with pytest.raises(ValueError):
        univ.fetch_data()


@patch("yfinance.download")
def test_anchor_history_rejects_disjoint_and_delisted_assets(mock_download) -> None:
    """Verify disjoint trading histories and halts exceeding max_ffill_days raise ValueError."""
    # 1. Disjoint histories: A trades days 1-50 (NaN 51-100), B trades days 60-100 (NaN 1-59)
    prices_a = [100.0 + i for i in range(50)] + [np.nan] * 50
    prices_b = [np.nan] * 59 + [200.0 + i for i in range(41)]

    mock_download.return_value = _make_multiindex_prices(
        ["SPY", "IVV"],
        {"SPY": prices_a, "IVV": prices_b},
        start="2026-01-01",
    )
    univ = Universe(
        ["SPY", "IVV"], {"SPY": "core", "IVV": "core"}, {"SPY": 0.0, "IVV": 0.0}
    )
    univ.fetch_data(max_ffill_days=5)
    with pytest.raises(ValueError, match="no valid returns after alignment"):
        univ.anchor_history()

    # 2. Halt exceeding max_ffill_days: 10-day halt in the middle
    halt_spy = [100.0] * 20 + [np.nan] * 10 + [105.0] * 20
    halt_ivv = [200.0] * 50
    mock_download.return_value = _make_multiindex_prices(
        ["SPY", "IVV"],
        {"SPY": halt_spy, "IVV": halt_ivv},
        start="2026-01-01",
    )
    univ2 = Universe(
        ["SPY", "IVV"], {"SPY": "core", "IVV": "core"}, {"SPY": 0.0, "IVV": 0.0}
    )
    univ2.fetch_data(max_ffill_days=5)
    with pytest.raises(ValueError, match="contains NaN return observations after"):
        univ2.anchor_history()


# ---------------------------------------------------------------------------
# 6. Weekday exchange calendar boundary & periodicity contract (Cycle 2)
# ---------------------------------------------------------------------------


def test_universe_rejects_mixed_weekday_and_weekend_calendar() -> None:
    """Out-of-scope 7-day feeds mixed with weekday ETFs must raise ValueError (never fabricate 3-day Monday bars)."""
    dates = pd.date_range("2026-01-05", periods=14, freq="D")
    prices = pd.DataFrame(
        {
            "SPY": [
                400.0 + i if d.dayofweek < 5 else np.nan for i, d in enumerate(dates)
            ],
            "BTC-USD": [100.0 * (1.02**i) for i in range(14)],
        },
        index=dates,
    )
    univ = Universe(
        ["SPY", "BTC-USD"],
        {"SPY": "equity", "BTC-USD": "other"},
        {"SPY": 0.0003, "BTC-USD": 0.001},
    )
    with pytest.raises(ValueError, match="weekend|weekday exchange"):
        univ.fetch_data(data_provider=_InMemoryProvider(prices))


def test_universe_rejects_pure_weekend_trading_feed() -> None:
    """Pure 7-day weekend-active feeds are outside ETF/equity scope and must raise ValueError."""
    dates = pd.date_range("2026-01-05", periods=14, freq="D")
    prices = pd.DataFrame(
        {"BTC-USD": [100.0 * (1.02**i) for i in range(14)]},
        index=dates,
    )
    univ = Universe(["BTC-USD"], {"BTC-USD": "other"}, {"BTC-USD": 0.001})

    with pytest.raises(ValueError, match="weekend|weekday exchange"):
        univ.fetch_data(data_provider=_InMemoryProvider(prices))


def test_universe_strips_all_nan_weekend_placeholder_rows() -> None:
    """Vendor-padded all-NaN Sat/Sun rows (including trailing weekends) are stripped cleanly for weekday ETFs."""
    # 14 calendar days: Mon Jan 5 .. Sun Jan 18 (ends on a weekend with all-NaN rows)
    dates = pd.date_range("2026-01-05", periods=14, freq="D")
    prices = pd.DataFrame(
        {
            "SPY": [
                400.0 + i if d.dayofweek < 5 else np.nan for i, d in enumerate(dates)
            ],
            "IVV": [
                200.0 + i if d.dayofweek < 5 else np.nan for i, d in enumerate(dates)
            ],
        },
        index=dates,
    )
    univ = Universe(
        ["SPY", "IVV"],
        {"SPY": "core", "IVV": "core"},
        {"SPY": 0.0003, "IVV": 0.0003},
    )
    univ.fetch_data(data_provider=_InMemoryProvider(prices))
    univ.anchor_history()

    assert univ._prices is not None
    assert len(univ._prices) == 10
    assert (univ._prices.index.dayofweek < 5).all()
    assert len(univ.returns) == 9
    assert (univ.returns.index.dayofweek < 5).all()
    assert univ.periods_per_year == 252


@pytest.mark.parametrize(
    ("dates", "expected_ppy"),
    [
        (pd.bdate_range("2026-01-05", periods=20), 252),
        (pd.date_range("2026-01-09", periods=10, freq="W-FRI"), 52),
        (pd.date_range("2025-01-31", periods=10, freq="BME"), 12),
        (pd.date_range("2024-03-29", periods=8, freq="BQE"), 4),
        (pd.date_range("2020-12-31", periods=5, freq="BYE"), 1),
        # 2-bar monthly series yields 1 return row; inference must still detect monthly (12)
        (pd.date_range("2025-01-31", periods=2, freq="BME"), 12),
    ],
)
def test_universe_infers_exchange_periods_per_year(
    dates: pd.DatetimeIndex, expected_ppy: int
) -> None:
    """Universe exposes periods_per_year matching standard ETF/equity exchange frequencies."""
    prices = pd.DataFrame(
        {"SPY": np.linspace(100.0, 120.0, len(dates))},
        index=dates,
    )
    univ = Universe(["SPY"], {"SPY": "core"}, {"SPY": 0.0003})
    assert univ.periods_per_year == 252  # Default before fetch_data()

    univ.fetch_data(data_provider=_InMemoryProvider(prices))
    assert univ.periods_per_year == expected_ppy


def test_universe_failed_refetch_preserves_periods_per_year() -> None:
    """A failed fetch_data() call must preserve the previously inferred periods_per_year."""
    weekly_dates = pd.date_range("2026-01-09", periods=10, freq="W-FRI")
    weekly_prices = pd.DataFrame(
        {"SPY": np.linspace(100.0, 110.0, len(weekly_dates))},
        index=weekly_dates,
    )
    univ = Universe(["SPY"], {"SPY": "core"}, {"SPY": 0.0003})
    univ.fetch_data(data_provider=_InMemoryProvider(weekly_prices))
    assert univ.periods_per_year == 52

    # Rejected weekend-active re-fetch must leave periods_per_year untouched
    weekend_dates = pd.date_range("2026-01-05", periods=7, freq="D")
    weekend_prices = pd.DataFrame(
        {"SPY": np.linspace(100.0, 106.0, len(weekend_dates))},
        index=weekend_dates,
    )
    with pytest.raises(ValueError, match="weekend|weekday exchange"):
        univ.fetch_data(data_provider=_InMemoryProvider(weekend_prices))
    assert univ.periods_per_year == 52


# ---------------------------------------------------------------------------
# 7. O4_PRICE_HYGIENE (Cycle 1: delisting, price validity & index ordering)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("max_ffill_days", [5, None])
def test_o4a_terminal_delisting_within_max_ffill_never_fabricates_zero_returns(
    max_ffill_days: int | None,
) -> None:
    """Trailing NaNs within max_ffill_days (or unbounded fill) must NOT emit 0.0 returns and must fail anchor_history."""
    dates = pd.bdate_range("2026-01-01", periods=10)
    prices = pd.DataFrame(
        {
            "SPY": np.linspace(100.0, 109.0, 10),
            # DEL stops trading 3 days before the end (< max_ffill_days=5)
            "DEL": [50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, np.nan, np.nan, np.nan],
        },
        index=dates,
    )
    univ = Universe(
        ["SPY", "DEL"], {"SPY": "core", "DEL": "sat"}, {"SPY": 0.0, "DEL": 0.0}
    )
    univ.fetch_data(
        data_provider=_InMemoryProvider(prices), max_ffill_days=max_ffill_days
    )

    # Layer 1: fetch_data must never forward-fill past last_valid_index into 0.0 returns
    assert univ.returns["DEL"].iloc[-3:].isna().all()

    # Layer 2: anchor_history must reject the stale/delisted tail loudly
    with pytest.raises(ValueError, match="delisting|trailing|stale"):
        univ.anchor_history()


def test_o4a_interior_holiday_gap_still_forward_fills() -> None:
    """Interior holiday gaps within max_ffill_days must remain forward-filled when trailing prices are valid."""
    dates = pd.bdate_range("2026-01-01", periods=8)
    prices = pd.DataFrame(
        {
            "SPY": np.linspace(100.0, 107.0, 8),
            "HOL": [50.0, 51.0, np.nan, np.nan, 54.0, 55.0, 56.0, 57.0],
        },
        index=dates,
    )
    univ = Universe(
        ["SPY", "HOL"], {"SPY": "core", "HOL": "sat"}, {"SPY": 0.0, "HOL": 0.0}
    )
    univ.fetch_data(data_provider=_InMemoryProvider(prices), max_ffill_days=5)
    univ.anchor_history()

    assert len(univ.returns) == 7
    assert not univ.returns.isna().any().any()


@pytest.mark.parametrize("bad_price", [0.0, -5.0, np.inf, -np.inf])
def test_o4b_non_positive_or_non_finite_price_raises_value_error(
    bad_price: float,
) -> None:
    """Prices <= 0 or non-finite must raise ValueError instead of leaking inf or sign-flipped returns."""
    dates = pd.bdate_range("2026-01-01", periods=5)
    prices = pd.DataFrame(
        {
            "SPY": [100.0, 101.0, bad_price, 102.0, 103.0],
            "IVV": [200.0, 202.0, 204.0, 206.0, 208.0],
        },
        index=dates,
    )
    univ = Universe(
        ["SPY", "IVV"], {"SPY": "core", "IVV": "core"}, {"SPY": 0.0, "IVV": 0.0}
    )

    with pytest.raises(ValueError, match="positive|finite"):
        univ.fetch_data(data_provider=_InMemoryProvider(prices))


def test_o4b_all_nan_column_via_custom_provider_raises_value_error() -> None:
    """All-NaN columns from a custom DataProviderProtocol must fail fast instead of being silently dropped."""
    dates = pd.bdate_range("2026-01-01", periods=5)
    prices = pd.DataFrame(
        {
            "SPY": [100.0, 101.0, 102.0, 103.0, 104.0],
            "IVV": [np.nan, np.nan, np.nan, np.nan, np.nan],
        },
        index=dates,
    )
    univ = Universe(
        ["SPY", "IVV"], {"SPY": "core", "IVV": "core"}, {"SPY": 0.0, "IVV": 0.0}
    )

    with pytest.raises(ValueError, match="no valid price"):
        univ.fetch_data(data_provider=_InMemoryProvider(prices))


def test_o4c_unsorted_3row_timestamps_sort_chronologically_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unsorted 3-row unique timestamps must warn, sort ascending, and yield exact chronological returns."""
    # Mon 2026-01-05, Wed 2026-01-07, Tue 2026-01-06
    unsorted_dates = pd.DatetimeIndex(["2026-01-05", "2026-01-07", "2026-01-06"])
    prices = pd.DataFrame(
        {
            "SPY": [100.0, 120.0, 110.0],
            "IVV": [200.0, 240.0, 220.0],
        },
        index=unsorted_dates,
    )
    univ = Universe(
        ["SPY", "IVV"], {"SPY": "core", "IVV": "core"}, {"SPY": 0.0, "IVV": 0.0}
    )
    with caplog.at_level("WARNING"):
        univ.fetch_data(data_provider=_InMemoryProvider(prices))
    univ.anchor_history()

    assert "not chronologically sorted" in caplog.text
    assert univ.returns.index.is_monotonic_increasing
    np.testing.assert_allclose(
        univ.returns["SPY"].to_numpy(), [0.10, 10.0 / 110.0], rtol=1e-12
    )


def test_o4c_unsorted_4row_timestamps_prevent_silent_reverse_returns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unsorted 4-row timestamps must sort ascending instead of silently emitting reverse-time returns."""
    # Mon 2026-01-05, Wed 2026-01-07, Tue 2026-01-06, Thu 2026-01-08
    unsorted_dates = pd.DatetimeIndex(
        ["2026-01-05", "2026-01-07", "2026-01-06", "2026-01-08"]
    )
    prices = pd.DataFrame(
        {
            "SPY": [100.0, 120.0, 110.0, 132.0],
            "IVV": [200.0, 240.0, 220.0, 264.0],
        },
        index=unsorted_dates,
    )
    univ = Universe(
        ["SPY", "IVV"], {"SPY": "core", "IVV": "core"}, {"SPY": 0.0, "IVV": 0.0}
    )
    with caplog.at_level("WARNING"):
        univ.fetch_data(data_provider=_InMemoryProvider(prices))
    univ.anchor_history()

    assert "not chronologically sorted" in caplog.text
    assert univ.returns.index.is_monotonic_increasing
    # Sorted prices: [100 -> 110 -> 120 -> 132] => [0.10, 10/110, 0.10]
    np.testing.assert_allclose(
        univ.returns["SPY"].to_numpy(), [0.10, 10.0 / 110.0, 0.10], rtol=1e-12
    )


def test_o4c_duplicate_timestamps_raise_value_error() -> None:
    """Duplicate timestamps must fail fast rather than silently picking or distorting a bar."""
    dup_dates = pd.DatetimeIndex(["2026-01-05", "2026-01-06", "2026-01-06"])
    dup_prices = pd.DataFrame({"SPY": [100.0, 110.0, 112.0]}, index=dup_dates)
    univ_dup = Universe(["SPY"], {"SPY": "core"}, {"SPY": 0.0})

    with pytest.raises(ValueError, match="duplicate"):
        univ_dup.fetch_data(data_provider=_InMemoryProvider(dup_prices))


def test_o4c_nat_timestamp_in_index_raises_value_error() -> None:
    """Missing (NaT) timestamps in the price index must fail fast with ValueError."""
    nat_dates = pd.DatetimeIndex(["2026-01-05", pd.NaT, "2026-01-07"])
    nat_prices = pd.DataFrame({"SPY": [100.0, 110.0, 120.0]}, index=nat_dates)
    univ_nat = Universe(["SPY"], {"SPY": "core"}, {"SPY": 0.0})

    with pytest.raises(ValueError, match="missing|NaT"):
        univ_nat.fetch_data(data_provider=_InMemoryProvider(nat_prices))


def test_o4_failed_refetch_preserves_prior_universe_state() -> None:
    """A failed fetch_data() call (e.g., empty returns from 1-row prices) must not partially mutate prior state."""
    dates = pd.bdate_range("2026-01-01", periods=5)
    valid_prices = pd.DataFrame(
        {
            "SPY": [100.0, 101.0, 102.0, 103.0, 104.0],
            "IVV": [200.0, 202.0, 204.0, 206.0, 208.0],
        },
        index=dates,
    )
    univ = Universe(
        ["SPY", "IVV"], {"SPY": "core", "IVV": "core"}, {"SPY": 0.0, "IVV": 0.0}
    )
    univ.fetch_data(data_provider=_InMemoryProvider(valid_prices))
    prior_returns = univ.returns.copy()

    # Single-row price DataFrame produces empty returns and must fail atomically
    one_row_prices = valid_prices.iloc[:1]
    with pytest.raises(ValueError, match="empty"):
        univ.fetch_data(data_provider=_InMemoryProvider(one_row_prices))

    pd.testing.assert_frame_equal(univ.returns, prior_returns)
    assert univ._prices is not None
    assert len(univ._prices) == 5
    # Must still anchor cleanly without stale _raw_last_valid poisoning
    univ.anchor_history()
    assert len(univ.returns) == 4
