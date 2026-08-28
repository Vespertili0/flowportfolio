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
    dates = pd.date_range(start, periods=n)
    data = {("Close", t): price_arrays[t] for t in tickers}
    df = pd.DataFrame(data, index=dates)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["Attributes", "Ticker"])
    return df


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
    dates = pd.date_range("2026-01-01", periods=5)
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
    dates = pd.date_range("2026-01-01", periods=3)
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

    The latest first_valid_index is therefore date 5 (2026-01-05), and
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
    assert univ.returns.index[0] == pd.Timestamp("2026-01-05")
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
