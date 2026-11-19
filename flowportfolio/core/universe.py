"""Universe manager module.

This module provides the :class:`Universe` class, which is the foundational
data state manager for all flowportfolio components. It handles asset
validation, historical price retrieval via ``yfinance``, and return
computation via ``skfolio``'s canonical preprocessing pipeline.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from skfolio.preprocessing import prices_to_returns

from flowportfolio.core.data import DataProviderProtocol, YFinanceProvider
from flowportfolio.core.protocols import UniverseProtocol  # noqa: F401
from flowportfolio.core.validation import validate_fee

logger = logging.getLogger(__name__)


def _normalize_price_index(prices: pd.DataFrame) -> pd.DataFrame:
    """Ensure price DataFrame has non-null, unique, chronologically sorted timestamps."""
    if prices.index.isna().any():
        raise ValueError("Price data index contains missing (NaT) timestamps.")
    if prices.index.has_duplicates:
        dup_vals = prices.index[prices.index.duplicated()].unique().tolist()[:5]
        raise ValueError(
            f"Price data contains duplicate timestamps (examples: {dup_vals})."
        )
    if not prices.index.is_monotonic_increasing:
        logger.warning(
            "Price data index is not chronologically sorted; sorting index ascending "
            "before computing returns."
        )
        return prices.sort_index()
    return prices


def _validate_positive_prices(prices: pd.DataFrame) -> None:
    """Reject all-NaN columns, non-finite prices, or non-positive prices (ignoring NaNs)."""
    for col in prices.columns:
        valid = prices[col].dropna()
        if valid.empty:
            raise ValueError(f"Asset '{col}' has no valid price observations.")
        arr = valid.to_numpy(dtype=float, copy=False)
        if not np.isfinite(arr).all():
            bad_idx = valid.index[~np.isfinite(arr)][0]
            raise ValueError(f"Asset '{col}' contains non-finite price at {bad_idx}.")
        if (arr <= 0.0).any():
            bad_idx = valid.index[arr <= 0.0][0]
            bad_val = float(valid.loc[bad_idx])
            raise ValueError(
                f"Asset '{col}' contains non-positive price ({bad_val}) at {bad_idx}; "
                "all observed prices must be strictly positive."
            )


def _ffill_interior_gaps(
    prices: pd.DataFrame, max_ffill_days: int | None
) -> pd.DataFrame:
    """Forward-fill interior holiday gaps only, never extrapolating past last_valid_index."""
    if max_ffill_days == 0:
        return prices.copy()

    filled = (
        prices.ffill() if max_ffill_days is None else prices.ffill(limit=max_ffill_days)
    )

    # Restore trailing NaNs strictly after each column's genuine last_valid_index
    # so delisted or stale assets never receive fabricated 0.0 returns.
    for col in prices.columns:
        last_valid = prices[col].last_valid_index()
        if last_valid is not None:
            trailing_mask = filled.index > last_valid
            if trailing_mask.any():
                filled.loc[trailing_mask, col] = np.nan

    return filled


def _validate_weekday_exchange_calendar(
    raw_prices: pd.DataFrame, tickers: list[str]
) -> None:
    """Reject feeds containing weekend trading observations (out of scope for ETFs/equities)."""
    if not isinstance(raw_prices.index, pd.DatetimeIndex):
        return
    weekend_mask = raw_prices.index.dayofweek >= 5
    if not weekend_mask.any():
        return

    weekend_active = [
        t for t in tickers if raw_prices.loc[weekend_mask, t].notna().any()
    ]
    if weekend_active:
        raise ValueError(
            f"Unsupported weekend trading observations detected for tickers {weekend_active}. "
            "flowportfolio only supports weekday exchange calendars (ETFs and equities)."
        )


def _infer_periods_per_year(index: pd.Index) -> int:
    """Infer annualisation frequency for weekday exchange calendars (252, 52, 12, 4, 1)."""
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return 252

    deltas = index.to_series().diff().dropna().dt.total_seconds() / 86400.0
    if deltas.empty:
        return 252

    median_days = float(deltas.median())
    if median_days <= 3.5:
        return 252
    if 5.0 <= median_days <= 9.0:
        return 52
    if 25.0 <= median_days <= 35.0:
        return 12
    if 80.0 <= median_days <= 100.0:
        return 4
    if 350.0 <= median_days <= 380.0:
        return 1
    return 252


class DataFetchError(Exception):
    """Raised when the yfinance download process fails unexpectedly.

    This exception wraps any underlying network error, connection timeout,
    or API exception thrown during price data retrieval, providing a stable
    exception type for callers to handle.
    """


class Universe:
    """Asset universe manager.

    Manages a universe of assets by storing their ticker list, group
    metadata, and management fees. Provides methods to retrieve historical
    adjusted close prices from ``yfinance`` and to align return histories
    across all assets to prevent survivorship bias and NaN-induced
    optimisation failures during cross-validation.

    The intended usage lifecycle is::

        universe = Universe(tickers, metadata, fees)
        universe.fetch_data(start="2022-01-01")
        universe.anchor_history()
        # universe.returns is now aligned and ready for skfolio

    Parameters
    ----------
    tickers : list[str]
        Non-empty list of asset ticker symbols as accepted by ``yfinance``.
    metadata : dict[str, str]
        Dictionary mapping each ticker to a group tag string (e.g.,
        ``"core"``, ``"satellite"``). Every ticker must have an entry.
    fees : dict[str, float]
        Dictionary mapping each ticker to its annual management fee
        expressed as a decimal (e.g., ``0.0007`` for 7 bps). Every
        ticker must have an entry.

    Raises
    ------
    TypeError
        If ``tickers`` is not a ``list``, or if ``metadata`` or ``fees``
        are not ``dict`` instances with the correct value types.
    ValueError
        If ``tickers`` is empty, contains non-string elements, or if any
        ticker is missing from ``metadata`` or ``fees``.
    """

    def __init__(
        self,
        tickers: list[str],
        metadata: dict[str, str],
        fees: dict[str, float],
    ) -> None:
        # --- tickers validation ---
        if not isinstance(tickers, list):
            raise TypeError("tickers must be a list.")
        if len(tickers) == 0:
            raise ValueError("tickers must not be empty.")
        if not all(isinstance(t, str) for t in tickers):
            raise ValueError("All elements of tickers must be strings.")
        if len(tickers) != len(set(tickers)):
            raise ValueError("Duplicate tickers are not allowed.")

        # --- metadata validation ---
        if not isinstance(metadata, dict):
            raise TypeError("metadata must be a dict.")
        if not all(
            isinstance(k, str) and isinstance(v, str) for k, v in metadata.items()
        ):
            raise TypeError("metadata must map string keys to string values.")

        # --- fees validation ---
        if not isinstance(fees, dict):
            raise TypeError("fees must be a dict.")
        if not all(isinstance(k, str) for k in fees):
            raise TypeError("fees must map string keys to numeric values.")
        for v in fees.values():
            validate_fee(v)

        # --- coverage checks ---
        missing_metadata = set(tickers) - set(metadata.keys())
        if missing_metadata:
            raise ValueError(
                f"Missing metadata entries for tickers: {missing_metadata}"
            )

        missing_fees = set(tickers) - set(fees.keys())
        if missing_fees:
            raise ValueError(f"Missing fees entries for tickers: {missing_fees}")

        self._tickers: list[str] = list(tickers)
        self._metadata: dict[str, str] = dict(metadata)
        self._fees: dict[str, float] = dict(fees)
        # Raw adjusted close prices — stored so anchor_history is
        # non-destructive: fetch_data() can be re-called to reset state.
        self._prices: pd.DataFrame | None = None
        # Daily returns derived from _prices via prices_to_returns.
        # anchor_history() slices this in place.
        self._returns: pd.DataFrame | None = None
        # Last genuine (un-forward-filled) price date per ticker, recorded by
        # fetch_data() and consumed by anchor_history().
        self._raw_last_valid: dict[str, pd.Timestamp | None] = {}
        # Exchange return frequency (observations per calendar year); updated on fetch_data().
        self._periods_per_year: int = 252

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def tickers(self) -> list[str]:
        """List of asset tickers in their original order.

        Returns
        -------
        list[str]
            A copy of the ticker list (mutating the returned value has no
            effect on the Universe's internal state).
        """
        return list(self._tickers)

    @property
    def metadata(self) -> dict[str, str]:
        """Asset group metadata mapping.

        Returns
        -------
        dict[str, str]
            Dictionary mapping tickers to their group tags.
        """
        return dict(self._metadata)

    @property
    def fees(self) -> dict[str, float]:
        """Asset management fees mapping.

        Returns
        -------
        dict[str, float]
            Dictionary mapping tickers to their annual management fees.
        """
        return dict(self._fees)

    @property
    def periods_per_year(self) -> int:
        """Number of return observations per calendar year.

        Defaults to ``252`` before :meth:`fetch_data` is called, and is
        inferred from the weekday exchange price index afterwards
        (``252`` daily, ``52`` weekly, ``12`` monthly, ``4`` quarterly,
        ``1`` annual).

        Returns
        -------
        int
            Annualisation frequency for the fetched return series.
        """
        return self._periods_per_year

    @property
    def returns(self) -> pd.DataFrame:
        """Daily asset returns DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame of daily percentage returns with a ``DatetimeIndex``
            and one column per ticker. After ``anchor_history()`` is called,
            all leading NaN rows are removed and all columns are fully valid
            from the common start date onward.

        Raises
        ------
        ValueError
            If ``fetch_data()`` has not yet been called.
        """
        if self._returns is None:
            raise ValueError("Returns are not yet available. Call fetch_data() first.")
        return self._returns.copy()

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def fetch_data(
        self,
        start: str | None = None,
        end: str | None = None,
        data_provider: DataProviderProtocol | None = None,
        max_ffill_days: int | None = 5,
    ) -> None:
        """Download adjusted close prices and compute simple returns.

        Fetches adjusted close prices from the provided data provider for all
        tickers in the universe, validates weekday exchange calendar hygiene,
        computes simple returns using
        :func:`skfolio.preprocessing.prices_to_returns`, and infers
        :attr:`periods_per_year` from the price observation cadence. Both raw
        prices and returns are stored internally.

        Assets with shorter listing histories will have leading ``NaN``
        values in the returns DataFrame. Call :meth:`anchor_history`
        afterwards to remove those NaNs and align all assets to a common
        start date.

        Intermediate cross-exchange holiday gaps (``NaN`` prices occurring
        between an asset's inception and its last valid observation) are
        forward-filled, up to ``max_ffill_days`` consecutive observations,
        before returns are computed; leading pre-inception ``NaN`` values and
        trailing ``NaN`` values after an asset's last genuine price are left
        untouched. Unsorted unique timestamps are sorted chronologically with
        a warning. Only weekday exchange calendars (Monday–Friday) are
        supported; weekend-active feeds raise ``ValueError``, while empty
        all-``NaN`` weekend placeholder rows are stripped cleanly.

        Parameters
        ----------
        start : str, optional
            Start date for data retrieval in ``YYYY-MM-DD`` format.
            If ``None``, retrieves all available history for each ticker.
        end : str, optional
            End date for data retrieval in ``YYYY-MM-DD`` format.
            If ``None``, retrieves data up to and including the current date.
        data_provider : DataProviderProtocol, optional
            The data provider to fetch data. Defaults to YFinanceProvider.
        max_ffill_days : int or None, default 5
            Maximum number of consecutive missing price observations that are
            forward-filled across interior holidays or halts. ``None``
            forward-fills interior gaps without a limit; ``0`` disables
            forward-filling.

        Raises
        ------
        DataFetchError
            If the data provider raises any exception during the download.
        ValueError
            If no data is returned, if the downloaded data is missing the
            ``"Close"`` price level, if any requested ticker is absent from
            the downloaded data, if the price index contains missing
            (``NaT``) or duplicate timestamps, if any ticker has no valid
            prices or contains non-finite or non-positive prices, if any
            ticker contains weekend trading observations, if the resulting
            returns DataFrame is empty, or if ``max_ffill_days`` is not a
            non-negative integer or ``None``.
        """
        if max_ffill_days is not None and (
            not isinstance(max_ffill_days, int)
            or isinstance(max_ffill_days, bool)
            or max_ffill_days < 0
        ):
            raise ValueError("max_ffill_days must be a non-negative integer or None.")

        if data_provider is None:
            data_provider = YFinanceProvider()

        try:
            data = data_provider.fetch(
                tickers=self._tickers,
                start=start,
                end=end,
            )
        except ValueError:
            raise
        except Exception as exc:
            raise DataFetchError(
                "Failed to download price data. "
                "Check network connectivity and that all tickers are valid."
            ) from exc

        # yfinance returns a MultiIndex DataFrame (Attribute × Ticker) for
        # multiple tickers, and a flat-column DataFrame for a single ticker.
        if isinstance(data.columns, pd.MultiIndex):
            level_zero = data.columns.get_level_values(0)
            if "Close" not in level_zero:
                raise ValueError("Downloaded data has no 'Close' price level.")
            prices = data["Close"]
        else:
            if "Close" in data.columns:
                prices = pd.DataFrame({self._tickers[0]: data["Close"]})
            elif len(self._tickers) == 1 and self._tickers[0] in data.columns:
                prices = data[[self._tickers[0]]]
            else:
                raise ValueError(
                    "Downloaded data has no 'Close' column and no ticker-named column."
                )

        missing_tickers = set(self._tickers) - set(prices.columns)
        if missing_tickers:
            raise ValueError(f"Download did not return data for: {missing_tickers}")

        # Reorder columns to match the original tickers ordering.
        raw_prices = prices[self._tickers]
        raw_prices = _normalize_price_index(raw_prices)
        _validate_positive_prices(raw_prices)
        _validate_weekday_exchange_calendar(raw_prices, self._tickers)

        # If a vendor returned empty all-NaN weekend placeholder rows with no active
        # weekend tickers, drop those empty rows cleanly before computing returns.
        if isinstance(raw_prices.index, pd.DatetimeIndex):
            raw_prices = raw_prices.loc[raw_prices.index.dayofweek < 5]

        # Record each ticker's last genuine price date before any forward-filling so that
        # anchor_history() can detect histories that ended before the common start or end.
        raw_last_valid = {
            col: raw_prices[col].last_valid_index() for col in self._tickers
        }

        # Forward-fill intermediate cross-exchange holiday gaps only. Leading
        # pre-inception NaNs and trailing stale/delisted NaNs remain NaN.
        filled_prices = _ffill_interior_gaps(raw_prices, max_ffill_days=max_ffill_days)

        # Compute returns using skfolio's canonical preprocessing function.
        # This ensures the returns DataFrame layout is compatible with all
        # skfolio optimisers and cross-validation utilities.
        returns = prices_to_returns(
            filled_prices, drop_inceptions_nan=False, fill_nan=False
        )

        if returns.empty:
            raise ValueError(
                "Calculated returns DataFrame is empty after preprocessing."
            )

        inferred_ppy = _infer_periods_per_year(filled_prices.index)

        # Commit state atomically so a failed fetch_data() call never leaves
        # _raw_last_valid, _prices, _returns, and _periods_per_year out of sync.
        self._raw_last_valid = raw_last_valid
        self._prices = filled_prices
        self._returns = returns
        self._periods_per_year = inferred_ppy

    def anchor_history(self) -> None:
        """Align return histories to the asset with the shortest history.

        Truncates the returns DataFrame to start from the latest
        ``first_valid_index`` across all assets, removing all leading
        ``NaN`` rows. This aligns all assets to a common start date,
        ensuring date synchronisation and preventing NaN-related failures in
        ``skfolio`` cross-validation.

        Raises
        ------
        ValueError
            If ``fetch_data()`` has not been called yet, if any asset has
            no valid return observations whatsoever, if any asset has no
            valid returns after the alignment cut (fully disjoint trading
            histories), if any asset has missing trailing prices before the
            universe end date (possible delisting or stale feed), if any
            asset contains ``NaN`` returns after the alignment cut (a
            delisting or trading halt exceeding ``max_ffill_days``), or if
            any asset contains non-finite return observations.
        """
        if self._returns is None:
            raise ValueError("Returns are not available. Call fetch_data() first.")

        first_valid_map: dict[str, pd.Timestamp] = {}
        for col in self._returns.columns:
            fvi = self._returns[col].first_valid_index()
            if fvi is None:
                raise ValueError(f"Asset '{col}' has no valid return observations.")
            first_valid_map[col] = fvi

        bottleneck_ticker = max(first_valid_map, key=lambda k: first_valid_map[k])
        latest_start = first_valid_map[bottleneck_ticker]
        total_rows = len(self._returns)
        anchored = self._returns.loc[latest_start:]
        dropped_rows = total_rows - len(anchored)

        if total_rows > 0 and (dropped_rows / total_rows) > 0.20:
            logger.warning(
                "anchor_history() dropped %d of %d rows (%.1f%% of history) due to "
                "bottleneck ticker '%s' (first valid observation: %s).",
                dropped_rows,
                total_rows,
                (dropped_rows / total_rows) * 100.0,
                bottleneck_ticker,
                latest_start,
            )

        # Disjoint / delisted-asset guards. ``_raw_last_valid`` records each
        # ticker's last genuine (un-forward-filled) price date so that bounded
        # forward-filling cannot mask an asset whose history ended before the
        # common start date or before the universe end date.
        series_end = anchored.index[-1]
        for col in anchored.columns:
            raw_last = self._raw_last_valid.get(col)
            if raw_last is not None and raw_last < latest_start:
                raise ValueError(
                    f"Asset '{col}' has no valid returns after alignment. "
                    "Trading histories may be fully disjoint."
                )
            if raw_last is not None and raw_last < series_end:
                raise ValueError(
                    f"Asset '{col}' has missing trailing prices after {raw_last} "
                    f"(universe ends at {series_end}; possible delisting or stale feed)."
                )
            if anchored[col].first_valid_index() is None:
                raise ValueError(
                    f"Asset '{col}' has no valid returns after alignment. "
                    "Trading histories may be fully disjoint."
                )
            if anchored[col].isna().any():
                raise ValueError(
                    f"Asset '{col}' contains NaN return observations after "
                    f"{latest_start} (possible delisting or trading halt exceeding "
                    "max_ffill_days)."
                )
            if not np.isfinite(anchored[col].to_numpy(dtype=float, copy=False)).all():
                raise ValueError(
                    f"Asset '{col}' contains non-finite return observations after "
                    f"{latest_start}."
                )

        self._returns = anchored
