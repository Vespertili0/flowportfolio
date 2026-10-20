"""Universe manager module.

This module provides the :class:`Universe` class, which is the foundational
data state manager for all flowportfolio components. It handles asset
validation, historical price retrieval via ``yfinance``, and return
computation via ``skfolio``'s canonical preprocessing pipeline.
"""

from __future__ import annotations

import logging

import pandas as pd
from skfolio.preprocessing import prices_to_returns

from flowportfolio.core.data import DataProviderProtocol, YFinanceProvider
from flowportfolio.core.protocols import UniverseProtocol  # noqa: F401
from flowportfolio.core.validation import validate_fee

logger = logging.getLogger(__name__)


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
        """Download adjusted close prices and compute daily returns.

        Fetches adjusted close prices from the provided data provider for all
        tickers in the universe, then computes daily simple returns using
        :func:`skfolio.preprocessing.prices_to_returns`. Both raw prices
        and returns are stored internally.

        Assets with shorter listing histories will have leading ``NaN``
        values in the returns DataFrame. Call :meth:`anchor_history`
        afterwards to remove those NaNs and align all assets to a common
        start date.

        Intermediate cross-exchange holiday gaps (``NaN`` prices occurring
        after an asset's inception) are forward-filled, up to
        ``max_ffill_days`` consecutive observations, before returns are
        computed; leading pre-inception ``NaN`` values are left untouched.

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
            forward-filled across holidays or halts. ``None`` forward-fills
            without a limit; ``0`` disables forward-filling.

        Raises
        ------
        DataFetchError
            If the data provider raises any exception during the download.
        ValueError
            If no data is returned, if the downloaded data is missing the
            ``"Close"`` price level, if any requested ticker is absent from
            the downloaded data, if the resulting returns DataFrame is
            empty, or if ``max_ffill_days`` is not a non-negative integer or
            ``None``.
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

        # Reorder columns to match the original tickers ordering. Record each
        # ticker's last genuine price date before any forward-filling so that
        # anchor_history() can detect histories that ended before the common start.
        raw_prices = prices[self._tickers]
        self._raw_last_valid = {
            col: raw_prices[col].last_valid_index() for col in self._tickers
        }

        # Forward-fill intermediate cross-exchange holiday gaps. The fill is
        # bounded by ``max_ffill_days`` so that delisted or halted assets do not
        # accumulate fabricated 0.0 returns. Leading pre-inception NaNs remain NaN.
        if max_ffill_days is None:
            prices = raw_prices.ffill()
        elif max_ffill_days > 0:
            prices = raw_prices.ffill(limit=max_ffill_days)
        else:
            prices = raw_prices.copy()

        # Store raw prices. These are preserved across anchor_history calls
        # so that state can be fully reset by re-calling fetch_data().
        self._prices = prices

        # Compute returns using skfolio's canonical preprocessing function.
        # This ensures the returns DataFrame layout is compatible with all
        # skfolio optimisers and cross-validation utilities.
        returns = prices_to_returns(prices, drop_inceptions_nan=False, fill_nan=False)

        if returns.empty:
            raise ValueError(
                "Calculated returns DataFrame is empty after preprocessing."
            )

        self._returns = returns

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
            histories), or if any asset contains ``NaN`` returns after the
            alignment cut (a delisting or trading halt exceeding
            ``max_ffill_days``).
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
        # common start date.
        for col in anchored.columns:
            raw_last = self._raw_last_valid.get(col)
            if raw_last is not None and raw_last < latest_start:
                raise ValueError(
                    f"Asset '{col}' has no valid returns after alignment. "
                    "Trading histories may be fully disjoint."
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

        self._returns = anchored
