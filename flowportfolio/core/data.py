import time
from typing import Protocol, runtime_checkable

import pandas as pd
import yfinance as yf

_TRANSIENT_DOWNLOAD_ERRORS = (ConnectionError, TimeoutError, OSError)


class _TransientEmptyDownloadError(ValueError):
    """Internal marker for transient empty or all-NaN yfinance responses."""


@runtime_checkable
class DataProviderProtocol(Protocol):
    def fetch(
        self, tickers: list[str], start: str | None = None, end: str | None = None
    ) -> pd.DataFrame: ...


class YFinanceProvider:
    def __init__(self, retries: int = 3, backoff_base: float = 1.0) -> None:
        if isinstance(retries, bool) or not isinstance(retries, int) or retries < 1:
            raise ValueError("retries must be an integer >= 1.")
        if (
            isinstance(backoff_base, bool)
            or not isinstance(backoff_base, (int, float))
            or backoff_base < 0.0
        ):
            raise ValueError("backoff_base must be a non-negative number.")
        self._retries = retries
        self._backoff_base = float(backoff_base)

    def fetch(
        self, tickers: list[str], start: str | None = None, end: str | None = None
    ) -> pd.DataFrame:
        data: pd.DataFrame | None = None
        for attempt in range(self._retries):
            try:
                candidate = yf.download(
                    tickers=tickers,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                )
                if candidate is None or candidate.empty:
                    raise _TransientEmptyDownloadError(
                        "yfinance returned an empty DataFrame."
                    )

                # Validate the price slice only, so missing Volume/Open/High/Low data
                # (e.g. for indices) does not abort an otherwise valid download.
                target_slice = (
                    candidate["Close"]
                    if isinstance(candidate.columns, pd.MultiIndex)
                    and "Close" in candidate.columns.get_level_values(0)
                    else candidate
                )
                if target_slice.isna().all().any():
                    raise _TransientEmptyDownloadError(
                        "yfinance returned all-NaN columns for one or more tickers."
                    )
                data = candidate
                break
            except (*_TRANSIENT_DOWNLOAD_ERRORS, _TransientEmptyDownloadError):
                if attempt == self._retries - 1:
                    raise
                if self._backoff_base > 0.0:
                    time.sleep(self._backoff_base * (2**attempt))

        assert data is not None  # pragma: no cover
        return data
