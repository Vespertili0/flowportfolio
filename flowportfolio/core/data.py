import time
from typing import Protocol, runtime_checkable

import pandas as pd
import yfinance as yf


@runtime_checkable
class DataProviderProtocol(Protocol):
    def fetch(
        self, tickers: list[str], start: str | None = None, end: str | None = None
    ) -> pd.DataFrame: ...


class YFinanceProvider:
    def fetch(
        self, tickers: list[str], start: str | None = None, end: str | None = None
    ) -> pd.DataFrame:
        retries = 3
        data: pd.DataFrame | None = None
        # Only the network call is retried; deterministic validation failures
        # below fail fast without sleeping.
        for attempt in range(retries):
            try:
                data = yf.download(
                    tickers=tickers,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                )
                break
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(2**attempt)

        if data is None or data.empty:
            raise ValueError("yfinance returned an empty DataFrame.")

        # Validate the price slice only, so missing Volume/Open/High/Low data
        # (e.g. for indices) does not abort an otherwise valid download.
        target_slice = (
            data["Close"]
            if isinstance(data.columns, pd.MultiIndex)
            and "Close" in data.columns.get_level_values(0)
            else data
        )
        if target_slice.isna().all().any():
            raise ValueError(
                "yfinance returned all-NaN columns for one or more tickers."
            )

        return data
