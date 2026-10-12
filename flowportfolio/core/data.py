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
        for attempt in range(retries):
            try:
                data = yf.download(
                    tickers=tickers,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                )
                if data.empty:
                    raise ValueError("yfinance returned an empty DataFrame.")

                # Check for all-NaN columns immediately
                if data.isna().all().any():
                    raise ValueError(
                        "yfinance returned all-NaN columns for one or more tickers."
                    )

                return data
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(2**attempt)
        raise RuntimeError("Failed to fetch data")
