"""Unit tests for the structural protocols.

This module verifies that the runtime-checkable protocols behave as expected
when checking structural subtyping.
"""

import pandas as pd

from flowportfolio.core.protocols import UniverseProtocol

# ---------------------------------------------------------------------------
# UniverseProtocol tests
# ---------------------------------------------------------------------------


def test_universe_protocol_valid_properties() -> None:
    """Test that a class with valid properties satisfies UniverseProtocol."""

    class ValidUniverseProperties:
        @property
        def tickers(self) -> list[str]:
            return ["A", "B"]

        @property
        def metadata(self) -> dict[str, str]:
            return {"A": "core", "B": "satellite"}

        @property
        def returns(self) -> pd.DataFrame:
            return pd.DataFrame()

        @property
        def fees(self) -> dict[str, float]:
            return {"A": 0.001, "B": 0.002}

        @property
        def periods_per_year(self) -> int:
            return 252

    instance = ValidUniverseProperties()
    assert isinstance(instance, UniverseProtocol)


def test_universe_protocol_valid_attributes() -> None:
    """Test that a class with valid instance attributes satisfies UniverseProtocol."""

    class ValidUniverseAttributes:
        def __init__(self) -> None:
            self.tickers = ["A", "B"]
            self.metadata = {"A": "core", "B": "satellite"}
            self.returns = pd.DataFrame()
            self.fees = {"A": 0.001, "B": 0.002}
            self.periods_per_year = 252

    instance = ValidUniverseAttributes()
    assert isinstance(instance, UniverseProtocol)


def test_universe_protocol_missing_tickers() -> None:
    """Test that a class missing the tickers attribute fails UniverseProtocol."""

    class MissingTickers:
        @property
        def metadata(self) -> dict[str, str]:
            return {"A": "core"}

    instance = MissingTickers()
    assert not isinstance(instance, UniverseProtocol)


def test_universe_protocol_missing_metadata() -> None:
    """Test that a class missing the metadata attribute fails UniverseProtocol."""

    class MissingMetadata:
        @property
        def tickers(self) -> list[str]:
            return ["A"]

    instance = MissingMetadata()
    assert not isinstance(instance, UniverseProtocol)


def test_universe_protocol_missing_periods_per_year() -> None:
    """Test that an object missing periods_per_year fails UniverseProtocol and is_universe_protocol."""
    from flowportfolio.core.protocols import is_universe_protocol

    class MissingPeriodsPerYear:
        def __init__(self) -> None:
            self.tickers = ["A"]
            self.metadata = {"A": "core"}
            self.fees = {"A": 0.001}
            self.returns = pd.DataFrame()

    instance = MissingPeriodsPerYear()
    assert not isinstance(instance, UniverseProtocol)
    assert is_universe_protocol(instance) is False


def test_universe_protocol_empty_class() -> None:
    """Test that an empty class fails UniverseProtocol."""

    class EmptyClass:
        pass

    instance = EmptyClass()
    assert not isinstance(instance, UniverseProtocol)


def test_validate_weight_and_fee_nan() -> None:
    """Test validate_weight and validate_fee reject NaN values."""
    import pytest

    from flowportfolio.core.validation import validate_fee, validate_weight

    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        validate_weight(float("nan"))

    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        validate_fee(float("nan"))


def test_is_universe_protocol_fees_dict_check() -> None:
    """Verify is_universe_protocol requires fees to be a dictionary."""
    from flowportfolio.core.protocols import is_universe_protocol

    class ValidObj:
        def __init__(self) -> None:
            self.tickers = ["A"]
            self.metadata = {"A": "core"}
            self.fees = {"A": 0.001}
            self.returns = pd.DataFrame()
            self.periods_per_year = 252

    assert is_universe_protocol(ValidObj()) is True

    class NonDictFees:
        def __init__(self) -> None:
            self.tickers = ["A"]
            self.metadata = {"A": "core"}
            self.fees = [0.001]
            self.returns = pd.DataFrame()
            self.periods_per_year = 252

    assert is_universe_protocol(NonDictFees()) is False


def test_is_universe_protocol_real_unfetched_universe_and_magic_mock() -> None:
    """Verify is_universe_protocol accepts unfetched Universe and MagicMock(spec=Universe) safely."""
    from unittest.mock import MagicMock

    from flowportfolio.core.protocols import is_universe_protocol
    from flowportfolio.core.universe import Universe

    unfetched = Universe(["SPY"], {"SPY": "core"}, {"SPY": 0.0003})
    assert is_universe_protocol(unfetched) is True

    mock_u = MagicMock(spec=Universe)
    mock_u.tickers = ["SPY"]
    mock_u.metadata = {"SPY": "core"}
    mock_u.fees = {"SPY": 0.0003}
    mock_u.returns = MagicMock()
    # periods_per_year is intentionally NOT assigned on mock_u (mirrors test_priors.py / test_delta.py)
    assert is_universe_protocol(mock_u) is True
