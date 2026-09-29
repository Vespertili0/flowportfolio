"""Unit tests for the structural protocols.

This module verifies that the runtime-checkable protocols behave as expected
when checking structural subtyping.
"""

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

    instance = ValidUniverseProperties()
    assert isinstance(instance, UniverseProtocol)


def test_universe_protocol_valid_attributes() -> None:
    """Test that a class with valid instance attributes satisfies UniverseProtocol."""

    class ValidUniverseAttributes:
        def __init__(self) -> None:
            self.tickers = ["A", "B"]
            self.metadata = {"A": "core", "B": "satellite"}

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


def test_universe_protocol_empty_class() -> None:
    """Test that an empty class fails UniverseProtocol."""

    class EmptyClass:
        pass

    instance = EmptyClass()
    assert not isinstance(instance, UniverseProtocol)
