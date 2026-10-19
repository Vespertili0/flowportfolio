"""Structural protocol abstractions for flowportfolio.

This module defines :class:`UniverseProtocol`, a lightweight
``typing.Protocol`` that captures the minimal structural interface
required by constraint compilers. Any object exposing ``tickers``
and ``metadata`` properties satisfies this protocol, allowing
lightweight mocks and alternative data containers to be used
in place of the full :class:`~flowportfolio.core.universe.Universe`
class without incurring its heavy third-party dependencies.

Because ``isinstance`` checks against a ``@runtime_checkable`` protocol
invoke property getters (and ``Universe.returns`` raises before
``fetch_data()`` is called), :func:`is_universe_protocol` is provided as a
non-evaluating structural validator.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class UniverseProtocol(Protocol):
    """Minimal structural interface required by constraint builders.

    Any object that exposes a ``tickers`` property returning a
    ``list[str]`` and a ``metadata`` property returning a
    ``dict[str, str]`` satisfies this protocol.

    This protocol is ``@runtime_checkable``, so ``isinstance`` checks
    work at runtime (structural attribute presence is verified, not
    value types).
    """

    @property
    def tickers(self) -> list[str]:
        """List of asset ticker symbols."""
        ...

    @property
    def metadata(self) -> dict[str, str]:
        """Mapping of ticker symbol to group tag string."""
        ...

    @property
    def returns(self) -> pd.DataFrame:
        """Daily asset returns DataFrame."""
        ...

    @property
    def fees(self) -> dict[str, float]:
        """Asset management fees mapping."""
        ...


def is_universe_protocol(obj: object) -> bool:
    """Verify structural conformance to UniverseProtocol without invoking lazy properties."""
    for attr in ("tickers", "metadata", "fees"):
        if not hasattr(obj, attr):
            return False
    # Check descriptor on class or instance dict without triggering Universe.returns ValueError
    has_returns = (
        hasattr(type(obj), "returns")
        or "returns" in getattr(obj, "__dict__", {})
        or hasattr(obj, "_returns")
    )
    if not has_returns:
        return False
    return isinstance(getattr(obj, "tickers", None), list) and isinstance(
        getattr(obj, "metadata", None), dict
    )
