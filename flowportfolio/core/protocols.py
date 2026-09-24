"""Structural protocol abstractions for flowportfolio.

This module defines :class:`UniverseProtocol`, a lightweight
``typing.Protocol`` that captures the minimal structural interface
required by constraint compilers. Any object exposing ``tickers``
and ``metadata`` properties satisfies this protocol, allowing
lightweight mocks and alternative data containers to be used
in place of the full :class:`~flowportfolio.core.universe.Universe`
class without incurring its heavy third-party dependencies.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


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
