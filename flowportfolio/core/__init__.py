"""Core sub-package for flowportfolio.

Provides the foundational data management components used by all other
flowportfolio modules.
"""

from flowportfolio.core.constraints import ConstraintBuilder, ConstraintSpec
from flowportfolio.core.experiment import PortfolioExperimentEngine
from flowportfolio.core.protocols import UniverseProtocol, is_universe_protocol
from flowportfolio.core.reporting import Reporter
from flowportfolio.core.universe import DataFetchError, Universe

__all__ = [
    "ConstraintBuilder",
    "ConstraintSpec",
    "DataFetchError",
    "DataProviderProtocol",
    "PortfolioExperimentEngine",
    "Reporter",
    "Universe",
    "UniverseProtocol",
    "YFinanceProvider",
    "is_universe_protocol",
    "validate_fee",
    "validate_weight",
]

from flowportfolio.core.data import DataProviderProtocol, YFinanceProvider
from flowportfolio.core.validation import validate_fee, validate_weight
