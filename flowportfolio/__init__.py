"""flowportfolio: A fluent, ETF-focused portfolio experimentation API built on top of skfolio."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _metadata_version

from flowportfolio.core.constraints import ConstraintBuilder
from flowportfolio.core.experiment import PortfolioExperimentEngine
from flowportfolio.core.protocols import UniverseProtocol
from flowportfolio.core.reporting import Reporter
from flowportfolio.core.universe import DataFetchError, Universe
from flowportfolio.delta import PortfolioDeltaEngine
from flowportfolio.persistence import PersistenceManager
from flowportfolio.priors import PriorSynthesiser
from flowportfolio.strategies import StrategyBuilder

try:
    __version__ = _metadata_version("flowportfolio")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0.dev0"

__all__ = [
    "ConstraintBuilder",
    "DataFetchError",
    "PersistenceManager",
    "PortfolioDeltaEngine",
    "PortfolioExperimentEngine",
    "PriorSynthesiser",
    "Reporter",
    "StrategyBuilder",
    "Universe",
    "UniverseProtocol",
    "__version__",
]
