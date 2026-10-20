from unittest.mock import MagicMock, patch

import pytest
from skfolio.moments import ShrunkCovariance
from skfolio.prior import EmpiricalPrior, SyntheticData

from flowportfolio.core.universe import Universe
from flowportfolio.priors import PriorSynthesiser


class MockMu:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


@pytest.fixture
def stub_universe() -> Universe:
    # A simple mock object mimicking Universe
    u = MagicMock(spec=Universe)
    u.tickers = ["AAPL", "MSFT"]
    u.metadata = {"AAPL": "tech", "MSFT": "tech"}
    u.fees = {"AAPL": 0.001, "MSFT": 0.001}
    u.returns = MagicMock()
    return u


def test_init_raises_type_error():
    with pytest.raises(TypeError, match="universe must implement UniverseProtocol"):
        PriorSynthesiser(universe="not a universe")


def test_build_empirical_prior_with_default_estimators(stub_universe):
    synthesiser = PriorSynthesiser(universe=stub_universe)
    prior = synthesiser.build_empirical_prior()
    assert isinstance(prior, EmpiricalPrior)


def test_build_empirical_prior_with_custom_estimators(stub_universe):
    synthesiser = PriorSynthesiser(universe=stub_universe)
    cov = ShrunkCovariance()
    mu = MockMu()
    prior = synthesiser.build_empirical_prior(covariance_estimator=cov, mu_estimator=mu)
    assert isinstance(prior, EmpiricalPrior)
    assert prior.covariance_estimator is cov
    assert prior.mu_estimator is mu


def test_add_market_view_raises_value_error_for_invalid_confidence(stub_universe):
    synthesiser = PriorSynthesiser(universe=stub_universe)
    with pytest.raises(ValueError, match="confidence must be between 0.0 and 1.0"):
        synthesiser.add_market_view("AAPL > 0.05", confidence=1.5)


def test_add_market_view_raises_value_error_no_valid_tickers(stub_universe):
    synthesiser = PriorSynthesiser(universe=stub_universe)
    with pytest.raises(ValueError, match="does not reference any valid tickers"):
        synthesiser.add_market_view("X > 0.05", confidence=0.5)


def test_add_market_view_raises_value_error_unknown_tickers(stub_universe):
    synthesiser = PriorSynthesiser(universe=stub_universe)
    with pytest.raises(ValueError, match="not present in the universe"):
        synthesiser.add_market_view("AAPL > 0.05 and TSLA < 0.0", confidence=0.5)


@patch("flowportfolio.priors.EntropyPooling")
def test_build_entropy_prior_success(mock_entropy_pooling, stub_universe):
    # Setup mock to return a simple mock object to avoid skfolio constructor issues
    mock_instance = MagicMock()
    mock_entropy_pooling.return_value = mock_instance

    synthesiser = PriorSynthesiser(universe=stub_universe)
    synthesiser.add_market_view("AAPL > 0.05", confidence=0.8)

    prior = synthesiser.build_entropy_prior()

    # Assert
    assert prior is mock_instance
    mock_entropy_pooling.assert_called_once_with(
        mean_views=["AAPL >= 0.0001984126984126984"], groups=None
    )


@patch("flowportfolio.priors.BlackLitterman")
def test_build_black_litterman_prior_success(mock_bl, stub_universe):
    mock_instance = MagicMock()
    mock_bl.return_value = mock_instance

    synthesiser = PriorSynthesiser(universe=stub_universe)
    synthesiser.add_market_view("AAPL == 0.05", confidence=0.8)

    prior = synthesiser.build_black_litterman_prior(tau=0.05)

    assert prior is mock_instance
    mock_bl.assert_called_once_with(
        views=["AAPL == 0.0001984126984126984"],
        tau=0.05,
        view_confidences=[0.8],
        groups=None,
        risk_free_rate=0.0,
    )


def test_build_entropy_prior_raises_runtime_error(stub_universe):
    synthesiser = PriorSynthesiser(universe=stub_universe)
    with pytest.raises(RuntimeError, match="No market views registered"):
        synthesiser.build_entropy_prior()


def test_build_synthetic_prior_success(stub_universe):
    synthesiser = PriorSynthesiser(universe=stub_universe)
    prior = synthesiser.build_synthetic_prior(n_samples=200)
    assert isinstance(prior, SyntheticData)


def test_build_synthetic_prior_raises_value_error(stub_universe):
    synthesiser = PriorSynthesiser(universe=stub_universe)
    with pytest.raises(ValueError, match="n_samples must be at least 100"):
        synthesiser.build_synthetic_prior(n_samples=50)


def test_add_market_view_ticker_with_digits():
    u = MagicMock(spec=Universe)
    u.tickers = ["SP500", "TLT20"]
    u.metadata = {"SP500": "equity", "TLT20": "bonds"}
    u.fees = {"SP500": 0.0003, "TLT20": 0.0015}
    u.returns = MagicMock()
    synthesiser = PriorSynthesiser(universe=u)
    synthesiser.add_market_view("SP500 >= 0.05", confidence=0.8)
    assert synthesiser._views[0]["view"].startswith("SP500 >=")


def test_relative_view_without_spaces_and_bl_inequality_rejection():
    u = MagicMock(spec=Universe)
    u.tickers = ["SPY", "TLT"]
    u.metadata = {"SPY": "equity", "TLT": "bonds"}
    u.fees = {"SPY": 0.0003, "TLT": 0.0015}
    u.returns = MagicMock()

    synth = PriorSynthesiser(universe=u)
    synth.add_market_view("SPY-TLT == 0.02", 0.8)
    assert len(synth._views) == 1

    synth_bl = PriorSynthesiser(universe=u)
    synth_bl.add_market_view("SPY >= 0.05", 0.8)
    with pytest.raises(ValueError, match="BlackLitterman only supports linear equality views"):
        synth_bl.build_black_litterman_prior()

    synth_ep = PriorSynthesiser(universe=u)
    synth_ep.add_market_view("SPY >= TLT", 1.0)
    with patch("flowportfolio.priors.EntropyPooling") as mock_ep:
        mock_ep.return_value = MagicMock()
        prior = synth_ep.build_entropy_prior()
        assert prior is not None
