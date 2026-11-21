import logging
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from skfolio.moments import EquilibriumMu, ShrunkCovariance
from skfolio.prior import BlackLitterman, EmpiricalPrior, EntropyPooling, SyntheticData
from skfolio.utils.equations import equations_to_matrix

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


def _o2_ladder_returns(
    n_obs: int = 60,
    ppy: int = 252,
    freq: str = "B",
    seed: int = 42,
) -> pd.DataFrame:
    """O2_ZERO_VOL_LADDER (+ tiny non-singular noise sigma=1e-4) for exact BL/EP recovery."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-02", periods=n_obs, freq=freq)
    rf_per_period = 0.0504 / ppy
    win_per_period = 0.1008 / ppy
    return pd.DataFrame(
        {
            "SPY": rng.normal(0.0, 1e-4, n_obs),
            "RF": rng.normal(rf_per_period, 1e-4, n_obs),
            "WIN": rng.normal(win_per_period, 1e-4, n_obs),
        },
        index=dates,
    )


def _o2_universe(n_obs: int = 60, ppy: int = 252, freq: str = "B") -> Universe:
    tickers = ["SPY", "RF", "WIN"]
    u = Universe(
        tickers=tickers,
        metadata={"SPY": "core", "RF": "cash", "WIN": "core"},
        fees={t: 0.0005 for t in tickers},
    )
    u._returns = _o2_ladder_returns(n_obs=n_obs, ppy=ppy, freq=freq)
    u._periods_per_year = ppy
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


def test_build_entropy_prior_success():
    u = _o2_universe()
    synthesiser = PriorSynthesiser(universe=u)
    synthesiser.add_market_view("WIN > 0.1008", confidence=1.0)

    prior = synthesiser.build_entropy_prior()

    assert isinstance(prior, EntropyPooling)
    assert prior.mean_views == [f"WIN >= {0.1008 / 252.0}"]
    assert prior.groups == {"SPY": ["core"], "RF": ["cash"], "WIN": ["core"]}

    prior.fit(u.returns)
    win_ann = float(prior.return_distribution_.mu[2]) * 252.0
    assert win_ann >= 0.1008 - 1e-6


def test_build_black_litterman_prior_success():
    u = _o2_universe()
    synthesiser = PriorSynthesiser(universe=u)
    synthesiser.add_market_view("WIN == 0.1008", confidence=1.0)

    prior = synthesiser.build_black_litterman_prior(tau=0.05)

    assert isinstance(prior, BlackLitterman)
    assert prior.views == [f"WIN == {0.1008 / 252.0}"]
    assert prior.tau == 0.05
    assert prior.view_confidences == [1.0]
    assert prior.groups == {"SPY": ["core"], "RF": ["cash"], "WIN": ["core"]}
    assert prior.risk_free_rate == 0.0

    prior.fit(u.returns)
    win_ann = float(prior.return_distribution_.mu[2]) * 252.0
    assert win_ann == pytest.approx(0.1008, abs=1e-4)


def test_build_entropy_prior_raises_runtime_error(stub_universe):
    synthesiser = PriorSynthesiser(universe=stub_universe)
    with pytest.raises(RuntimeError, match="No market views registered"):
        synthesiser.build_entropy_prior()


def test_build_synthetic_prior_success(stub_universe):
    synthesiser = PriorSynthesiser(universe=stub_universe)
    prior = synthesiser.build_synthetic_prior(n_samples=200, random_state=42)
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
    u = _o2_universe()

    synth = PriorSynthesiser(universe=u)
    synth.add_market_view("WIN-RF == 0.02", 0.8)
    assert len(synth._views) == 1

    synth_bl = PriorSynthesiser(universe=u)
    synth_bl.add_market_view("WIN >= 0.05", 0.8)
    with pytest.raises(
        ValueError, match="BlackLitterman only supports linear equality views"
    ):
        synth_bl.build_black_litterman_prior()

    synth_ep = PriorSynthesiser(universe=u)
    synth_ep.add_market_view("WIN >= SPY", 1.0)
    prior = synth_ep.build_entropy_prior()
    assert isinstance(prior, EntropyPooling)
    prior.fit(u.returns)
    mu_ann = prior.return_distribution_.mu * 252.0
    assert float(mu_ann[2]) >= float(mu_ann[0]) - 1e-6


def test_relative_view_with_rhs_spread_and_no_bl_double_division(stub_universe):
    """Verify RHS relative + spread views scale only the spread and BL avoids 1/N^2."""
    synth = PriorSynthesiser(universe=stub_universe)
    synth.add_market_view("AAPL == MSFT + 0.0252", confidence=0.8)
    assert synth._views[0]["view"] == f"AAPL == MSFT + {0.0252 / 252.0}"
    assert synth._views[0]["is_ranking"] is False

    synth_grp = PriorSynthesiser(universe=stub_universe)
    synth_grp.add_market_view("tech == 0.0504", confidence=0.9, average_groups=True)
    # EntropyPooling gets 0.5 * tech, whereas BlackLitterman gets unscaled 'tech'
    # because BlackLitterman already sets sum_to_one=True in equations_to_matrix.
    ep = synth_grp.build_entropy_prior()
    bl = synth_grp.build_black_litterman_prior()
    assert ep.mean_views == [f"0.5 * tech == {0.0504 / 252.0}"]
    assert bl.views == [f"tech == {0.0504 / 252.0}"]


def test_empirical_prior_bayes_stein_and_post_multiplied_views(stub_universe):
    """Verify Bayes-Stein shrinkage, LedoitWolf covariance, post-multiplied view parsing, and relative equality in BL."""
    import numpy as np
    import pandas as pd
    from skfolio.moments import LedoitWolf, ShrunkMu, ShrunkMuMethods

    # 1. Defaults
    synth = PriorSynthesiser(universe=stub_universe)
    emp_prior = synth.build_empirical_prior()
    assert isinstance(emp_prior.covariance_estimator, LedoitWolf)
    assert isinstance(emp_prior.mu_estimator, ShrunkMu)
    assert emp_prior.mu_estimator.method == ShrunkMuMethods.BAYES_STEIN

    # Fit on synthetic 2-asset returns and ensure beta_ in (0, 1]
    rng = np.random.default_rng(42)
    rets = pd.DataFrame(rng.normal(0.0005, 0.01, (100, 2)), columns=["AAPL", "MSFT"])
    emp_prior.fit(rets)
    assert 0.0 < emp_prior.mu_estimator_.beta_ <= 1.0
    assert 0.0 <= emp_prior.mu_estimator_.alpha_ < 1.0

    # 2. Post-multiplied view expression parsing
    synth.add_market_view("tech * 2.0 == 0.0504", confidence=0.8)
    # Default average_groups=True: tech has 2 members, so 2.0 * tech / 2 -> 1 * tech
    ep = synth.build_entropy_prior()
    assert ep.mean_views == [f"1 * tech == {0.0504 / 252.0}"]

    # 3. Relative equality in BlackLitterman (SPY == TLT)
    u_rel = MagicMock(spec=Universe)
    u_rel.tickers = ["AAPL", "MSFT"]
    u_rel.metadata = {"AAPL": "tech", "MSFT": "tech"}
    u_rel.fees = {"AAPL": 0.001, "MSFT": 0.001}
    u_rel.returns = MagicMock()
    synth_rel = PriorSynthesiser(universe=u_rel)
    synth_rel.add_market_view("AAPL == MSFT", confidence=0.7)
    bl = synth_rel.build_black_litterman_prior()
    assert bl.views == ["AAPL == MSFT"]


class _PeriodicUniverseStub:
    """Minimal UniverseProtocol-compliant stub exposing configurable periods_per_year."""

    def __init__(
        self,
        tickers: list[str],
        metadata: dict[str, str],
        fees: dict[str, float],
        periods_per_year: int,
    ) -> None:
        self.tickers = tickers
        self.metadata = metadata
        self.fees = fees
        self.returns = MagicMock()
        self.periods_per_year = periods_per_year


def test_leading_dot_group_coefficient_exact_matrix_no_silent_10x_corruption() -> None:
    """Leading-dot coefficients on multi-member groups must never corrupt strings or scale 10x."""
    # Case 1: N=2 members (".5 * tech" currently emits ".2.5 * tech" and crashes)
    u2 = Universe(
        tickers=["AAPL", "MSFT", "SPY"],
        metadata={"AAPL": "tech", "MSFT": "tech", "SPY": "broad"},
        fees={"AAPL": 0.001, "MSFT": 0.001, "SPY": 0.0003},
    )
    s2 = PriorSynthesiser(u2)
    s2.add_market_view(".5 * tech == 0.0504", confidence=0.8)

    groups_2 = np.array(
        [["AAPL", "MSFT", "SPY"], ["tech", "tech", "broad"]], dtype=object
    )
    a_ep, b_ep, _, _ = equations_to_matrix(
        groups=groups_2,
        equations=s2.build_entropy_prior().mean_views,
        sum_to_one=False,
    )
    a_bl, b_bl, _, _ = equations_to_matrix(
        groups=groups_2,
        equations=s2.build_black_litterman_prior().views,
        sum_to_one=True,
    )
    np.testing.assert_allclose(a_ep[0], [0.25, 0.25, 0.0], atol=1e-12)
    np.testing.assert_allclose(a_bl[0], [0.25, 0.25, 0.0], atol=1e-12)
    np.testing.assert_allclose(b_ep[0], 0.0504 / 252.0, atol=1e-12)
    np.testing.assert_allclose(b_bl[0], 0.0504 / 252.0, atol=1e-12)

    # Case 2: N=5 members (".25 * tech" currently silently emits ".5 * tech" -> 10x error!)
    tickers_5 = ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL"]
    u5 = Universe(
        tickers=tickers_5,
        metadata={t: "tech" for t in tickers_5},
        fees={t: 0.001 for t in tickers_5},
    )
    s5 = PriorSynthesiser(u5)
    s5.add_market_view(".25 * tech == 0.0504", confidence=0.8)

    groups_5 = np.array([tickers_5, ["tech"] * 5], dtype=object)
    a_ep5, b_ep5, _, _ = equations_to_matrix(
        groups=groups_5,
        equations=s5.build_entropy_prior().mean_views,
        sum_to_one=False,
    )
    a_bl5, b_bl5, _, _ = equations_to_matrix(
        groups=groups_5,
        equations=s5.build_black_litterman_prior().views,
        sum_to_one=True,
    )
    # 0.25 / 5 = 0.05 per member (NOT 0.5!)
    np.testing.assert_allclose(a_ep5[0], [0.05] * 5, atol=1e-12)
    np.testing.assert_allclose(a_bl5[0], [0.05] * 5, atol=1e-12)
    np.testing.assert_allclose(b_ep5[0], 0.0504 / 252.0, atol=1e-12)
    np.testing.assert_allclose(b_bl5[0], 0.0504 / 252.0, atol=1e-12)


@pytest.mark.parametrize(
    ("view_str", "expected_a", "expected_b"),
    [
        ("BRK-B-SPY == 0.0252", [1.0, -1.0, 0.0, 0.0, 0.0], 0.0252 / 252.0),
        ("SPY-BRK-B == 0.0252", [-1.0, 1.0, 0.0, 0.0, 0.0], 0.0252 / 252.0),
        ("-BRK-B == -0.0252", [-1.0, 0.0, 0.0, 0.0, 0.0], -0.0252 / 252.0),
        ("0.5*BRK-B-0.5*SPY == 0.0252", [0.5, -0.5, 0.0, 0.0, 0.0], 0.0252 / 252.0),
        ("VAS.AX-BRK-B == 0.0252", [-1.0, 0.0, 1.0, 0.0, 0.0], 0.0252 / 252.0),
        # Prefix-collision longest-match (both BRK-B and BRK exist in universe)
        ("BRK-B-BRK == 0.0252", [1.0, 0.0, 0.0, -1.0, 0.0], 0.0252 / 252.0),
        # Digit-prefixed exchange tickers (e.g. HSBC Hong Kong 0005.HK)
        ("0005.HK-SPY == 0.0252", [0.0, -1.0, 0.0, 0.0, 1.0], 0.0252 / 252.0),
        ("0.5*0005.HK == 0.0252", [0.0, 0.0, 0.0, 0.0, 0.5], 0.0252 / 252.0),
    ],
)
def test_hyphenated_and_dotted_equity_tickers_parse_without_spaces(
    view_str: str, expected_a: list[float], expected_b: float
) -> None:
    """Real equity/ETF tickers with hyphens, dots, shared prefixes, and leading digits parse cleanly."""
    tickers = ["BRK-B", "SPY", "VAS.AX", "BRK", "0005.HK"]
    metadata = {
        "BRK-B": "value",
        "SPY": "core",
        "VAS.AX": "intl",
        "BRK": "holding",
        "0005.HK": "hk",
    }
    u = Universe(
        tickers=tickers,
        metadata=metadata,
        fees={t: 0.0005 for t in tickers},
    )
    synth = PriorSynthesiser(u)
    synth.add_market_view(view_str, confidence=0.8)

    groups_arr = np.array([tickers, [metadata[t] for t in tickers]], dtype=object)
    a_ep, b_ep, _, _ = equations_to_matrix(
        groups=groups_arr,
        equations=synth.build_entropy_prior().mean_views,
        sum_to_one=False,
    )
    a_bl, b_bl, _, _ = equations_to_matrix(
        groups=groups_arr,
        equations=synth.build_black_litterman_prior().views,
        sum_to_one=True,
    )
    np.testing.assert_allclose(a_ep[0], expected_a, atol=1e-12)
    np.testing.assert_allclose(a_bl[0], expected_a, atol=1e-12)
    np.testing.assert_allclose(b_ep[0], expected_b, atol=1e-12)
    np.testing.assert_allclose(b_bl[0], expected_b, atol=1e-12)


@pytest.mark.parametrize(
    "bad_view",
    [
        "APPLE == 0.05",
        "BRK-C == 0.05",
        "SPY_2 == 0.05",
    ],
)
def test_prefix_boundary_rejects_unknown_identifiers(bad_view: str) -> None:
    """Valid tickers ('A', 'BRK-B', 'SPY') must never partially match unknown longer identifiers."""
    tickers = ["A", "BRK-B", "SPY"]
    u = Universe(
        tickers=tickers,
        metadata={"A": "health", "BRK-B": "value", "SPY": "core"},
        fees={t: 0.0005 for t in tickers},
    )
    synth = PriorSynthesiser(u)
    with pytest.raises(ValueError):
        synth.add_market_view(bad_view, confidence=0.8)


def test_rhs_group_post_multiply_leading_dot_and_average_groups_false() -> None:
    """RHS post-multiplied leading-dot group views and average_groups=False compile to exact matrices."""
    tickers = ["AAPL", "MSFT", "SPY"]
    metadata = {"AAPL": "tech", "MSFT": "tech", "SPY": "broad"}
    u = Universe(
        tickers=tickers,
        metadata=metadata,
        fees={"AAPL": 0.001, "MSFT": 0.001, "SPY": 0.0003},
    )
    groups_arr = np.array([tickers, ["tech", "tech", "broad"]], dtype=object)

    # 1. RHS group with post-multiplied leading dot + leading-dot spread, average_groups=True
    s_rhs = PriorSynthesiser(u)
    s_rhs.add_market_view(
        "SPY == tech * .5 + .0252", confidence=0.8, average_groups=True
    )
    a_ep, b_ep, _, _ = equations_to_matrix(
        groups=groups_arr,
        equations=s_rhs.build_entropy_prior().mean_views,
        sum_to_one=False,
    )
    a_bl, b_bl, _, _ = equations_to_matrix(
        groups=groups_arr,
        equations=s_rhs.build_black_litterman_prior().views,
        sum_to_one=True,
    )
    np.testing.assert_allclose(a_ep[0], [-0.25, -0.25, 1.0], atol=1e-12)
    np.testing.assert_allclose(a_bl[0], [-0.25, -0.25, 1.0], atol=1e-12)
    np.testing.assert_allclose(b_ep[0], 0.0252 / 252.0, atol=1e-12)
    np.testing.assert_allclose(b_bl[0], 0.0252 / 252.0, atol=1e-12)

    # 2. Leading-dot group view with average_groups=False: EP stays unaveraged (0.5 per member), BL sums to 0.5
    s_unavg = PriorSynthesiser(u)
    s_unavg.add_market_view(".5 * tech == 0.0504", confidence=0.8, average_groups=False)
    a_ep_u, b_ep_u, _, _ = equations_to_matrix(
        groups=groups_arr,
        equations=s_unavg.build_entropy_prior().mean_views,
        sum_to_one=False,
    )
    a_bl_u, b_bl_u, _, _ = equations_to_matrix(
        groups=groups_arr,
        equations=s_unavg.build_black_litterman_prior().views,
        sum_to_one=True,
    )
    np.testing.assert_allclose(a_ep_u[0], [0.5, 0.5, 0.0], atol=1e-12)
    np.testing.assert_allclose(a_bl_u[0], [0.25, 0.25, 0.0], atol=1e-12)
    np.testing.assert_allclose(b_ep_u[0], 0.0504 / 252.0, atol=1e-12)
    np.testing.assert_allclose(b_bl_u[0], 0.0504 / 252.0, atol=1e-12)


@pytest.mark.parametrize("ppy", [52, 12])
def test_market_view_scales_constants_by_universe_periods_per_year_and_mock_fallback(
    ppy: int,
) -> None:
    """Standalone scalar constants scale by 1/periods_per_year while mocks without periods_per_year default to 252."""
    tickers = ["AAPL", "MSFT"]
    metadata = {"AAPL": "tech", "MSFT": "tech"}
    fees = {"AAPL": 0.001, "MSFT": 0.001}
    groups_arr = np.array([tickers, ["tech", "tech"]], dtype=object)

    u_periodic = _PeriodicUniverseStub(
        tickers=tickers, metadata=metadata, fees=fees, periods_per_year=ppy
    )
    synth = PriorSynthesiser(u_periodic)
    synth.add_market_view("2.0 * AAPL - MSFT == 0.06", confidence=0.8)

    a_ep, b_ep, _, _ = equations_to_matrix(
        groups=groups_arr,
        equations=synth.build_entropy_prior().mean_views,
        sum_to_one=False,
    )
    a_bl, b_bl, _, _ = equations_to_matrix(
        groups=groups_arr,
        equations=synth.build_black_litterman_prior().views,
        sum_to_one=True,
    )
    np.testing.assert_allclose(a_ep[0], [2.0, -1.0], atol=1e-12)
    np.testing.assert_allclose(a_bl[0], [2.0, -1.0], atol=1e-12)
    np.testing.assert_allclose(b_ep[0], 0.06 / float(ppy), atol=1e-12)
    np.testing.assert_allclose(b_bl[0], 0.06 / float(ppy), atol=1e-12)

    # Legacy MagicMock(spec=Universe) without explicit periods_per_year must NOT resolve to 1.0
    u_mock = MagicMock(spec=Universe)
    u_mock.tickers = tickers
    u_mock.metadata = metadata
    u_mock.fees = fees
    u_mock.returns = MagicMock()
    synth_mock = PriorSynthesiser(u_mock)
    synth_mock.add_market_view("AAPL == 0.0504", confidence=0.8)
    _, b_mock_ep, _, _ = equations_to_matrix(
        groups=groups_arr,
        equations=synth_mock.build_entropy_prior().mean_views,
        sum_to_one=False,
    )
    _, b_mock_bl, _, _ = equations_to_matrix(
        groups=groups_arr,
        equations=synth_mock.build_black_litterman_prior().views,
        sum_to_one=True,
    )
    np.testing.assert_allclose(b_mock_ep[0], 0.0504 / 252.0, atol=1e-12)
    np.testing.assert_allclose(b_mock_bl[0], 0.0504 / 252.0, atol=1e-12)


def test_group_averaging_high_precision_and_typed_ir_stored() -> None:
    """Non-terminating group averages keep >=11 digits of precision and store typed IR terms."""
    tickers = ["AAPL", "MSFT", "GOOG"]
    metadata = {t: "tech" for t in tickers}
    u = Universe(tickers=tickers, metadata=metadata, fees={t: 0.001 for t in tickers})
    synth = PriorSynthesiser(u)
    synth.add_market_view("tech == 0.03", confidence=0.8)

    groups_arr = np.array([tickers, ["tech", "tech", "tech"]], dtype=object)
    a_ep, b_ep, _, _ = equations_to_matrix(
        groups=groups_arr,
        equations=synth.build_entropy_prior().mean_views,
        sum_to_one=False,
    )
    np.testing.assert_allclose(a_ep[0], [1.0 / 3.0] * 3, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(b_ep[0], 0.03 / 252.0, atol=1e-12)

    view_record = synth._views[0]
    assert "lhs_terms" in view_record and "rhs_terms" in view_record
    assert isinstance(view_record["lhs_terms"], tuple)
    assert isinstance(view_record["rhs_terms"], tuple)
    assert len(view_record["lhs_terms"]) == 1
    assert len(view_record["rhs_terms"]) == 1
    assert view_record["lhs_terms"][0].ident == "tech"
    assert view_record["lhs_terms"][0].coef == pytest.approx(1.0)
    assert view_record["rhs_terms"][0].ident is None
    assert view_record["rhs_terms"][0].coef == pytest.approx(0.03)


def test_parse_view_side_empty_valid_identifiers_fails_fast() -> None:
    """Direct calls to _parse_view_side with empty valid_identifiers must raise ValueError, not loop forever."""
    from flowportfolio.priors import _parse_view_side

    with pytest.raises(ValueError, match="valid linear view expression"):
        _parse_view_side(
            "AAPL + MSFT",
            set(),
            "AAPL + MSFT == 0.05",
            side_label="Left-hand side",
        )


def test_bl_absolute_view_does_not_double_count_risk_free_rate() -> None:
    """Default EquilibriumMu BL must recover exact total posterior return without 2x rfr."""
    u = _o2_universe()
    synth = PriorSynthesiser(u)
    synth.add_market_view("WIN == 0.1008", confidence=1.0)

    bl = synth.build_black_litterman_prior(risk_free_rate=0.0504)
    bl.fit(u.returns)

    win_idx = list(u.returns.columns).index("WIN")
    posterior_win_ann = float(bl.return_distribution_.mu[win_idx]) * 252.0
    assert posterior_win_ann == pytest.approx(0.1008, abs=1e-4)


def test_bl_relative_weighted_and_rhs_identifier_views_scale_rfr_by_net_loading() -> (
    None
):
    """Pure relative views (sum p_j = 0) are rfr-invariant; weighted and RHS-identifier views scale by sum p_j."""
    u = _o2_universe()

    # 1. Pure relative (net loading 0) + weighted linear (net loading 0.5)
    synth = PriorSynthesiser(u)
    synth.add_market_view("WIN - RF == 0.0252", confidence=1.0)
    synth.add_market_view("1.0 * WIN - 0.5 * SPY == 0.0756", confidence=1.0)

    bl = synth.build_black_litterman_prior(risk_free_rate=0.0504)
    bl.fit(u.returns)

    mu_ann = bl.return_distribution_.mu * 252.0
    spy_mu, rf_mu, win_mu = float(mu_ann[0]), float(mu_ann[1]), float(mu_ann[2])
    assert (win_mu - rf_mu) == pytest.approx(0.0252, abs=1e-4)
    assert (1.0 * win_mu - 0.5 * spy_mu) == pytest.approx(0.0756, abs=1e-4)

    # 2. RHS-identifier spread (net loading 0) + zero-constant RHS-identifier view (2*WIN == SPY -> net loading +1)
    synth_rhs = PriorSynthesiser(u)
    synth_rhs.add_market_view("WIN == RF + 0.0252", confidence=1.0)
    synth_rhs.add_market_view("2.0 * WIN == SPY", confidence=1.0)

    bl_rhs = synth_rhs.build_black_litterman_prior(risk_free_rate=0.0504)
    bl_rhs.fit(u.returns)

    mu_rhs = bl_rhs.return_distribution_.mu * 252.0
    assert (float(mu_rhs[2]) - float(mu_rhs[1])) == pytest.approx(0.0252, abs=1e-4)
    assert (2.0 * float(mu_rhs[2]) - float(mu_rhs[0])) == pytest.approx(0.0, abs=1e-4)


def test_bl_group_views_recover_total_returns_under_excess_and_empirical_priors() -> (
    None
):
    """Group views ('core == 0.1008') recover exact total group average under both EquilibriumMu and EmpiricalPrior."""
    u = _o2_universe()
    synth = PriorSynthesiser(u)
    synth.add_market_view("core == 0.1008", confidence=1.0)

    # Default EquilibriumMu (excess-return prior)
    bl_eq = synth.build_black_litterman_prior(risk_free_rate=0.0504)
    bl_eq.fit(u.returns)
    mu_eq = bl_eq.return_distribution_.mu * 252.0
    assert 0.5 * (float(mu_eq[0]) + float(mu_eq[2])) == pytest.approx(0.1008, abs=1e-4)

    # Total-return EmpiricalPrior
    emp = synth.build_empirical_prior()
    bl_emp = synth.build_black_litterman_prior(
        prior_estimator=emp, risk_free_rate=0.0504
    )
    bl_emp.fit(u.returns)
    mu_emp = bl_emp.return_distribution_.mu * 252.0
    assert 0.5 * (float(mu_emp[0]) + float(mu_emp[2])) == pytest.approx(
        0.1008, abs=1e-4
    )
    assert float(mu_emp[1]) == pytest.approx(0.0504, abs=1e-2)


def test_bl_with_empirical_total_return_prior_never_adds_rfr_twice(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Total-return priors suppress additive rfr (with warning), while explicit EquilibriumMu retains excess mode."""
    u = _o2_universe()
    synth = PriorSynthesiser(u)
    synth.add_market_view("WIN == 0.1008", confidence=1.0)

    # 1. EmpiricalPrior with ShrunkMu (build_empirical_prior)
    emp_shrunk = synth.build_empirical_prior()
    with caplog.at_level(logging.WARNING, logger="flowportfolio.priors"):
        bl = synth.build_black_litterman_prior(
            prior_estimator=emp_shrunk, risk_free_rate=0.0504
        )
    assert bl.risk_free_rate == 0.0
    assert any("double-counting" in rec.message for rec in caplog.records)

    bl.fit(u.returns)
    mu_ann = bl.return_distribution_.mu * 252.0
    assert float(mu_ann[2]) == pytest.approx(0.1008, abs=1e-4)
    assert float(mu_ann[1]) == pytest.approx(0.0504, abs=1e-2)
    assert float(mu_ann[0]) == pytest.approx(0.0, abs=1e-2)

    # 2. Default EmpiricalPrior() (mu_estimator=None -> EmpiricalMu total returns)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="flowportfolio.priors"):
        bl_default_emp = synth.build_black_litterman_prior(
            prior_estimator=EmpiricalPrior(), risk_free_rate=0.0504
        )
    assert bl_default_emp.risk_free_rate == 0.0
    assert any("double-counting" in rec.message for rec in caplog.records)
    bl_default_emp.fit(u.returns)
    mu_default_emp = bl_default_emp.return_distribution_.mu * 252.0
    assert float(mu_default_emp[2]) == pytest.approx(0.1008, abs=1e-4)
    assert float(mu_default_emp[0]) == pytest.approx(0.0, abs=1e-2)

    # 3. Explicit EmpiricalPrior(mu_estimator=EquilibriumMu()) must use excess-return mode
    caplog.clear()
    bl_explicit_eq = synth.build_black_litterman_prior(
        prior_estimator=EmpiricalPrior(mu_estimator=EquilibriumMu()),
        risk_free_rate=0.0504,
    )
    assert bl_explicit_eq.risk_free_rate == pytest.approx(0.0504 / 252.0, abs=1e-12)
    assert not any("double-counting" in rec.message for rec in caplog.records)
    bl_explicit_eq.fit(u.returns)
    assert float(bl_explicit_eq.return_distribution_.mu[2]) * 252.0 == pytest.approx(
        0.1008, abs=1e-4
    )


def test_bl_and_ep_respect_live_universe_periods_per_year() -> None:
    """Weekly universe (periods_per_year=52) scales views and rfr by 1/52 in both BL and EP, even after late fetch."""
    # 1. Views added on an already-weekly universe
    u_weekly = _o2_universe(n_obs=60, ppy=52, freq="W-FRI")
    synth = PriorSynthesiser(u_weekly)
    synth.add_market_view("WIN == 0.1040", confidence=1.0)

    bl = synth.build_black_litterman_prior(risk_free_rate=0.0520)
    bl.fit(u_weekly.returns)
    assert bl.risk_free_rate == pytest.approx(0.0520 / 52.0, abs=1e-12)
    assert float(bl.return_distribution_.mu[2]) * 52.0 == pytest.approx(
        0.1040, abs=1e-4
    )

    ep = synth.build_entropy_prior()
    assert ep.mean_views == [f"WIN == {0.1040 / 52.0}"]
    ep.fit(u_weekly.returns)
    assert float(ep.return_distribution_.mu[2]) * 52.0 == pytest.approx(
        0.1040, abs=1e-4
    )

    # 2. Views registered BEFORE universe periodicity updates (252 -> 52) re-format live at build time
    tickers = ["SPY", "RF", "WIN"]
    u_late = Universe(
        tickers=tickers,
        metadata={"SPY": "core", "RF": "cash", "WIN": "core"},
        fees={t: 0.0005 for t in tickers},
    )
    assert u_late.periods_per_year == 252
    synth_late = PriorSynthesiser(u_late)
    synth_late.add_market_view("WIN == 0.1040", confidence=1.0)
    synth_late.add_market_view("core == 0.0520", confidence=1.0, average_groups=False)

    # Simulate fetch_data() updating periodicity to weekly (52) after view registration
    u_late._returns = _o2_ladder_returns(n_obs=60, ppy=52, freq="W-FRI")
    u_late._periods_per_year = 52

    ep_late = synth_late.build_entropy_prior()
    assert ep_late.mean_views == [
        f"WIN == {0.1040 / 52.0}",
        f"core == {0.0520 / 52.0}",
    ]

    bl_late = synth_late.build_black_litterman_prior(risk_free_rate=0.0520)
    assert bl_late.risk_free_rate == pytest.approx(0.0520 / 52.0, abs=1e-12)
    bl_late.fit(u_late.returns)
    mu_late = bl_late.return_distribution_.mu * 52.0
    assert float(mu_late[2]) == pytest.approx(0.1040, abs=1e-4)
    assert 0.5 * (float(mu_late[0]) + float(mu_late[2])) == pytest.approx(
        0.0520, abs=1e-4
    )


def test_build_synthetic_prior_deterministic_validation_and_unseeded_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Identical random_state yields bit-for-bit identical samples; invalid seeds fail; None logs warning."""
    u = _o2_universe(n_obs=60)
    synth = PriorSynthesiser(u)

    p1 = synth.build_synthetic_prior(n_samples=150, random_state=42)
    p2 = synth.build_synthetic_prior(n_samples=150, random_state=42)
    p1.fit(u.returns)
    p2.fit(u.returns)
    np.testing.assert_allclose(
        p1.return_distribution_.returns,
        p2.return_distribution_.returns,
        atol=1e-12,
    )

    for bad_seed in (-1, True, "42", 1.5):
        with pytest.raises(ValueError, match="random_state"):
            synth.build_synthetic_prior(n_samples=150, random_state=bad_seed)  # type: ignore[arg-type]

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="flowportfolio.priors"):
        p_unseeded = synth.build_synthetic_prior(n_samples=150, random_state=None)
    assert isinstance(p_unseeded, SyntheticData)
    assert any("non-deterministic" in rec.message for rec in caplog.records)


def test_nested_prior_excess_return_detection_prevents_double_counting(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """P2-9a: _prior_uses_excess_returns recursively detects nested EquilibriumMu priors."""
    from skfolio.moments import EmpiricalMu, EquilibriumMu
    from skfolio.prior import EmpiricalPrior, EntropyPooling

    from flowportfolio.priors import _prior_uses_excess_returns

    nested_excess = EntropyPooling(
        prior_estimator=EmpiricalPrior(mu_estimator=EquilibriumMu())
    )
    nested_total = EntropyPooling(
        prior_estimator=EmpiricalPrior(mu_estimator=EmpiricalMu())
    )
    ep_default = EntropyPooling()
    nested_bl = BlackLitterman(
        prior_estimator=EmpiricalPrior(mu_estimator=EquilibriumMu()),
        views=["WIN == 0.0004"],
    )

    assert _prior_uses_excess_returns(None) is True
    assert _prior_uses_excess_returns(nested_excess) is True
    assert _prior_uses_excess_returns(nested_total) is False
    assert _prior_uses_excess_returns(ep_default) is False
    # Nested BlackLitterman already converts excess -> total via its own risk_free_rate
    assert _prior_uses_excess_returns(nested_bl) is False

    u = _o2_universe(n_obs=60)
    synth = PriorSynthesiser(u)
    synth.add_market_view("WIN == 0.1008", confidence=1.0)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="flowportfolio.priors"):
        bl = synth.build_black_litterman_prior(
            risk_free_rate=0.0504,
            prior_estimator=nested_excess,
        )
    assert bl.risk_free_rate == pytest.approx(0.0504 / 252.0, abs=1e-12)
    assert not any("double-counting" in rec.message for rec in caplog.records)


def test_black_litterman_preserves_economic_risk_free_rate_across_clone() -> None:
    """Sprint 5 (#6b): Economic risk_free_rate survives total-return suppression and sklearn clone."""
    from sklearn.base import clone

    u = _o2_universe(n_obs=60)
    synth = PriorSynthesiser(u)
    synth.add_market_view("WIN == 0.1008", confidence=0.8)

    bl_excess = synth.build_black_litterman_prior(risk_free_rate=0.0504)
    assert getattr(bl_excess, "portfolio_risk_free_rate", None) == pytest.approx(
        0.0504 / 252.0, abs=1e-12
    )
    bl_excess_cloned = clone(bl_excess)
    assert getattr(bl_excess_cloned, "portfolio_risk_free_rate", None) == pytest.approx(
        0.0504 / 252.0, abs=1e-12
    )

    bl_total = synth.build_black_litterman_prior(
        risk_free_rate=0.0504,
        prior_estimator=EmpiricalPrior(),
    )
    assert bl_total.risk_free_rate == 0.0
    assert getattr(bl_total, "portfolio_risk_free_rate", None) == pytest.approx(
        0.0504 / 252.0, abs=1e-12
    )
    bl_total_cloned = clone(bl_total)
    assert getattr(bl_total_cloned, "portfolio_risk_free_rate", None) == pytest.approx(
        0.0504 / 252.0, abs=1e-12
    )
