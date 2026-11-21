"""Unit and live skfolio oracle tests for flowportfolio.core.optimisation (Cycle 7)."""

from __future__ import annotations

import inspect
import math
import warnings

import numpy as np
import pandas as pd
import pytest
from skfolio.measures import RatioMeasure
from skfolio.metrics import make_scorer
from skfolio.model_selection import WalkForward
from skfolio.optimization import MeanRisk, ObjectiveFunction
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline

from flowportfolio.core import optimisation as opt_module
from flowportfolio.core.optimisation import (
    SkfolioGridSearchCV,
    set_previous_weights_chain,
)


def _sample_returns(n_obs: int = 90, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "A": rng.normal(0.0002, 0.015, n_obs),
            "B": rng.normal(0.0005, 0.010, n_obs),
            "C": rng.normal(0.0012, 0.008, n_obs),
        },
        index=pd.bdate_range("2025-01-02", periods=n_obs),
    )


class _DropFirstColSelector(BaseEstimator, TransformerMixin):
    """Simple pre-selector that drops column 0 while returning a DataFrame."""

    def fit(self, X: pd.DataFrame, y=None):
        self.cols_ = list(X.columns[1:])
        self.mask_ = np.array([False] + [True] * (X.shape[1] - 1), dtype=bool)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.cols_]

    def get_support(self, indices: bool = False) -> np.ndarray:
        return np.where(self.mask_)[0] if indices else self.mask_


def test_grid_search_cv_uses_consistent_previous_weights_in_inner_cv_and_refit_without_mutating_estimator() -> (
    None
):
    """Sprint 5 (#1): Inner CV isolates future previous_weights; final refit enforces caller previous_weights without mutating estimator."""
    X = _sample_returns(n_obs=90, seed=7)
    concentrated_pw = {"A": 0.90, "B": 0.05, "C": 0.05}
    neutral_eq = {"A": 1.0 / 3.0, "B": 1.0 / 3.0, "C": 1.0 / 3.0}
    observed_pw: list[object] = []

    class _RecordingMeanRisk(MeanRisk):
        def fit(self, X, y=None, **fit_params):
            observed_pw.append(
                dict(self.previous_weights)
                if isinstance(self.previous_weights, dict)
                else self.previous_weights
            )
            return super().fit(X, y=y, **fit_params)

    # 1. With max_turnover active: inner folds run against equal-weight baseline (no future leakage),
    # while final refit (5th call) receives concentrated_pw.
    orig_estimator = _RecordingMeanRisk(max_turnover=0.05)
    assert orig_estimator.previous_weights is None

    gscv = SkfolioGridSearchCV(
        estimator=orig_estimator,
        param_grid={"l2_coef": [0.0, 0.1]},
        cv=WalkForward(train_size=40, test_size=25),
        scoring=make_scorer(RatioMeasure.CVAR_RATIO),
        previous_weights=concentrated_pw,
    )
    gscv.fit(X)

    # 2 candidates x 2 inner folds = 4 inner fits + 1 final refit = 5 fits
    assert len(observed_pw) == 5
    for inner_pw in observed_pw[:-1]:
        assert isinstance(inner_pw, dict)
        for col, val in neutral_eq.items():
            assert inner_pw[col] == pytest.approx(val, abs=1e-12)
    assert observed_pw[-1] == concentrated_pw
    # Final fitted weights honor concentrated_pw max_turnover bound
    assert abs(float(gscv.weights_[0]) - 0.90) <= 0.05 + 1e-6

    # Caller's estimator hyperparameter must remain unmutated
    assert orig_estimator.previous_weights is None
    assert gscv.estimator is orig_estimator
    assert gscv.estimator.previous_weights is None

    # 2. Without max_turnover (transaction_costs only): inner folds run with previous_weights=None
    observed_tc_pw: list[object] = []

    class _RecordingTCMeanRisk(MeanRisk):
        def fit(self, X, y=None, **fit_params):
            observed_tc_pw.append(
                dict(self.previous_weights)
                if isinstance(self.previous_weights, dict)
                else self.previous_weights
            )
            return super().fit(X, y=y, **fit_params)

    gscv_tc = SkfolioGridSearchCV(
        estimator=_RecordingTCMeanRisk(transaction_costs=0.001),
        param_grid={"l2_coef": [0.0, 0.1]},
        cv=WalkForward(train_size=40, test_size=25),
        scoring=make_scorer(RatioMeasure.CVAR_RATIO),
        previous_weights=concentrated_pw,
    )
    gscv_tc.fit(X)
    assert len(observed_tc_pw) == 5
    assert all(pw is None for pw in observed_tc_pw[:-1])
    assert observed_tc_pw[-1] == concentrated_pw


def test_grid_search_cv_scores_match_concentrated_turnover_regime_oracle() -> None:
    """Sprint 5 (#1): Inner CV scores match neutral-baseline oracle while final refit enforces concentrated_pw."""
    X = _sample_returns(n_obs=100, seed=11)
    concentrated_pw = {"A": 0.90, "B": 0.05, "C": 0.05}
    neutral_eq = {col: 1.0 / 3.0 for col in X.columns}
    cv = WalkForward(train_size=50, test_size=25)
    scorer = make_scorer(RatioMeasure.CVAR_RATIO)
    l2_grid = [0.0, 0.05, 0.5]

    gscv = SkfolioGridSearchCV(
        estimator=MeanRisk(
            objective_function=ObjectiveFunction.MAXIMIZE_RETURN,
            max_turnover=0.05,
        ),
        param_grid={"l2_coef": l2_grid},
        cv=cv,
        scoring=scorer,
        previous_weights=concentrated_pw,
    )
    gscv.fit(X)

    # Analytical oracle: inner folds evaluate under leak-free neutral baseline weights
    expected_means: list[float] = []
    for l2 in l2_grid:
        fold_scores = []
        for train_idx, test_idx in cv.split(X):
            m = MeanRisk(
                objective_function=ObjectiveFunction.MAXIMIZE_RETURN,
                max_turnover=0.05,
                l2_coef=l2,
                previous_weights=dict(neutral_eq),
            )
            m.fit(X.iloc[train_idx])
            fold_scores.append(float(scorer(m, X.iloc[test_idx])))
        expected_means.append(float(np.mean(fold_scores)))

    np.testing.assert_allclose(
        gscv.cv_results_["mean_test_score"],
        expected_means,
        rtol=0,
        atol=1e-8,
    )
    # Final refit on full X enforces concentrated_pw (NOT neutral_eq)
    assert abs(float(gscv.weights_[0]) - 0.90) <= 0.05 + 1e-6


def test_grid_search_cv_normalizes_array_previous_weights_for_preselection_pipeline() -> (
    None
):
    """F-02: 1D ndarray previous_weights aligns by DataFrame column name across pre-selection pipelines."""
    rng = np.random.default_rng(19)
    X = pd.DataFrame(
        rng.normal(0.0005, 0.01, (90, 4)),
        columns=["DROP_ME", "B", "C", "D"],
        index=pd.bdate_range("2025-01-02", periods=90),
    )
    pipe = Pipeline(
        [
            ("pre_0", _DropFirstColSelector()),
            ("optimizer", MeanRisk(max_turnover=0.20)),
        ]
    )
    prev_arr = np.array([0.10, 0.30, 0.30, 0.30])

    gscv = SkfolioGridSearchCV(
        estimator=pipe,
        param_grid={"optimizer__l2_coef": [0.0, 0.1]},
        cv=WalkForward(train_size=40, test_size=25),
        previous_weights=prev_arr,
    )
    gscv.fit(X)

    assert gscv.weights_ is not None
    assert len(gscv.weights_) == 4
    assert gscv.weights_[0] == pytest.approx(0.0, abs=1e-12)
    assert sum(gscv.weights_) == pytest.approx(1.0, abs=1e-6)


def test_grid_search_cv_sanitizes_degenerate_cvar_zero_and_sharpe_inf_folds() -> None:
    """F-02: Zero-loss (CVaR=0 / std=0) and all-zero inner folds must not produce inf/NaN/1e15 ties."""
    rng = np.random.default_rng(42)
    X_train = pd.DataFrame(
        {
            "A": rng.normal(0.0020, 0.025, 40),
            "B": rng.normal(0.0005, 0.008, 40),
            "C": rng.normal(0.0004, 0.008, 40),
        }
    )
    # Fold 0 test: zero losses with 1 zero return -> raw CVaR ~= -2.5e-18 -> raw CVaR ratio ~= -1.07e15!
    X_cvar0 = pd.DataFrame(
        {
            "A": [0.005] * 19 + [0.0],
            "B": [0.002] * 19 + [0.0],
            "C": [0.001] * 19 + [0.0],
        }
    )
    # Fold 0 alt: strictly positive constant returns -> raw Sharpe = +inf, raw CVaR ratio = -1.0
    X_pos_const = pd.DataFrame(
        {
            "A": [0.002] * 20,
            "B": [0.002] * 20,
            "C": [0.002] * 20,
        }
    )
    # Fold 0 alt: all zeros -> raw CVaR ratio = NaN, raw Sharpe = NaN
    X_all_zero = pd.DataFrame(
        {
            "A": [0.0] * 20,
            "B": [0.0] * 20,
            "C": [0.0] * 20,
        }
    )
    # Fold 1 test: genuinely informative volatile fold where high-vol A crashes and diversified B/C win
    X_volatile = pd.DataFrame(
        {
            "A": rng.normal(-0.0040, 0.025, 20),
            "B": rng.normal(0.0015, 0.008, 20),
            "C": rng.normal(0.0015, 0.008, 20),
        }
    )

    param_grid = {"l2_coef": [0.0, 5.0]}
    cv = WalkForward(train_size=40, test_size=20)

    # 1. CVaR ratio on X_cvar0 + X_volatile: must not explode to -1e15 and must pick l2_coef=5.0
    X_case_cvar0 = pd.concat([X_train, X_cvar0, X_volatile], ignore_index=True)
    with warnings.catch_warnings(record=True) as w_cvar:
        warnings.simplefilter("always")
        gscv_cvar = SkfolioGridSearchCV(
            estimator=MeanRisk(objective_function=ObjectiveFunction.MAXIMIZE_RETURN),
            param_grid=param_grid,
            cv=cv,
            scoring=make_scorer(RatioMeasure.CVAR_RATIO),
        ).fit(X_case_cvar0)
    runtime_warns = [x for x in w_cvar if issubclass(x.category, RuntimeWarning)]
    assert runtime_warns == []
    scores_cvar = gscv_cvar.cv_results_["mean_test_score"]
    assert np.all(np.isfinite(scores_cvar))
    assert np.all(np.abs(scores_cvar) < 1e4)
    assert gscv_cvar.best_params_ == {"l2_coef": 5.0}

    # 2. Sharpe ratio & default scoring=None on X_pos_const + X_volatile: no +inf tie, picks l2_coef=5.0
    X_case_pos = pd.concat([X_train, X_pos_const, X_volatile], ignore_index=True)
    for scoring_arg in (make_scorer(RatioMeasure.SHARPE_RATIO), None):
        with warnings.catch_warnings(record=True) as w_sh:
            warnings.simplefilter("always")
            gscv_sh = SkfolioGridSearchCV(
                estimator=MeanRisk(
                    objective_function=ObjectiveFunction.MAXIMIZE_RETURN
                ),
                param_grid=param_grid,
                cv=cv,
                scoring=scoring_arg,
            ).fit(X_case_pos)
        runtime_warns = [x for x in w_sh if issubclass(x.category, RuntimeWarning)]
        assert runtime_warns == []
        scores_sh = gscv_sh.cv_results_["mean_test_score"]
        assert np.all(np.isfinite(scores_sh))
        assert gscv_sh.best_params_ == {"l2_coef": 5.0}

    # 3. All-zero fold + volatile fold: no NaN poisoning, picks l2_coef=5.0
    X_case_zero = pd.concat([X_train, X_all_zero, X_volatile], ignore_index=True)
    with warnings.catch_warnings(record=True) as w_zero:
        warnings.simplefilter("always")
        gscv_zero = SkfolioGridSearchCV(
            estimator=MeanRisk(objective_function=ObjectiveFunction.MAXIMIZE_RETURN),
            param_grid=param_grid,
            cv=cv,
            scoring=make_scorer(RatioMeasure.CVAR_RATIO),
        ).fit(X_case_zero)
    runtime_warns = [x for x in w_zero if issubclass(x.category, RuntimeWarning)]
    assert runtime_warns == []
    scores_zero = gscv_zero.cv_results_["mean_test_score"]
    assert np.all(np.isfinite(scores_zero))
    assert gscv_zero.best_params_ == {"l2_coef": 5.0}


def test_make_safe_scorer_stateless_helper() -> None:
    """F-02: _make_safe_scorer preserves finite volatile scores and sanitizes degenerate ratio folds."""
    from flowportfolio.core.optimisation import _make_safe_scorer

    X_train = _sample_returns(n_obs=50, seed=3)
    m = MeanRisk().fit(X_train)

    raw_cvar = make_scorer(RatioMeasure.CVAR_RATIO)
    safe_cvar = _make_safe_scorer(raw_cvar)
    safe_sharpe = _make_safe_scorer(make_scorer(RatioMeasure.SHARPE_RATIO))

    # Normal volatile fold: exact passthrough
    X_vol = _sample_returns(n_obs=30, seed=99)
    assert safe_cvar(m, X_vol) == pytest.approx(raw_cvar(m, X_vol), rel=0, abs=1e-15)

    # Zero-loss fold with CVaR <= 0: finite positive ratio-unit score (not raw daily mean ~0.004), no warnings
    X_cvar0 = pd.DataFrame(
        {
            "A": [0.005] * 19 + [0.0],
            "B": [0.002] * 19 + [0.0],
            "C": [0.001] * 19 + [0.0],
        }
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        s_cvar = safe_cvar(m, X_cvar0)
    assert [x for x in w if issubclass(x.category, RuntimeWarning)] == []
    assert math.isfinite(s_cvar)
    assert s_cvar > float(m.predict(X_cvar0).mean) * 10.0

    # Constant positive fold with Sharpe = +inf: finite positive ratio-unit score, no warnings
    X_pos = pd.DataFrame({"A": [0.005] * 20, "B": [0.002] * 20, "C": [0.001] * 20})
    with warnings.catch_warnings(record=True) as w2:
        warnings.simplefilter("always")
        s_sh = safe_sharpe(m, X_pos)
    assert [x for x in w2 if issubclass(x.category, RuntimeWarning)] == []
    assert math.isfinite(s_sh)
    assert s_sh > float(m.predict(X_pos).mean) * 10.0

    # All-zero fold: 0.0, no warnings
    X_zero = pd.DataFrame({"A": [0.0] * 20, "B": [0.0] * 20, "C": [0.0] * 20})
    with warnings.catch_warnings(record=True) as w3:
        warnings.simplefilter("always")
        assert safe_cvar(m, X_zero) == 0.0
        assert safe_sharpe(m, X_zero) == 0.0
    assert [x for x in w3 if issubclass(x.category, RuntimeWarning)] == []

    # Custom non-skfolio scorer returning NaN or inf falls back to 0.0
    safe_nan = _make_safe_scorer(lambda est, X: float("nan"))
    safe_inf = _make_safe_scorer(lambda est, X: float("inf"))
    assert safe_nan(m, X_vol) == 0.0
    assert safe_inf(m, X_vol) == 0.0


def test_safe_scorer_negative_cvar_all_gains_fold_outranks_normal_fold() -> None:
    """P2-1: All-positive-gains folds (negative CVaR) must score in ratio units and outrank modest losing-tail folds."""
    from flowportfolio.core.optimisation import _make_safe_scorer

    rng = np.random.default_rng(42)
    X_train = _sample_returns(n_obs=60, seed=11)
    m = MeanRisk().fit(X_train)
    safe_cvar = _make_safe_scorer(make_scorer(RatioMeasure.CVAR_RATIO))

    # All-positive gains fold -> CVaR < 0 (raw cvar_ratio would be negative!)
    X_all_gains = pd.DataFrame(
        {
            "A": rng.uniform(0.0005, 0.0035, 25),
            "B": rng.uniform(0.0005, 0.0035, 25),
            "C": rng.uniform(0.0005, 0.0035, 25),
        }
    )
    assert float(m.predict(X_all_gains).cvar) < 0.0

    # Modest normal fold with positive CVaR > 0 and finite positive cvar_ratio (~0.10)
    X_modest = pd.DataFrame(
        {
            "A": rng.normal(0.001, 0.01, 25),
            "B": rng.normal(0.001, 0.01, 25),
            "C": rng.normal(0.001, 0.01, 25),
        }
    )
    normal_score = safe_cvar(m, X_modest)
    all_gains_score = safe_cvar(m, X_all_gains)

    assert normal_score > 0.0
    assert all_gains_score > normal_score

    # Strictly higher-mean all-positive fold (same dispersion) must score higher
    X_higher_gains = X_all_gains + 0.002
    assert safe_cvar(m, X_higher_gains) > all_gains_score


def test_set_previous_weights_chain_mock_free_cycle_safe_and_defensive_copy() -> None:
    """F-10: optimisation.py contains zero _mock_name branches; chain propagation is cycle-safe."""
    source = inspect.getsource(opt_module)
    assert "_mock_name" not in source

    fb = MeanRisk()
    primary = MeanRisk(fallback=fb)
    pipe = Pipeline([("pre_0", _DropFirstColSelector()), ("optimizer", primary)])
    gscv = SkfolioGridSearchCV(estimator=pipe, param_grid={})

    weights_dict = {"A": 0.5, "B": 0.5}
    set_previous_weights_chain(gscv, weights_dict)
    assert gscv.previous_weights == weights_dict
    assert primary.previous_weights == weights_dict
    assert fb.previous_weights == weights_dict
    # Defensive copies: mutating original dict does not mutate propagated weights
    weights_dict["A"] = 0.99
    assert primary.previous_weights["A"] == 0.5
    assert fb.previous_weights["A"] == 0.5

    # Defensive copy for numpy arrays
    weights_arr = np.array([0.6, 0.4])
    set_previous_weights_chain(primary, weights_arr)
    weights_arr[0] = 0.99
    assert primary.previous_weights[0] == pytest.approx(0.6)
    assert fb.previous_weights[0] == pytest.approx(0.6)

    # Cycle guard on self-referential fallback chain
    a = MeanRisk()
    b = MeanRisk()
    a.fallback = b
    b.fallback = a
    set_previous_weights_chain(a, {"A": 1.0})
    assert a.previous_weights == {"A": 1.0}
    assert b.previous_weights == {"A": 1.0}


def test_grid_search_cv_refit_false_and_clone_immutability() -> None:
    """Edge cases: refit=False does not crash and fit() does not mutate portfolio_params."""
    X = _sample_returns(n_obs=80, seed=5)

    gscv_no_refit = SkfolioGridSearchCV(
        estimator=MeanRisk(),
        param_grid={"l2_coef": [0.0, 0.1]},
        cv=WalkForward(train_size=40, test_size=20),
        refit=False,
    )
    gscv_no_refit.fit(X)
    assert not hasattr(gscv_no_refit, "best_estimator_")
    assert "mean_test_score" in gscv_no_refit.cv_results_

    inner_est = MeanRisk(portfolio_params={"management_fees": 0.0002})
    gscv = SkfolioGridSearchCV(
        estimator=inner_est,
        param_grid={"l2_coef": [0.0]},
        cv=WalkForward(train_size=40, test_size=20),
        portfolio_params={"tag": "grid_tag"},
    )
    gscv.fit(X)

    # __init__ parameter portfolio_params must NOT be mutated in place by fit()
    assert gscv.portfolio_params == {"tag": "grid_tag"}
    cloned = clone(gscv)
    assert cloned.portfolio_params == {"tag": "grid_tag"}

    # predict() still honors merged inner + outer portfolio_params
    port = gscv.predict(X)
    assert port.tag == "grid_tag"
    assert port.management_fees == pytest.approx(0.0002)


def test_grid_search_cv_exposes_wrapped_risk_free_rate_for_cross_val_predict_mpp() -> (
    None
):
    """Sprint 5 (#6b gap): cross_val_predict must inherit risk_free_rate on outer MultiPeriodPortfolio."""
    from skfolio import Portfolio
    from skfolio.model_selection import cross_val_predict
    from skfolio.optimization import HierarchicalRiskParity

    X = _sample_returns(n_obs=120, seed=31)
    rfr = 0.0504 / 252.0
    cv = WalkForward(train_size=60, test_size=30)

    # 1. Optimizer with top-level risk_free_rate (MeanRisk)
    gscv_mr = SkfolioGridSearchCV(
        estimator=Pipeline([("optimizer", MeanRisk(risk_free_rate=rfr))]),
        param_grid={"optimizer__l2_coef": [0.0]},
        cv=WalkForward(train_size=30, test_size=15),
    )
    mpp_mr = cross_val_predict(gscv_mr, X, cv=cv)
    assert float(mpp_mr.risk_free_rate) == pytest.approx(rfr, abs=1e-12)
    oracle_mr = Portfolio(X=mpp_mr.returns[:, None], weights=[1.0], risk_free_rate=rfr)
    assert float(mpp_mr.sharpe_ratio) == pytest.approx(
        float(oracle_mr.sharpe_ratio), rel=1e-9, abs=1e-12
    )

    # 2. Optimizer carrying risk_free_rate via portfolio_params (HierarchicalRiskParity)
    gscv_hrp = SkfolioGridSearchCV(
        estimator=Pipeline(
            [
                (
                    "optimizer",
                    HierarchicalRiskParity(portfolio_params={"risk_free_rate": rfr}),
                )
            ]
        ),
        param_grid={},
        cv=WalkForward(train_size=30, test_size=15),
    )
    mpp_hrp = cross_val_predict(gscv_hrp, X, cv=cv)
    assert float(mpp_hrp.risk_free_rate) == pytest.approx(rfr, abs=1e-12)


def test_grid_search_cv_portfolio_params_risk_free_rate_overrides_zero_estimator_attr() -> (
    None
):
    """Sprint 5 (S1): inner.portfolio_params['risk_free_rate'] must take precedence over default 0.0 attr."""
    from skfolio.model_selection import cross_val_predict

    X = _sample_returns(n_obs=120, seed=37)
    rfr = 0.0504 / 252.0
    gscv = SkfolioGridSearchCV(
        estimator=MeanRisk(
            risk_free_rate=0.0, portfolio_params={"risk_free_rate": rfr}
        ),
        param_grid={"l2_coef": [0.0]},
        cv=WalkForward(train_size=30, test_size=15),
    )
    assert gscv.risk_free_rate == pytest.approx(rfr, abs=1e-12)
    mpp = cross_val_predict(gscv, X, cv=WalkForward(train_size=60, test_size=30))
    assert float(mpp.risk_free_rate) == pytest.approx(rfr, abs=1e-12)
    gscv.fit(X)
    assert gscv.risk_free_rate == pytest.approx(rfr, abs=1e-12)


def test_grid_search_cv_tuning_max_turnover_in_param_grid_uses_feasible_inner_baseline() -> (
    None
):
    """Sprint 5 (S8): Tuning max_turnover in param_grid on an unconstrained base estimator must remain feasible in inner CV."""
    X = _sample_returns(n_obs=100, seed=41)
    pw = {"A": 0.60, "B": 0.20, "C": 0.20}
    gscv = SkfolioGridSearchCV(
        estimator=MeanRisk(),
        param_grid={"max_turnover": [0.05, 0.15]},
        cv=WalkForward(train_size=40, test_size=20),
        previous_weights=pw,
    )
    assert gscv.needs_previous_weights is True
    gscv.fit(X)
    assert gscv.best_params_["max_turnover"] in (0.05, 0.15)
    assert gscv.weights_ is not None
    assert (
        abs(float(gscv.weights_[0]) - 0.60) <= gscv.best_params_["max_turnover"] + 1e-6
    )

    # Also verify list-of-dicts param_grid with Pipeline step prefix
    gscv_list = SkfolioGridSearchCV(
        estimator=Pipeline([("optimizer", MeanRisk())]),
        param_grid=[
            {"optimizer__max_turnover": [0.05]},
            {"optimizer__max_turnover": [0.15]},
        ],
        cv=WalkForward(train_size=40, test_size=20),
        previous_weights=pw,
    )
    assert gscv_list.needs_previous_weights is True
    gscv_list.fit(X)
    assert gscv_list.best_params_["optimizer__max_turnover"] in (0.05, 0.15)
