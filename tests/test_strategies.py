import numpy as np
import pandas as pd
import pytest
from skfolio.optimization import (
    HierarchicalRiskParity,
    MeanRisk,
    NestedClustersOptimization,
)
from skfolio.prior import EmpiricalPrior
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline

from flowportfolio.core.constraints import ConstraintSpec
from flowportfolio.strategies import StrategyBuilder


class MockTransformer(BaseEstimator, TransformerMixin):
    def fit_transform(self, X):
        return X


def test_init_raises_type_error():
    with pytest.raises(TypeError, match="constraints must be a list"):
        StrategyBuilder(constraints="not a list")


def test_init_raises_type_error_elements():
    with pytest.raises(TypeError, match="All constraints must be strings"):
        StrategyBuilder(constraints=[123])


def test_init_raises_type_error_prior():
    with pytest.raises(TypeError, match="prior must be an instance of BasePrior"):
        StrategyBuilder(constraints=[], prior="not a prior")


def test_add_pre_selection_raises_type_error():
    builder = StrategyBuilder(constraints=[])
    with pytest.raises(
        TypeError, match="transformer must implement a fit_transform method"
    ):
        builder.add_pre_selection("not a transformer")


def test_set_optimizer_raises_type_error_if_none():
    builder = StrategyBuilder(constraints=[])
    with pytest.raises(TypeError, match="optimizer must not be None"):
        builder.set_optimizer(None)


def test_set_optimizer_raises_runtime_error_if_called_twice():
    builder = StrategyBuilder(constraints=[])
    builder.set_optimizer(MeanRisk())
    with pytest.raises(
        RuntimeError, match="set_optimizer\\(\\) has already been called"
    ):
        builder.set_optimizer(MeanRisk())


def test_set_optimizer_fallback_unsupported_raises_type_error():
    class UnsupportedOptimizer:
        pass

    builder = StrategyBuilder(constraints=[])
    with pytest.raises(
        TypeError,
        match="does not implement a 'get_params' method|does not support fallback semantics",
    ):
        builder.set_optimizer(UnsupportedOptimizer(), fallback=MeanRisk())


def test_build_pipeline_raises_runtime_error_no_optimizer():
    builder = StrategyBuilder(constraints=[])
    with pytest.raises(RuntimeError, match="No optimizer set"):
        builder.build_pipeline()


def test_build_pipeline_ordering_and_prior_injection():
    prior = EmpiricalPrior()
    builder = StrategyBuilder(constraints=[], prior=prior)

    pre = MockTransformer()
    opt = MeanRisk()

    builder.add_pre_selection(pre)
    builder.set_optimizer(opt)

    pipeline = builder.build_pipeline()
    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.steps) == 2
    assert pipeline.steps[0][0] == "pre_0"
    assert pipeline.steps[1][0] == "optimizer"

    # Check prior injection
    assert pipeline.steps[1][1].prior_estimator is not prior
    assert type(pipeline.steps[1][1].prior_estimator) is type(prior)


def test_build_nco_clustering_dispatch_and_no_mutation():
    builder = StrategyBuilder()

    inner = MeanRisk()
    outer = MeanRisk()

    nco = builder.build_nco(inner, outer, clusterer="kmeans")
    assert isinstance(nco, NestedClustersOptimization)

    # Check invalid clusterer
    with pytest.raises(ValueError, match="clusterer must be one of"):
        builder.build_nco(inner, outer, clusterer="invalid")


def test_build_nco_does_not_mutate_estimators():
    prior = EmpiricalPrior()
    builder = StrategyBuilder(prior=prior)

    inner = MeanRisk()
    outer = HierarchicalRiskParity()

    builder.build_nco(inner, outer)

    # Original inner should not have prior injected
    assert (
        not hasattr(inner, "prior_estimator")
        or inner.prior_estimator is None
        or inner.prior_estimator is not prior
    )


def test_build_nco_raises_type_error_if_none():
    builder = StrategyBuilder(constraints=[])
    with pytest.raises(TypeError, match="inner_estimator must not be None"):
        builder.build_nco(None, MeanRisk())
    with pytest.raises(TypeError, match="outer_estimator must not be None"):
        builder.build_nco(MeanRisk(), None)


# ---------------------------------------------------------------------------
# Reset lifecycle
# ---------------------------------------------------------------------------


def test_reset_clears_optimizer_state() -> None:
    """Test reset() allows set_optimizer() to be called again."""
    builder = StrategyBuilder(constraints=[])
    builder.set_optimizer(MeanRisk())
    builder.reset()
    # Should not raise after reset
    builder.set_optimizer(MeanRisk())
    assert builder._optimizer is not None


def test_reset_clears_pipeline_queues() -> None:
    """Test reset() clears pre_selection queue."""
    builder = StrategyBuilder(constraints=[])
    builder.add_pre_selection(MockTransformer())
    builder.reset()
    assert builder._pre_selection == []


def test_reset_preserves_base_config_by_default() -> None:
    """Test reset() retains constraints and prior when clear_base_config=False."""
    prior = EmpiricalPrior()
    builder = StrategyBuilder(constraints=["A >= 0.1"], prior=prior)
    builder.reset()
    assert builder.constraints == ["A >= 0.1"]
    assert builder.prior is prior


def test_reset_clears_base_config_when_flag_true() -> None:
    """Test reset(clear_base_config=True) also wipes constraints and prior."""
    prior = EmpiricalPrior()
    builder = StrategyBuilder(constraints=["A >= 0.1"], prior=prior)
    builder.reset(clear_base_config=True)
    assert builder.constraints == []
    assert builder.prior is None


def test_reset_returns_self_for_chaining() -> None:
    """Test reset() returns self, enabling fluent chaining."""
    builder = StrategyBuilder(constraints=[])
    returned = builder.reset()
    assert returned is builder


def test_init_defaults_to_empty_constraints() -> None:
    """Test StrategyBuilder can be instantiated with no arguments."""
    builder = StrategyBuilder()
    assert builder.constraints == []
    assert builder.prior is None


def test_init_accepts_none_constraints() -> None:
    """Test StrategyBuilder explicitly accepts constraints=None."""
    builder = StrategyBuilder(constraints=None)
    assert builder.constraints == []


def test_build_pipeline_clones_transformers() -> None:
    """Test build_pipeline creates independent clones of queued transformers."""
    pre = MockTransformer()
    builder = StrategyBuilder()
    builder.add_pre_selection(pre)
    builder.set_optimizer(MeanRisk())

    pipeline1 = builder.build_pipeline()
    pipeline2 = builder.build_pipeline()

    assert pipeline1.steps[0][1] is not pipeline2.steps[0][1]
    assert pipeline1.steps[1][1] is not pipeline2.steps[1][1]


def test_set_optimizer_preserves_pre_configured_constraints_when_builder_empty():
    opt = MeanRisk(linear_constraints=["pre_existing <= 0.2"])
    builder = StrategyBuilder(constraints=[])
    builder.set_optimizer(opt)

    pipeline = builder.build_pipeline()
    pipeline_opt = pipeline.steps[-1][1]
    assert pipeline_opt.linear_constraints == ["pre_existing <= 0.2"]


def test_set_optimizer_merges_linear_constraints():
    opt = MeanRisk(linear_constraints=["pre_existing <= 0.2"])
    builder = StrategyBuilder(constraints=["new_constraint >= 0.1"])
    builder.set_optimizer(opt)

    pipeline = builder.build_pipeline()
    pipeline_opt = pipeline.steps[-1][1]
    assert pipeline_opt.linear_constraints == [
        "pre_existing <= 0.2",
        "new_constraint >= 0.1",
    ]


def test_constraint_spec_binds_groups_and_linear_constraints():
    spec = ConstraintSpec(
        linear_constraints=["core >= 0.5"],
        groups={"A": ["core"], "B": ["satellite"]},
    )
    builder = StrategyBuilder(constraints=spec)
    opt = MeanRisk()
    builder.set_optimizer(opt)

    pipeline = builder.build_pipeline()
    pipeline_opt = pipeline.steps[-1][1]
    assert pipeline_opt.linear_constraints == ["core >= 0.5"]
    assert pipeline_opt.groups == {"A": ["core"], "B": ["satellite"]}


def test_build_nco_binds_groups_and_merges_constraints():
    spec = ConstraintSpec(
        linear_constraints=["core >= 0.4"],
        groups={"A": ["core"], "B": ["satellite"]},
    )
    builder = StrategyBuilder(constraints=spec)
    inner = MeanRisk()
    outer = MeanRisk()

    with pytest.raises(
        ValueError,
        match="Cross-asset linear constraints and turnover limits cannot be enforced",
    ):
        builder.build_nco(inner, outer)


def test_constraints_reassignment_replaces_not_accumulates():
    builder = StrategyBuilder(constraints=["old_constraint >= 0.1"])
    builder.set_optimizer(MeanRisk())
    builder.constraints = ["core <= 0.5"]

    pipeline = builder.build_pipeline()
    pipeline_opt = pipeline.steps[-1][1]
    assert pipeline_opt.linear_constraints == ["core <= 0.5"]

    # Verify build_nco populates portfolio_params["management_fees"] when fees present
    fee_spec = ConstraintSpec(management_fees={"A": 0.0001, "B": 0.0002})
    fee_builder = StrategyBuilder(constraints=fee_spec)
    nco = fee_builder.build_nco(MeanRisk(), MeanRisk())
    assert nco.portfolio_params is not None
    assert nco.portfolio_params.get("management_fees") == {"A": 0.0001, "B": 0.0002}


def test_safe_pre_selection_preserves_constrained_group():
    """Verify _SafePreSelectionTransformer preserves assets required by linear constraints."""
    import numpy as np
    from skfolio.pre_selection import DropCorrelated

    from flowportfolio.core.constraints import ConstraintBuilder
    from flowportfolio.core.universe import Universe
    from tests.test_audit_remediation_integration import StubProvider, _synthetic_prices

    u = Universe(
        tickers=["SPY", "QQQ", "TLT", "GLD"],
        metadata={
            "SPY": "satellite",
            "QQQ": "core",
            "TLT": "defensive",
            "GLD": "defensive",
        },
        fees={"SPY": 0.0009, "QQQ": 0.0020, "TLT": 0.0003, "GLD": 0.0040},
    )
    prices = _synthetic_prices(600)
    # Ensure QQQ is almost identical to SPY so DropCorrelated would otherwise drop QQQ
    prices["QQQ"] = prices["SPY"] * 1.001 + np.random.randn(600) * 0.0001
    u.fetch_data(data_provider=StubProvider(prices))
    u.anchor_history()

    spec = (
        ConstraintBuilder(u)
        .min_group("core", 0.40)
        .max_group("satellite", 0.30)
        .build()
    )
    pipe = (
        StrategyBuilder(constraints=spec)
        .add_pre_selection(DropCorrelated(threshold=0.85))
        .set_optimizer(MeanRisk())
        .build_pipeline()
    )

    pipe.fit(u.returns)
    opt = pipe.named_steps["optimizer"]
    assert "QQQ" in opt.feature_names_in_
    weights_dict = dict(zip(opt.feature_names_in_, opt.weights_))
    assert weights_dict["QQQ"] >= 0.40 - 1e-5


def test_adaptive_kmeans_and_pickle():
    """Verify _AdaptiveKMeans adapts cluster count on small N and is picklable."""
    import pickle

    import numpy as np

    from flowportfolio.strategies import _AdaptiveKMeans

    ak = _AdaptiveKMeans(n_clusters=8, random_state=42)
    # 8 assets, 100 features
    X = np.random.randn(8, 100)
    ak.fit(X)

    unique_clusters = len(set(ak.labels_))
    # Effective k should be min(8, 8//2) = 4
    assert unique_clusters == 4
    assert ak.n_clusters == 8  # unmutated

    # Picklable
    pickled = pickle.dumps(ak)
    restored = pickle.loads(pickled)
    assert np.array_equal(restored.labels_, ak.labels_)


def test_fee_aware_nco_deducts_fees():
    """Verify _FeeAwareNCO fits successfully and attributes fees to predicted portfolio."""
    import numpy as np
    import pandas as pd

    from flowportfolio.strategies import _FeeAwareNCO

    np.random.seed(42)
    returns = pd.DataFrame(
        np.random.randn(100, 4) * 0.01 + 0.0005, columns=["A", "B", "C", "D"]
    )
    fees = {"A": 0.001 / 252, "B": 0.002 / 252, "C": 0.001 / 252, "D": 0.003 / 252}

    nco = _FeeAwareNCO(
        inner_estimator=MeanRisk(),
        outer_estimator=MeanRisk(),
        portfolio_params={"management_fees": fees},
    )
    nco.fit(returns)
    assert nco.weights_ is not None
    assert abs(sum(nco.weights_) - 1.0) < 1e-5

    port = nco.predict(returns)
    assert hasattr(port, "management_fees")


def test_warnings_on_unsupported_constraints_in_optimizer(caplog):
    """Verify StrategyBuilder logs warnings when optimizer lacks linear_constraints or max_turnover."""
    import logging

    from skfolio.optimization import HierarchicalRiskParity

    builder = StrategyBuilder(constraints=["core >= 0.5"])
    builder._max_turnover = 0.05
    builder._previous_weights = {"A": 0.5, "B": 0.5}

    with caplog.at_level(logging.WARNING):
        builder.set_optimizer(HierarchicalRiskParity())

    assert "does not support linear_constraints" in caplog.text
    assert "does not support max_turnover" in caplog.text


# ---------------------------------------------------------------------------
# Cycle 6 (F-07 & F-11): Multi-Asset Restore, _FeeAwareNCO, Seeding & Cleanup
# ---------------------------------------------------------------------------


def _correlated_core_returns(n_obs: int = 160) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    base = rng.normal(0.0006, 0.01, n_obs)
    return pd.DataFrame(
        {
            "C1": base,
            "C2": base + rng.normal(0.0, 1e-4, n_obs),
            "C3": base + rng.normal(0.0, 1e-4, n_obs),
            "S1": rng.normal(0.0003, 0.01, n_obs),
            "S2": rng.normal(0.0002, 0.01, n_obs),
        }
    )


class _NumpyMaskSelector(BaseEstimator, TransformerMixin):
    """Minimal selector returning raw np.ndarray (no set_output support)."""

    def __init__(self, keep_indices: tuple[int, ...] = (0, 3, 4)) -> None:
        self.keep_indices = keep_indices

    def fit(self, X, y=None):
        n_cols = X.shape[1]
        mask = np.zeros(n_cols, dtype=bool)
        for idx in self.keep_indices:
            mask[idx] = True
        self.mask_ = mask
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.mask_]

    def get_support(self, indices: bool = False):
        return np.where(self.mask_)[0] if indices else self.mask_


def test_safe_pre_selection_restores_multiple_group_members_for_max_weights() -> None:
    """F-07 (O5): min_group('core', 0.60) + MeanRisk(max_weights=0.40) preserves >= 2 core assets."""
    from skfolio.pre_selection import DropCorrelated

    X = _correlated_core_returns()
    spec = ConstraintSpec(
        linear_constraints=["core >= 0.6"],
        groups={
            "C1": ["core"],
            "C2": ["core"],
            "C3": ["core"],
            "S1": ["sat"],
            "S2": ["sat"],
        },
    )
    pipe = (
        StrategyBuilder(constraints=spec)
        .add_pre_selection(DropCorrelated(threshold=0.9))
        .set_optimizer(MeanRisk(max_weights=0.40))
        .build_pipeline()
    )

    pipe.fit(X)
    opt = pipe.named_steps["optimizer"]
    surviving = list(opt.feature_names_in_)
    core_surviving = [c for c in surviving if c in {"C1", "C2", "C3"}]
    assert len(core_surviving) >= 2

    weights = dict(zip(surviving, opt.weights_))
    assert sum(weights[c] for c in core_surviving) >= 0.60 - 1e-5
    assert max(weights.values()) <= 0.40 + 1e-5


def test_safe_pre_selection_restores_complement_for_max_group_and_max_weights() -> None:
    """F-07: max_group('sat', 0.30) + MeanRisk(max_weights=0.40) restores >= 2 non-sat assets."""
    from skfolio.pre_selection import DropCorrelated

    X = _correlated_core_returns()
    spec = ConstraintSpec(
        linear_constraints=["sat <= 0.3"],
        groups={
            "C1": ["core"],
            "C2": ["core"],
            "C3": ["core"],
            "S1": ["sat"],
            "S2": ["sat"],
        },
    )
    pipe = (
        StrategyBuilder(constraints=spec)
        .add_pre_selection(DropCorrelated(threshold=0.9))
        .set_optimizer(MeanRisk(max_weights=0.40))
        .build_pipeline()
    )

    pipe.fit(X)
    opt = pipe.named_steps["optimizer"]
    surviving = list(opt.feature_names_in_)
    non_sat = [c for c in surviving if c not in {"S1", "S2"}]
    assert len(non_sat) >= 2

    weights = dict(zip(surviving, opt.weights_))
    assert sum(weights.get(c, 0.0) for c in ("S1", "S2")) <= 0.30 + 1e-5
    assert max(weights.values()) <= 0.40 + 1e-5


def test_safe_pre_selection_restores_asset_level_linear_constraint() -> None:
    """F-07: Asset-level linear constraint ('C2 >= 0.25') restores dropped ticker C2."""
    X = _correlated_core_returns()
    spec = ConstraintSpec(
        linear_constraints=["C2 >= 0.25"],
        groups={
            "C1": ["core"],
            "C2": ["core"],
            "C3": ["core"],
            "S1": ["sat"],
            "S2": ["sat"],
        },
    )
    pipe = (
        StrategyBuilder(constraints=spec)
        .add_pre_selection(_NumpyMaskSelector(keep_indices=(0, 3, 4)))
        .set_optimizer(MeanRisk())
        .build_pipeline()
    )

    pipe.fit(X)
    opt = pipe.named_steps["optimizer"]
    weights = dict(zip(opt.feature_names_in_, opt.weights_))
    assert "C2" in weights
    assert weights["C2"] >= 0.25 - 1e-5


def test_safe_pre_selection_restores_global_simplex_capacity_for_max_weights() -> None:
    """F-07: MeanRisk(max_weights=0.40) without linear constraints restores >= 3 assets."""
    X = _correlated_core_returns()
    pipe = (
        StrategyBuilder()
        .add_pre_selection(_NumpyMaskSelector(keep_indices=(0, 3)))
        .set_optimizer(MeanRisk(max_weights=0.40))
        .build_pipeline()
    )

    pipe.fit(X)
    opt = pipe.named_steps["optimizer"]
    assert len(opt.feature_names_in_) >= 3
    assert max(opt.weights_) <= 0.40 + 1e-5


def test_restore_feasible_subset_stateless_helper() -> None:
    """F-07: Standalone stateless feasibility helper restores minimal feasible asset set."""
    from flowportfolio.strategies import _restore_feasible_subset

    all_cols = ["C1", "C2", "C3", "S1", "S2"]
    groups = {
        "C1": ["core"],
        "C2": ["core"],
        "C3": ["core"],
        "S1": ["sat"],
        "S2": ["sat"],
    }
    restored, reasons = _restore_feasible_subset(
        surviving=["C1", "S1", "S2"],
        all_cols=all_cols,
        groups=groups,
        linear_constraints=["core >= 0.6"],
        max_weights=0.40,
    )
    assert restored == ["C1", "C2", "S1", "S2"]
    assert len(reasons) >= 1


def test_build_pipeline_wraps_pre_selection_for_fee_and_cost_dicts() -> None:
    """F-07: Pre-selection is wrapped when dict management_fees or transaction_costs are present."""
    from flowportfolio.strategies import _SafePreSelectionTransformer

    X = _correlated_core_returns()
    fees = {"C1": 0.0001, "C2": 0.0002, "C3": 0.0001, "S1": 0.0003, "S2": 0.0002}

    # Case 1: management_fees dict only (no linear_constraints, no max_turnover)
    fee_pipe = (
        StrategyBuilder(constraints=ConstraintSpec(management_fees=fees))
        .add_pre_selection(_NumpyMaskSelector(keep_indices=(0, 3, 4)))
        .set_optimizer(MeanRisk())
        .build_pipeline()
    )
    assert isinstance(fee_pipe.named_steps["pre_0"], _SafePreSelectionTransformer)
    fee_pipe.fit(X)
    assert list(fee_pipe.named_steps["optimizer"].feature_names_in_) == [
        "C1",
        "S1",
        "S2",
    ]

    # Case 2: transaction_costs dict + previous_weights (no linear_constraints, no max_turnover)
    prev = {"C1": 0.2, "C2": 0.2, "C3": 0.2, "S1": 0.2, "S2": 0.2}
    tc_pipe = (
        StrategyBuilder(
            constraints=ConstraintSpec(transaction_costs=fees, previous_weights=prev)
        )
        .add_pre_selection(_NumpyMaskSelector(keep_indices=(0, 3, 4)))
        .set_optimizer(MeanRisk())
        .build_pipeline()
    )
    assert isinstance(tc_pipe.named_steps["pre_0"], _SafePreSelectionTransformer)
    tc_pipe.fit(X)
    assert list(tc_pipe.named_steps["optimizer"].feature_names_in_) == [
        "C1",
        "S1",
        "S2",
    ]


def test_fee_aware_nco_ndarray_fit_predict_parity_and_validation() -> None:
    """F-07 & P2-9b: _FeeAwareNCO rejects ambiguous dict fees on raw ndarray, matches positional array fees, and preserves fitted DataFrame feature names."""
    from skfolio.cluster import HierarchicalClustering, LinkageMethod
    from skfolio.distance import PearsonDistance

    from flowportfolio.strategies import _FeeAwareNCO

    X_df = _correlated_core_returns(n_obs=120)
    X_arr = X_df.to_numpy()
    fees = {
        "C1": 0.0001,
        "C2": 0.0004,
        "C3": 0.0002,
        "S1": 0.0003,
        "S2": 0.0005,
    }
    fee_arr = np.array([fees[c] for c in X_df.columns], dtype=float)
    builder_dict = StrategyBuilder(constraints=ConstraintSpec(management_fees=fees))

    nco_df = builder_dict.build_nco(MeanRisk(), MeanRisk(), clusterer="ward")
    nco_arr = _FeeAwareNCO(
        inner_estimator=MeanRisk(),
        outer_estimator=MeanRisk(),
        distance_estimator=PearsonDistance(),
        clustering_estimator=HierarchicalClustering(linkage_method=LinkageMethod.WARD),
        portfolio_params={"management_fees": fee_arr},
    )

    nco_df.fit(X_df)
    nco_arr.fit(X_arr)

    np.testing.assert_allclose(nco_arr.weights_, nco_df.weights_, rtol=0, atol=1e-6)

    port_df = nco_df.predict(X_df)
    port_arr = nco_arr.predict(X_arr)
    port_df_on_arr = nco_df.predict(X_arr)

    np.testing.assert_allclose(port_arr.returns, port_df.returns, rtol=0, atol=1e-6)
    np.testing.assert_allclose(
        port_df_on_arr.returns, port_df.returns, rtol=0, atol=1e-6
    )

    # Raw unlabeled ndarray with dict management_fees must fail instead of guessing insertion order
    nco_dict_raw = builder_dict.build_nco(MeanRisk(), MeanRisk(), clusterer="ward")
    with pytest.raises(ValueError, match="DataFrame|feature names"):
        nco_dict_raw.fit(X_arr)

    # Mismatched fee array width vs ndarray columns raises ValueError
    bad_arr = X_arr[:, :3]
    nco_bad = _FeeAwareNCO(
        inner_estimator=MeanRisk(),
        outer_estimator=MeanRisk(),
        portfolio_params={"management_fees": fee_arr},
    )
    with pytest.raises(ValueError, match="management_fees"):
        nco_bad.fit(bad_arr)


def test_fee_aware_nco_does_not_mutate_inner_estimator_during_fit() -> None:
    """F-07: _FeeAwareNCO.fit never mutates self.inner_estimator (even mid-fit)."""
    from flowportfolio.strategies import _FeeAwareNCO

    X_df = _correlated_core_returns(n_obs=120)
    fees = {c: 0.0002 for c in X_df.columns}

    nco_ref: list[object] = []
    observed_during_fit: list[tuple[bool, object]] = []

    class _SpyMeanRisk(MeanRisk):
        def fit(self, X, y=None, **fit_params):
            if nco_ref:
                parent = nco_ref[0]
                observed_during_fit.append(
                    (
                        parent.inner_estimator is orig_inner,  # type: ignore[attr-defined]
                        parent.inner_estimator.management_fees,  # type: ignore[attr-defined]
                    )
                )
            return super().fit(X, y=y, **fit_params)

    orig_inner = _SpyMeanRisk(management_fees=dict(fees))
    nco = _FeeAwareNCO(
        inner_estimator=orig_inner,
        outer_estimator=MeanRisk(),
        portfolio_params={"management_fees": dict(fees)},
    )
    nco_ref.append(nco)
    nco.fit(X_df)

    assert len(observed_during_fit) > 0
    assert all(same_obj and mf == fees for same_obj, mf in observed_during_fit)
    assert nco.inner_estimator is orig_inner
    assert orig_inner.management_fees == fees


def test_subtract_management_fees_stateless_helper() -> None:
    """F-07: Standalone _subtract_management_fees handles DataFrame, ndarray, and mismatches."""
    from flowportfolio.strategies import _subtract_management_fees

    df = pd.DataFrame({"A": [0.01, 0.02], "B": [0.03, 0.04]})
    fees = {"A": 0.001, "B": 0.002}

    df_net = _subtract_management_fees(df, fees)
    assert isinstance(df_net, pd.DataFrame)
    np.testing.assert_allclose(df_net["A"].to_numpy(), [0.009, 0.019], atol=1e-15)
    np.testing.assert_allclose(df_net["B"].to_numpy(), [0.028, 0.038], atol=1e-15)
    # Original DataFrame must remain unmutated
    assert df["A"].iloc[0] == 0.01

    arr_net = _subtract_management_fees(df.to_numpy(), fees, feature_names=["A", "B"])
    assert isinstance(arr_net, np.ndarray)
    np.testing.assert_allclose(arr_net, df_net.to_numpy(), atol=1e-15)

    arr_pos = _subtract_management_fees(df.to_numpy(), np.array([0.001, 0.002]))
    np.testing.assert_allclose(arr_pos, df_net.to_numpy(), atol=1e-15)

    scalar_net = _subtract_management_fees(df.to_numpy(), 0.001)
    np.testing.assert_allclose(scalar_net, df.to_numpy() - 0.001, atol=1e-15)

    with pytest.raises(ValueError, match="management_fees"):
        _subtract_management_fees(df.to_numpy(), fees)

    with pytest.raises(ValueError, match="management_fees"):
        _subtract_management_fees(np.zeros((5, 3)), np.array([0.001, 0.002]))


def test_build_nco_kmeans_deterministic_default_and_adaptive_metadata() -> None:
    """F-07: build_nco(clusterer='kmeans') uses deterministic seed and exposes effective_n_clusters_."""
    rng = np.random.default_rng(123)
    X_df = pd.DataFrame(
        rng.normal(0.0005, 0.01, (120, 6)), columns=[f"A{i}" for i in range(6)]
    )
    builder = StrategyBuilder()
    nco1 = builder.build_nco(MeanRisk(), MeanRisk(), clusterer="kmeans")
    nco2 = builder.build_nco(MeanRisk(), MeanRisk(), clusterer="kmeans")

    assert nco1.clustering_estimator.random_state is not None
    assert (
        nco1.clustering_estimator.random_state == nco2.clustering_estimator.random_state
    )

    nco1.fit(X_df)
    nco2.fit(X_df)
    np.testing.assert_allclose(nco1.weights_, nco2.weights_, rtol=0, atol=1e-12)
    assert nco1.clustering_estimator_.effective_n_clusters_ == len(
        set(nco1.clustering_estimator_.labels_)
    )
    assert nco1.clustering_estimator_.effective_n_clusters_ == 3
    assert nco1.clustering_estimator_.n_clusters == 8  # hyperparameter unmutated

    for bad_seed in (-1, True, "42"):
        with pytest.raises(ValueError, match="random_state"):
            builder.build_nco(
                MeanRisk(),
                MeanRisk(),
                clusterer="kmeans",
                random_state=bad_seed,  # type: ignore[arg-type]
            )


def test_strategy_builder_has_no_dead_fallback_attribute() -> None:
    """F-11: Unused self._fallback dead code is removed from StrategyBuilder."""
    builder = StrategyBuilder()
    assert not hasattr(builder, "_fallback")
    builder.reset()
    assert not hasattr(builder, "_fallback")


def test_custom_estimators_sklearn_clone_roundtrip() -> None:
    """Verify custom wrappers preserve sklearn.base.clone() hyperparameter compliance."""
    from skfolio.pre_selection import DropCorrelated

    from flowportfolio.strategies import (
        _AdaptiveKMeans,
        _FeeAwareNCO,
        _SafePreSelectionTransformer,
    )

    tr = _SafePreSelectionTransformer(
        DropCorrelated(),
        groups={"A": ["core"]},
        linear_constraints=["core >= 0.5"],
        min_weights=0.0,
        max_weights=0.4,
    )
    tr_clone = clone(tr)
    assert tr_clone.max_weights == 0.4
    assert tr_clone.groups == {"A": ["core"]}

    ak = _AdaptiveKMeans(n_clusters=6, random_state=42)
    ak_clone = clone(ak)
    assert ak_clone.n_clusters == 6
    assert ak_clone.random_state == 42

    nco = _FeeAwareNCO(
        inner_estimator=MeanRisk(),
        outer_estimator=MeanRisk(),
        portfolio_params={"management_fees": {"A": 0.001}},
    )
    nco_clone = clone(nco)
    assert nco_clone.portfolio_params == {"management_fees": {"A": 0.001}}
