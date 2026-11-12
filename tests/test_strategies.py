import pytest
from skfolio.optimization import (
    HierarchicalRiskParity,
    MeanRisk,
    NestedClustersOptimization,
)
from skfolio.prior import EmpiricalPrior
from sklearn.base import BaseEstimator, TransformerMixin
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


# def test_add_cross_sectional_raises_type_error():


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
    MockTransformer()
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
    """Test reset() clears pre_selection and cross_sectional queues."""
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
    MockTransformer()
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
