import pytest
from skfolio.optimization import (
    HierarchicalRiskParity,
    MeanRisk,
    NestedClustersOptimization,
)
from skfolio.prior import EmpiricalPrior
from sklearn.pipeline import Pipeline

from flowportfolio.strategies import StrategyBuilder


class MockTransformer:
    def fit_transform(self, X):
        return X


def test_init_raises_type_error():
    with pytest.raises(TypeError, match="constraints must be a list"):
        StrategyBuilder(constraints="not a list")


def test_add_pre_selection_raises_type_error():
    builder = StrategyBuilder(constraints=[])
    with pytest.raises(
        TypeError, match="transformer must implement a fit_transform method"
    ):
        builder.add_pre_selection("not a transformer")


def test_add_cross_sectional_raises_type_error():
    builder = StrategyBuilder(constraints=[])
    with pytest.raises(
        TypeError, match="transformer must implement a fit_transform method"
    ):
        builder.add_cross_sectional("not a transformer")


def test_set_optimizer_raises_runtime_error_if_called_twice():
    builder = StrategyBuilder(constraints=[])
    builder.set_optimizer(MeanRisk())
    with pytest.raises(
        RuntimeError, match="set_optimizer called more than once without resetting"
    ):
        builder.set_optimizer(MeanRisk())


def test_set_optimizer_fallback_unsupported_raises_type_error():
    class UnsupportedOptimizer:
        pass

    builder = StrategyBuilder(constraints=[])
    with pytest.raises(
        TypeError, match="The provided optimizer does not support fallback semantics."
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
    cs = MockTransformer()
    opt = MeanRisk()

    builder.add_pre_selection(pre)
    builder.add_cross_sectional(cs)
    builder.set_optimizer(opt)

    pipeline = builder.build_pipeline()
    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.steps) == 3
    assert pipeline.steps[0][0] == "pre_0"
    assert pipeline.steps[1][0] == "cs_0"
    assert pipeline.steps[2][0] == "optimizer"

    # Check prior injection
    assert pipeline.steps[2][1].prior_estimator is prior


def test_build_nco_clustering_dispatch_and_no_mutation():
    builder = StrategyBuilder(constraints=["A >= 0.1"])

    inner = MeanRisk()
    outer = MeanRisk()

    nco = builder.build_nco(inner, outer, clusterer="kmeans")
    assert isinstance(nco, NestedClustersOptimization)

    # Check injection
    assert nco.inner_estimator.linear_constraints == ["A >= 0.1"]

    # Check invalid clusterer
    with pytest.raises(ValueError, match="clusterer must be one of"):
        builder.build_nco(inner, outer, clusterer="invalid")


def test_build_nco_does_not_mutate_estimators():
    builder = StrategyBuilder(constraints=["constraint1"])

    inner = MeanRisk()
    outer = HierarchicalRiskParity()

    nco = builder.build_nco(inner, outer)

    # Original inner should not have constraints
    assert (
        not hasattr(inner, "linear_constraints")
        or inner.linear_constraints is None
        or inner.linear_constraints != ["constraint1"]
    )

    # But NCO's inner should
    assert nco.inner_estimator.linear_constraints == ["constraint1"]
