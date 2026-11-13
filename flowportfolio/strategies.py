from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
import pandas as pd
from skfolio.optimization import NestedClustersOptimization
from skfolio.prior import BasePrior
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

from flowportfolio.core.constraints import ConstraintSpec

logger = logging.getLogger(__name__)


def _safe_clone(estimator: Any) -> Any:
    """Clone an estimator safely."""
    if estimator is None:
        return None
    return clone(estimator)


class _SafePreSelectionTransformer(BaseEstimator, TransformerMixin):
    """Wrapper around pre-selection transformers preventing infeasible constraint subsets."""

    columns_keep_: np.ndarray | None
    selected_columns_: list[str] | None
    feature_names_in_: np.ndarray
    n_features_in_: int

    def __init__(
        self,
        transformer: TransformerMixin | BaseEstimator,
        groups: dict[str, list[str]] | None = None,
        linear_constraints: list[str] | None = None,
        max_turnover: float | None = None,
        previous_weights: dict[str, float] | None = None,
    ) -> None:
        self.transformer = transformer
        self.groups = groups
        self.linear_constraints = linear_constraints
        self.max_turnover = max_turnover
        self.previous_weights = previous_weights

    def fit(
        self, X: Any, y: Any = None, **fit_params: Any
    ) -> _SafePreSelectionTransformer:
        self.transformer_ = _safe_clone(self.transformer)
        if hasattr(self.transformer_, "set_output"):
            from contextlib import suppress

            with suppress(Exception):
                self.transformer_.set_output(transform="pandas")

        if hasattr(self.transformer_, "fit_transform"):
            res = self.transformer_.fit_transform(X, y=y, **fit_params)
        else:
            self.transformer_.fit(X, y=y, **fit_params)
            res = self.transformer_.transform(X)

        if isinstance(X, pd.DataFrame):
            input_cols = list(X.columns)
            self.feature_names_in_ = np.asarray(input_cols, dtype=object)
            self.n_features_in_ = len(input_cols)

            if isinstance(res, pd.DataFrame):
                surviving = list(res.columns)
            elif hasattr(self.transformer_, "get_support"):
                supp = np.asarray(self.transformer_.get_support(), dtype=bool)
                surviving = [col for col, keep in zip(input_cols, supp) if keep]
            else:
                surviving = list(input_cols)

            # 1. Group Protection
            if self.groups and self.linear_constraints:
                all_known_groups = {g for grps in self.groups.values() for g in grps}
                constrained_groups = set()
                for g in all_known_groups:
                    for lc in self.linear_constraints:
                        if re.search(r"\b" + re.escape(g) + r"\b", lc):
                            constrained_groups.add(g)
                            break

                for g in constrained_groups:
                    group_members = [
                        col
                        for col in input_cols
                        if col in self.groups and g in self.groups[col]
                    ]
                    if len(group_members) >= 1:
                        surviving_group_members = [
                            c for c in surviving if c in group_members
                        ]
                        if not surviving_group_members:
                            restore_col = group_members[0]
                            if restore_col not in surviving:
                                surviving.append(restore_col)
                                logger.warning(
                                    "Pre-selection dropped all assets for constrained group '%s'; "
                                    "restoring asset '%s' to maintain feasibility.",
                                    g,
                                    restore_col,
                                )

            # 2. Turnover Feasibility Protection
            if self.max_turnover is not None and isinstance(
                self.previous_weights, dict
            ):
                prev_weights = self.previous_weights
                max_t = float(self.max_turnover)

                def _calc_max_cap(cols: list[str]) -> float:
                    total_cap: float = 0.0
                    for c in cols:
                        w_val = prev_weights.get(c, 0.0)
                        val = 0.0 if w_val is None else float(w_val)
                        total_cap += min(1.0, val + max_t)
                    return total_cap

                dropped = [c for c in input_cols if c not in surviving]
                dropped.sort(
                    key=lambda c: (
                        0.0 if prev_weights.get(c) is None else float(prev_weights[c])
                    ),
                    reverse=True,
                )

                while _calc_max_cap(surviving) < 1.0 - 1e-9 and dropped:
                    restore_col = dropped.pop(0)
                    surviving.append(restore_col)
                    logger.warning(
                        "Pre-selection produced an infeasible subset under max_turnover=%.4g; "
                        "restoring asset '%s'.",
                        self.max_turnover,
                        restore_col,
                    )

            self.selected_columns_ = [c for c in input_cols if c in surviving]
            self.columns_keep_ = np.asarray(self.selected_columns_, dtype=object)
        else:
            if hasattr(self.transformer_, "feature_names_in_"):
                self.feature_names_in_ = self.transformer_.feature_names_in_
                self.n_features_in_ = len(self.feature_names_in_)
            elif hasattr(X, "shape") and len(X.shape) > 1:
                self.n_features_in_ = X.shape[1]
            if hasattr(self.transformer_, "get_support"):
                supp = self.transformer_.get_support()
                self.columns_keep_ = (
                    np.where(supp)[0]
                    if not isinstance(supp, np.ndarray) or supp.dtype != int
                    else supp
                )
            else:
                self.columns_keep_ = None
            self.selected_columns_ = None

        return self

    def transform(self, X: Any) -> Any:
        if isinstance(X, pd.DataFrame) and self.selected_columns_ is not None:
            return X[self.selected_columns_]
        if hasattr(self, "selected_columns_") and self.selected_columns_ is not None:
            mask = self.get_support(indices=False)
            return X[:, mask]
        return self.transformer_.transform(X)

    def fit_transform(self, X: Any, y: Any = None, **fit_params: Any) -> Any:
        return self.fit(X, y=y, **fit_params).transform(X)

    def get_support(self, indices: bool = False) -> np.ndarray:
        if hasattr(self, "feature_names_in_") and self.selected_columns_ is not None:
            if indices:
                return np.array(
                    [
                        i
                        for i, c in enumerate(self.feature_names_in_)
                        if c in self.selected_columns_
                    ],
                    dtype=int,
                )
            return np.array(
                [c in self.selected_columns_ for c in self.feature_names_in_],
                dtype=bool,
            )
        if hasattr(self, "transformer_") and hasattr(self.transformer_, "get_support"):
            return self.transformer_.get_support(indices=indices)
        raise RuntimeError("Estimator has not been fitted.")


class _AdaptiveKMeans(KMeans):
    """KMeans wrapper that dynamically adapts cluster count without mutating hyperparameter."""

    def fit(self, X: Any, y: Any = None, sample_weight: Any = None) -> _AdaptiveKMeans:
        n_samples = X.shape[0] if hasattr(X, "shape") else len(X)
        effective_k = (
            max(2, min(int(self.n_clusters), n_samples // 2))
            if n_samples >= 4
            else max(1, n_samples - 1)
        )
        internal = KMeans(
            n_clusters=effective_k,
            init=self.init,
            n_init=self.n_init,
            max_iter=self.max_iter,
            tol=self.tol,
            verbose=self.verbose,
            random_state=self.random_state,
            copy_x=self.copy_x,
            algorithm=self.algorithm,
        )
        internal.fit(X, y=y, sample_weight=sample_weight)
        for attr in (
            "cluster_centers_",
            "labels_",
            "inertia_",
            "n_iter_",
            "_n_threads",
            "n_features_in_",
            "feature_names_in_",
        ):
            if hasattr(internal, attr):
                setattr(self, attr, getattr(internal, attr))
        return self


class _FeeAwareNCO(NestedClustersOptimization):
    """NestedClustersOptimization subclass ensuring symmetric fee deduction across clusters."""

    def fit(self, X: Any, y: Any = None, **fit_params: Any) -> _FeeAwareNCO:
        fees_map = None
        if (
            isinstance(self.portfolio_params, dict)
            and "management_fees" in self.portfolio_params
            and isinstance(self.portfolio_params["management_fees"], dict)
        ):
            fees_map = self.portfolio_params["management_fees"]

        if fees_map is not None and isinstance(X, pd.DataFrame):
            X_net = X.copy()
            for col in X_net.columns:
                fee = float(fees_map.get(str(col), 0.0))
                if fee != 0.0:
                    X_net[col] = X_net[col] - fee

            orig_inner = getattr(self, "inner_estimator", None)
            cloned_inner = _safe_clone(orig_inner)
            if cloned_inner is not None and hasattr(cloned_inner, "management_fees"):
                cloned_inner.management_fees = None
            self.inner_estimator = cloned_inner
            try:
                super().fit(X_net, y=y, **fit_params)
            finally:
                self.inner_estimator = orig_inner
            return self

        return super().fit(X, y=y, **fit_params)


class StrategyBuilder:
    """Assembles skfolio-compatible pipeline estimators for portfolio optimisation.

    Acts as a fluent factory for sklearn.pipeline.Pipeline objects and
    NestedClustersOptimization ensembles. Consumes pre-selection transformers,
    cross-sectional scalers, a prior, and an optimizer, then stitches them
    into a single leak-proof estimator.

    Parameters
    ----------
    constraints : list[str], ConstraintSpec, or None, optional
        Linear constraint strings or structured ConstraintSpec generated by
        ConstraintBuilder.build(). Example: ["core >= 0.50", "satellite <= 0.20"].
        Default is None (unconstrained strategy).
    prior : BasePrior or None, optional
        A configured skfolio prior object from PriorSynthesiser.
        If None, the optimizer uses its own internal default prior.

    Raises
    ------
    TypeError
        If constraints is not a list of strings or ConstraintSpec, or if prior is not a BasePrior.
    """

    _constraints: list[str] | ConstraintSpec
    _linear_constraints: list[str]
    _groups: dict[str, list[str]]
    _transaction_costs: dict[str, float] | float | None

    def _apply_constraints(self, value: list[str] | ConstraintSpec | None) -> None:
        if value is None:
            self._constraints = []
            self._linear_constraints = []
            self._groups = {}
            self._previous_weights = None
            self._max_turnover = None
            self._management_fees = None
            self._transaction_costs = None
        elif isinstance(value, ConstraintSpec):
            self._constraints = value
            self._linear_constraints = list(value.linear_constraints)
            self._groups = dict(value.groups)
            self._previous_weights = (
                dict(value.previous_weights)
                if value.previous_weights is not None
                else None
            )
            self._max_turnover = value.max_turnover
            self._management_fees = (
                dict(value.management_fees)
                if value.management_fees is not None
                else None
            )
            self._transaction_costs = (
                dict(value.transaction_costs)
                if isinstance(value.transaction_costs, dict)
                else value.transaction_costs
            )
        elif isinstance(value, list):
            if not all(isinstance(c, str) for c in value):
                raise TypeError("All constraints must be strings.")
            self._constraints = list(value)
            self._linear_constraints = list(value)
            self._groups = {}
            self._previous_weights = None
            self._max_turnover = None
            self._management_fees = None
            self._transaction_costs = None
        else:
            raise TypeError("constraints must be a list or ConstraintSpec.")

    @property
    def constraints(self) -> list[str] | ConstraintSpec:
        """Accumulated strategy constraints."""
        return self._constraints

    @constraints.setter
    def constraints(self, value: list[str] | ConstraintSpec | None) -> None:
        self._apply_constraints(value)
        # Re-wire from the clean template so that earlier constraints do not
        # accumulate on an already-injected optimizer.
        if getattr(self, "_raw_optimizer", None) is not None:
            self._optimizer = self._wire_from_raw(include_prior=False)

    def __init__(
        self,
        constraints: list[str] | ConstraintSpec | None = None,
        prior: BasePrior | None = None,
    ) -> None:
        self._apply_constraints(constraints)

        if prior is not None and not isinstance(prior, BasePrior):
            raise TypeError("prior must be an instance of BasePrior or None.")

        self.prior = prior
        self._pre_selection: list = []
        self._optimizer = None
        self._raw_optimizer = None
        self._raw_fallbacks: list = []
        self._fallback = None
        self._fallbacks: list = []

    def add_pre_selection(
        self, transformer: TransformerMixin | BaseEstimator
    ) -> StrategyBuilder:
        """Add a pre-selection transformer to the pipeline queue.

        Pre-selection transformers dynamically filter the asset universe
        before the optimizer sees it. They execute before cross-sectional
        transformers in the assembled pipeline.

        Parameters
        ----------
        transformer : skfolio-compatible transformer
            Must implement fit_transform(X). Examples: DropCorrelated,
            SelectNonDominated, SelectComplete.

        Returns
        -------
        StrategyBuilder
            Returns self for method chaining.

        Raises
        ------
        TypeError
            If transformer does not implement fit_transform.
        """
        if not callable(getattr(transformer, "fit_transform", None)):
            raise TypeError("transformer must implement a fit_transform method.")

        self._pre_selection.append(transformer)
        return self

    def _inject_into_estimator(self, opt, include_prior: bool = True) -> None:
        """Inject accumulated constraints (and optionally the prior) into ``opt``.

        Recurses into ``opt.fallback`` when present so the whole fallback
        chain shares the same constraint configuration.

        Parameters
        ----------
        opt : skfolio optimizer or None
            The estimator to mutate in place. ``None`` is ignored.
        include_prior : bool, default True
            Whether to clone ``self.prior`` into ``opt.prior_estimator``.
        """
        if opt is None:
            return
        if self._linear_constraints and hasattr(opt, "linear_constraints"):
            existing = getattr(opt, "linear_constraints", None) or []
            merged = list(existing) + [
                c for c in self._linear_constraints if c not in existing
            ]
            opt.linear_constraints = merged

        if self._groups and hasattr(opt, "groups"):
            existing_groups = getattr(opt, "groups", None) or {}
            merged_groups = dict(existing_groups)
            merged_groups.update(self._groups)
            opt.groups = merged_groups

        if hasattr(opt, "previous_weights") and self._previous_weights is not None:
            opt.previous_weights = dict(self._previous_weights)
        if hasattr(opt, "max_turnover") and self._max_turnover is not None:
            opt.max_turnover = self._max_turnover
        if hasattr(opt, "management_fees") and self._management_fees is not None:
            opt.management_fees = dict(self._management_fees)

        if include_prior and self.prior is not None and hasattr(opt, "prior_estimator"):
            opt.prior_estimator = clone(self.prior)

        if hasattr(opt, "transaction_costs") and self._transaction_costs is not None:
            opt.transaction_costs = (
                dict(self._transaction_costs)
                if isinstance(self._transaction_costs, dict)
                else self._transaction_costs
            )

        if self._linear_constraints and not hasattr(opt, "linear_constraints"):
            logger.warning(
                "Estimator '%s' does not support linear_constraints; "
                "group/linear constraints will be ignored by this estimator.",
                type(opt).__name__,
            )

        if self._max_turnover is not None and not hasattr(opt, "max_turnover"):
            logger.warning(
                "Estimator '%s' does not support max_turnover; "
                "turnover constraints will be ignored by this estimator.",
                type(opt).__name__,
            )

        if hasattr(opt, "fallback") and opt.fallback is not None:
            self._inject_into_estimator(opt.fallback, include_prior=include_prior)

    def _wire_from_raw(self, include_prior: bool):
        """Clone the clean optimizer template, wire fallbacks and inject config.

        Parameters
        ----------
        include_prior : bool
            Whether to inject a clone of ``self.prior`` into each estimator.

        Returns
        -------
        skfolio optimizer
            A freshly cloned and fully configured optimizer chain.
        """
        wired = _safe_clone(self._raw_optimizer)
        current = wired
        for fb in self._raw_fallbacks:
            cloned_fb = _safe_clone(fb)
            current.fallback = cloned_fb
            current = cloned_fb
        self._inject_into_estimator(wired, include_prior=include_prior)
        return wired

    def set_optimizer(
        self,
        optimizer,
        fallback=None,
        fallbacks: list | None = None,
    ) -> StrategyBuilder:
        """Configure the terminal optimizer for the pipeline.

        Parameters
        ----------
        optimizer : skfolio optimizer
            The primary portfolio optimizer (e.g., MeanRisk, RiskBudgeting,
            HierarchicalRiskParity).
        fallback : skfolio optimizer or None, optional
            A single fallback estimator to attempt if the primary
            optimizer fails.
        fallbacks : list or None, optional
            Ordered list of fallback optimizers.

        Returns
        -------
        StrategyBuilder
            Returns self for method chaining.

        Raises
        ------
        RuntimeError
            If set_optimizer() is called more than once without reset.
        """
        if self._optimizer is not None:
            raise RuntimeError(
                "set_optimizer() has already been called. "
                "Call .reset() to clear the pipeline configuration before reconfiguring."
            )
        if optimizer is None:
            raise TypeError("optimizer must not be None.")

        fbs = []
        if fallback is not None:
            fbs.append(fallback)
        if fallbacks is not None:
            fbs.extend(fallbacks)

        raw_optimizer = _safe_clone(optimizer)
        raw_fallbacks = [_safe_clone(f) for f in fbs]

        current = raw_optimizer
        for fb in raw_fallbacks:
            if current is None or not hasattr(current, "fallback"):
                raise TypeError(
                    "The provided optimizer does not support fallback semantics."
                )
            current = fb

        # Keep clean templates; the wired working copy is rebuilt from them
        # whenever the constraints are reassigned or the pipeline is built.
        self._raw_optimizer = raw_optimizer
        self._raw_fallbacks = raw_fallbacks
        self._fallbacks = list(raw_fallbacks)
        self._optimizer = self._wire_from_raw(include_prior=False)

        return self

    def reset(self, clear_base_config: bool = False) -> StrategyBuilder:
        """Reset the builder pipeline configuration.

        Clears the transformer queues and optimizer, allowing the builder
        to be reconfigured with new pipeline components. By default, the
        foundational ``constraints`` and ``prior`` supplied at construction
        are retained.

        Parameters
        ----------
        clear_base_config : bool, default False
            If ``True``, also resets ``self.constraints`` to ``[]`` and
            ``self.prior`` to ``None`` in addition to clearing the pipeline
            staging queues.

        Returns
        -------
        StrategyBuilder
            The builder instance for method chaining.
        """
        self._pre_selection = []
        self._optimizer = None
        self._raw_optimizer = None
        self._raw_fallbacks = []
        self._fallback = None
        self._fallbacks = []
        if clear_base_config:
            self.constraints = []
            self.prior = None
        return self

    def build_pipeline(self) -> Pipeline:
        """Assemble all queued transformers and the optimizer into a Pipeline.

        Assembly order (strict):
        1. All pre-selection transformers (in registration order).
        2. All cross-sectional transformers (in registration order).
        3. The configured optimizer (with prior injected if provided).

        Returns
        -------
        sklearn.pipeline.Pipeline

        Raises
        ------
        RuntimeError
            If no optimizer has been set via set_optimizer().
        """
        if self._optimizer is None:
            raise RuntimeError(
                "No optimizer set. Call set_optimizer() before build_pipeline()."
            )

        has_constraints = bool(
            self._linear_constraints or self._max_turnover is not None
        )
        steps = []
        for i, transformer in enumerate(self._pre_selection):
            cloned_tr = _safe_clone(transformer)
            if has_constraints:
                tr_step = _SafePreSelectionTransformer(
                    transformer=cloned_tr,
                    groups=self._groups,
                    linear_constraints=self._linear_constraints,
                    max_turnover=self._max_turnover,
                    previous_weights=self._previous_weights,
                )
            else:
                tr_step = cloned_tr
                if hasattr(tr_step, "set_output"):
                    from contextlib import suppress

                    with suppress(Exception):
                        tr_step.set_output(transform="pandas")
            steps.append((f"pre_{i}", tr_step))

        cloned_optimizer = self._wire_from_raw(include_prior=True)

        steps.append(("optimizer", cloned_optimizer))

        return Pipeline(steps)

    def build_nco(
        self,
        inner_estimator,
        outer_estimator,
        clusterer: str = "ward",
        random_state: int | None = None,
    ) -> NestedClustersOptimization:
        """Assemble a Nested Clusters Optimisation ensemble.

        Linear group constraints and turnover limits cannot be enforced
        inside isolated sub-clusters and are rejected. Daily management fees
        and a compatible prior (e.g. EmpiricalPrior, SyntheticData) are
        applied to the inner estimator only; the outer estimator receives
        neither, as it operates on synthetic cluster return series.

        Parameters
        ----------
        inner_estimator : skfolio optimizer
            Optimizes weights within individual clusters.
        outer_estimator : skfolio optimizer
            Allocates capital across clusters using out-of-sample
            cross-validated cluster performance.
        clusterer : str, optional
            Clustering algorithm. One of: "ward", "complete", "single",
            "average", "kmeans". Default is "ward".

        Returns
        -------
        skfolio.optimization.NestedClustersOptimization

        Raises
        ------
        TypeError
            If inner_estimator or outer_estimator is None.
        ValueError
            If clusterer is not one of the permitted values, if linear group
            constraints or turnover limits have been configured (they cannot
            be enforced inside isolated sub-clusters), or if the prior is a
            ticker-level view prior (BlackLitterman or EntropyPooling).
        """
        if inner_estimator is None:
            raise TypeError("inner_estimator must not be None.")
        if outer_estimator is None:
            raise TypeError("outer_estimator must not be None.")

        permitted_clusterers = ["ward", "complete", "single", "average", "kmeans"]
        if not isinstance(clusterer, str) or clusterer not in permitted_clusterers:
            raise ValueError(f"clusterer must be one of {permitted_clusterers}")

        if (
            self._linear_constraints
            or self._max_turnover is not None
            or self._previous_weights is not None
        ):
            raise ValueError(
                "Cross-asset linear constraints and turnover limits cannot be enforced "
                "inside isolated NestedClustersOptimization sub-clusters."
            )

        from skfolio.prior import BlackLitterman, EntropyPooling

        if self.prior is not None and isinstance(
            self.prior, (BlackLitterman, EntropyPooling)
        ):
            raise ValueError(
                "Ticker-level view priors (BlackLitterman, EntropyPooling) cannot be "
                "applied to NestedClustersOptimization because sub-clusters and outer "
                "cluster allocations do not preserve the full ticker universe."
            )

        cloned_inner = _safe_clone(inner_estimator)
        cloned_outer = _safe_clone(outer_estimator)

        if self._management_fees is not None and hasattr(
            cloned_inner, "management_fees"
        ):
            cloned_inner.management_fees = dict(self._management_fees)

        if self.prior is not None and hasattr(cloned_inner, "prior_estimator"):
            cloned_inner.prior_estimator = clone(self.prior)

        if clusterer == "kmeans":
            clustering_estimator = _AdaptiveKMeans(random_state=random_state)
        else:
            from skfolio.cluster import HierarchicalClustering, LinkageMethod

            clustering_estimator = HierarchicalClustering(
                linkage_method=LinkageMethod(clusterer)
            )

        nco_kwargs: dict[str, object] = {
            "inner_estimator": cloned_inner,
            "outer_estimator": cloned_outer,
            "clustering_estimator": clustering_estimator,
        }
        if self._management_fees is not None:
            # Fees are never injected into the outer estimator (it operates on
            # cluster-level series); they are applied when evaluating the
            # resulting out-of-sample Portfolio.
            nco_kwargs["portfolio_params"] = {
                "management_fees": dict(self._management_fees)
            }

        return _FeeAwareNCO(**nco_kwargs)
