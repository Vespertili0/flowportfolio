from __future__ import annotations

import logging
import re
from collections.abc import Sequence
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

_SIMPLE_SUM_CONSTRAINT_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_.\-]*(?:\s*\+\s*[A-Za-z_][A-Za-z0-9_.\-]*)*)"
    r"\s*(<=|>=|==)\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$"
)


def _safe_clone(estimator: Any) -> Any:
    """Clone an estimator safely."""
    if estimator is None:
        return None
    return clone(estimator)


def _resolve_weight_bound(
    bound: Any, all_cols: list[str], default: float
) -> dict[str, float]:
    """Resolve scalar, dict, or positional weight bounds across ``all_cols``."""
    if bound is None:
        return {c: default for c in all_cols}
    if isinstance(bound, (int, float)) and not isinstance(bound, bool):
        val = float(bound)
        return {c: val for c in all_cols}
    if isinstance(bound, dict):
        return {
            c: float(bound[c]) if c in bound and bound[c] is not None else default
            for c in all_cols
        }
    if isinstance(bound, (list, tuple, np.ndarray)):
        arr = np.asarray(bound, dtype=float)
        if arr.ndim == 1 and len(arr) == len(all_cols):
            return {c: float(arr[i]) for i, c in enumerate(all_cols)}
    return {c: default for c in all_cols}


def _effective_asset_bounds(
    all_cols: list[str],
    *,
    min_weights: Any = None,
    max_weights: Any = None,
    previous_weights: dict[str, float] | None = None,
    max_turnover: float | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute effective per-asset ``(lower, upper)`` weight bounds."""
    lbs = _resolve_weight_bound(min_weights, all_cols, 0.0)
    ubs = _resolve_weight_bound(max_weights, all_cols, 1.0)
    if max_turnover is not None and isinstance(previous_weights, dict):
        max_t = float(max_turnover)
        for col in all_cols:
            w_raw = previous_weights.get(col, 0.0)
            w_prev = 0.0 if w_raw is None else float(w_raw)
            ubs[col] = min(ubs[col], w_prev + max_t)
            lbs[col] = max(lbs[col], w_prev - max_t)
    for col in all_cols:
        ubs[col] = max(0.0, ubs[col])
        lbs[col] = max(0.0, lbs[col])
    return lbs, ubs


def _restore_feasible_subset(
    surviving: list[str],
    all_cols: list[str],
    *,
    groups: dict[str, list[str]] | None = None,
    linear_constraints: list[str] | None = None,
    min_weights: Any = None,
    max_weights: Any = None,
    previous_weights: dict[str, float] | None = None,
    max_turnover: float | None = None,
) -> tuple[list[str], list[str]]:
    """Statelessly restore dropped assets needed for constraint feasibility.

    Ensures the surviving asset subset satisfies:
    1. Group and asset-level linear lower-bound capacity (respecting ``max_weights``
       and ``max_turnover`` upper bounds).
    2. Complement upper-bound capacity for linear upper-bound constraints
       (e.g. ``satellite <= 0.30`` requires non-satellite capacity ``>= 0.70``).
    3. At least one surviving member per referenced group/ticker so ``skfolio``'s
       constraint parser does not fail on missing identifiers.
    4. Global simplex upper-bound capacity ``sum(ub_i) >= 1.0``.

    Returns
    -------
    tuple[list[str], list[str]]
        ``(restored_columns, reasons)`` with columns ordered as in ``all_cols``.
    """
    _, ubs = _effective_asset_bounds(
        all_cols,
        min_weights=min_weights,
        max_weights=max_weights,
        previous_weights=previous_weights,
        max_turnover=max_turnover,
    )
    all_cols_set = set(all_cols)
    surviving_set = {c for c in surviving if c in all_cols_set}
    reasons: list[str] = []

    def _members_for_term(term: str) -> list[str]:
        if term in all_cols_set:
            return [term]
        if groups:
            return [c for c in all_cols if c in groups and term in groups[c]]
        return []

    def _restore_until_capacity(pool: list[str], required: float, label: str) -> None:
        pool_set = set(pool)
        cap = sum(ubs[c] for c in all_cols if c in surviving_set and c in pool_set)
        dropped = [c for c in pool if c not in surviving_set]
        dropped.sort(key=lambda c: ubs[c], reverse=True)
        while cap < required - 1e-9 and dropped:
            col = dropped.pop(0)
            surviving_set.add(col)
            cap += ubs[col]
            reasons.append(
                f"Restored asset '{col}' to satisfy {label} (capacity={cap:.4g} >= {required:.4g})."
            )

    def _ensure_at_least_one(members: list[str], label: str) -> None:
        if not members:
            return
        if any(c in surviving_set for c in members):
            return
        best = max(members, key=lambda c: ubs[c])
        surviving_set.add(best)
        reasons.append(
            f"Restored asset '{best}' because pre-selection dropped all members of {label}."
        )

    # 1. Linear constraints (group-level and asset-level)
    if linear_constraints:
        all_known_groups = (
            {g for grps in groups.values() for g in grps} if groups else set()
        )
        for lc in linear_constraints:
            match = _SIMPLE_SUM_CONSTRAINT_RE.match(lc)
            if match:
                terms = [t.strip() for t in match.group(1).split("+")]
                op = match.group(2)
                rhs = float(match.group(3))
                member_set: set[str] = set()
                for term in terms:
                    term_members = _members_for_term(term)
                    _ensure_at_least_one(term_members, f"'{term}'")
                    member_set.update(term_members)
                members = [c for c in all_cols if c in member_set]
                if op in (">=", "==") and rhs > 0.0 and members:
                    _restore_until_capacity(members, rhs, f"linear constraint '{lc}'")
                if op in ("<=", "==") and rhs < 1.0 and members:
                    complement = [c for c in all_cols if c not in member_set]
                    _restore_until_capacity(
                        complement, 1.0 - rhs, f"complement of '{lc}'"
                    )
            else:
                for g in sorted(all_known_groups):
                    if re.search(r"\b" + re.escape(g) + r"\b", lc):
                        _ensure_at_least_one(_members_for_term(g), f"group '{g}'")
                for col in all_cols:
                    if re.search(
                        r"(?<![A-Za-z0-9_.\-])"
                        + re.escape(col)
                        + r"(?![A-Za-z0-9_.\-])",
                        lc,
                    ):
                        _ensure_at_least_one([col], f"asset '{col}'")

    # 2. Global simplex upper-bound feasibility (max_turnover & max_weights)
    prev_map = previous_weights if isinstance(previous_weights, dict) else {}
    dropped_all = [c for c in all_cols if c not in surviving_set]
    dropped_all.sort(
        key=lambda c: (
            ubs[c],
            0.0 if prev_map.get(c) is None else float(prev_map[c]),
        ),
        reverse=True,
    )
    total_cap = sum(ubs[c] for c in all_cols if c in surviving_set)
    while total_cap < 1.0 - 1e-9 and dropped_all:
        col = dropped_all.pop(0)
        surviving_set.add(col)
        total_cap += ubs[col]
        reasons.append(
            f"Restored asset '{col}' to maintain global simplex feasibility "
            f"(capacity={total_cap:.4g} >= 1.0)."
        )

    restored = [c for c in all_cols if c in surviving_set]
    return restored, reasons


def _resolve_fee_vector(
    management_fees: Any,
    n_assets: int,
    feature_names: Sequence[str] | None = None,
) -> np.ndarray:
    """Convert scalar, dict, or array ``management_fees`` into a 1D float array."""
    if isinstance(management_fees, (int, float)) and not isinstance(
        management_fees, bool
    ):
        return np.full(n_assets, float(management_fees), dtype=float)
    if isinstance(management_fees, dict):
        if feature_names is not None:
            return np.array(
                [float(management_fees.get(str(c), 0.0)) for c in feature_names],
                dtype=float,
            )
        raise ValueError(
            "management_fees dict requires a pandas DataFrame or fitted feature names "
            "to align per-asset fees unambiguously without guessing column order."
        )
    arr = np.asarray(management_fees, dtype=float)
    if arr.ndim != 1 or len(arr) != n_assets:
        raise ValueError(
            f"management_fees length ({arr.shape}) does not match "
            f"number of columns in X ({n_assets})."
        )
    return arr


def _subtract_management_fees(
    X: Any,
    management_fees: Any,
    feature_names: Sequence[str] | None = None,
) -> Any:
    """Return a copy of ``X`` with per-period ``management_fees`` subtracted."""
    if management_fees is None:
        return X
    if isinstance(X, pd.DataFrame):
        fee_vec = _resolve_fee_vector(
            management_fees,
            X.shape[1],
            feature_names=[str(c) for c in X.columns],
        )
        X_net = X.copy()
        if np.any(fee_vec != 0.0):
            X_net = X_net.sub(fee_vec, axis=1)
        return X_net
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D returns array for X, got shape {arr.shape}.")
    fee_vec = _resolve_fee_vector(
        management_fees, arr.shape[1], feature_names=feature_names
    )
    if np.any(fee_vec != 0.0):
        return arr - fee_vec
    return arr.copy()


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
        min_weights: Any = None,
        max_weights: Any = None,
    ) -> None:
        self.transformer = transformer
        self.groups = groups
        self.linear_constraints = linear_constraints
        self.max_turnover = max_turnover
        self.previous_weights = previous_weights
        self.min_weights = min_weights
        self.max_weights = max_weights

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

            restored, reasons = _restore_feasible_subset(
                surviving=surviving,
                all_cols=input_cols,
                groups=self.groups,
                linear_constraints=self.linear_constraints,
                min_weights=self.min_weights,
                max_weights=self.max_weights,
                previous_weights=self.previous_weights,
                max_turnover=self.max_turnover,
            )
            for reason in reasons:
                logger.warning("Pre-selection feasibility guard: %s", reason)

            self.selected_columns_ = restored
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

    effective_n_clusters_: int

    def fit(self, X: Any, y: Any = None, sample_weight: Any = None) -> _AdaptiveKMeans:
        n_samples = X.shape[0] if hasattr(X, "shape") else len(X)
        if n_samples < 1:
            raise ValueError("X must contain at least 1 sample.")
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
        self.effective_n_clusters_ = effective_k
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


def _copy_fitted_attrs(src: object, dst: object) -> None:
    """Copy scikit-learn fitted attributes (trailing underscore) from ``src`` to ``dst``."""
    for key, value in src.__dict__.items():
        if key.endswith("_") and not key.startswith("__"):
            setattr(dst, key, value)


class _FeeAwareNCO(NestedClustersOptimization):
    """NestedClustersOptimization subclass ensuring symmetric fee deduction across clusters."""

    def _build_delegate(
        self,
        *,
        inner_estimator: Any,
        portfolio_params: dict[str, Any] | None,
    ) -> NestedClustersOptimization:
        return NestedClustersOptimization(
            inner_estimator=inner_estimator,
            outer_estimator=self.outer_estimator,
            distance_estimator=self.distance_estimator,
            clustering_estimator=self.clustering_estimator,
            cv=self.cv,
            quantile=self.quantile,
            quantile_measure=self.quantile_measure,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            portfolio_params=portfolio_params,
            fallback=self.fallback,
            previous_weights=self.previous_weights,
            raise_on_failure=self.raise_on_failure,
        )

    def fit(self, X: Any, y: Any = None, **fit_params: Any) -> _FeeAwareNCO:
        fees_spec = (
            self.portfolio_params.get("management_fees")
            if isinstance(self.portfolio_params, dict)
            else None
        )

        cloned_inner = _safe_clone(self.inner_estimator)
        if fees_spec is not None:
            X_net = _subtract_management_fees(X, fees_spec)
            if cloned_inner is not None and hasattr(cloned_inner, "management_fees"):
                cloned_inner.management_fees = None
        else:
            X_net = X

        delegate = self._build_delegate(
            inner_estimator=cloned_inner,
            portfolio_params=self.portfolio_params,
        )
        delegate.fit(X_net, y=y, **fit_params)
        _copy_fitted_attrs(delegate, self)
        return self

    def predict(self, X: Any) -> Any:
        if (
            isinstance(self.portfolio_params, dict)
            and isinstance(self.portfolio_params.get("management_fees"), dict)
            and not isinstance(X, pd.DataFrame)
        ):
            from skfolio.prior import ReturnDistribution

            if isinstance(X, ReturnDistribution):
                if hasattr(self, "feature_names_in_"):
                    return super().predict(X)
                n_assets = X.returns.shape[1]
            else:
                arr = np.asarray(X)
                if arr.ndim != 2:
                    raise ValueError(
                        f"Expected 2D returns array for X, got shape {arr.shape}."
                    )
                n_assets = arr.shape[1]
                if hasattr(self, "feature_names_in_"):
                    # Convert array back to DataFrame with fitted asset names so
                    # dict management_fees align by column without sklearn warnings.
                    X_df = pd.DataFrame(arr, columns=list(self.feature_names_in_))
                    return super().predict(X_df)

            fee_vec = _resolve_fee_vector(
                self.portfolio_params["management_fees"],
                n_assets,
            )
            adjusted_params = dict(self.portfolio_params)
            adjusted_params["management_fees"] = fee_vec
            delegate = self._build_delegate(
                inner_estimator=self.inner_estimator,
                portfolio_params=adjusted_params,
            )
            _copy_fitted_attrs(self, delegate)
            return delegate.predict(X)

        return super().predict(X)


class StrategyBuilder:
    """Assembles skfolio-compatible pipeline estimators for portfolio optimisation.

    Acts as a fluent factory for sklearn.pipeline.Pipeline objects and
    NestedClustersOptimization ensembles. Consumes pre-selection transformers,
    a prior, and an optimizer, then stitches them into a single leak-proof
    estimator.

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
    _holding_period: float | None
    _tc_from_spec: bool

    def _apply_constraints(self, value: list[str] | ConstraintSpec | None) -> None:
        if value is None:
            self._constraints = []
            self._linear_constraints = []
            self._groups = {}
            self._previous_weights = None
            self._max_turnover = None
            self._management_fees = None
            self._transaction_costs = None
            self._holding_period = None
            self._tc_from_spec = False
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
            self._holding_period = value.holding_period
            self._tc_from_spec = value.transaction_costs is not None
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
            self._holding_period = None
            self._tc_from_spec = False
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
        self._fallbacks: list = []

    def add_pre_selection(
        self, transformer: TransformerMixin | BaseEstimator
    ) -> StrategyBuilder:
        """Add a pre-selection transformer to the pipeline queue.

        Pre-selection transformers dynamically filter the asset universe
        before the optimizer sees it.

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

        Clears the pre-selection queue and optimizer, allowing the builder
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
        self._fallbacks = []
        if clear_base_config:
            self.constraints = []
            self.prior = None
        return self

    def build_pipeline(self) -> Pipeline:
        """Assemble all queued pre-selection transformers and the optimizer into a Pipeline.

        Assembly order (strict):
        1. All pre-selection transformers (in registration order).
        2. The configured optimizer (with prior injected if provided).

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

        cloned_optimizer = self._wire_from_raw(include_prior=True)
        opt_min_w = getattr(cloned_optimizer, "min_weights", None)
        opt_max_w = getattr(cloned_optimizer, "max_weights", None)

        has_weight_bounds = (
            opt_min_w is not None
            and not (isinstance(opt_min_w, (int, float)) and opt_min_w == 0.0)
        ) or (
            opt_max_w is not None
            and not (isinstance(opt_max_w, (int, float)) and opt_max_w == 1.0)
        )
        needs_safe_wrapper = bool(
            self._linear_constraints
            or self._max_turnover is not None
            or isinstance(self._management_fees, dict)
            or isinstance(self._transaction_costs, dict)
            or isinstance(self._previous_weights, dict)
            or has_weight_bounds
        )
        steps = []
        for i, transformer in enumerate(self._pre_selection):
            cloned_tr = _safe_clone(transformer)
            if needs_safe_wrapper:
                tr_step = _SafePreSelectionTransformer(
                    transformer=cloned_tr,
                    groups=self._groups,
                    linear_constraints=self._linear_constraints,
                    max_turnover=self._max_turnover,
                    previous_weights=self._previous_weights,
                    min_weights=opt_min_w,
                    max_weights=opt_max_w,
                )
            else:
                tr_step = cloned_tr
                if hasattr(tr_step, "set_output"):
                    from contextlib import suppress

                    with suppress(Exception):
                        tr_step.set_output(transform="pandas")
            steps.append((f"pre_{i}", tr_step))

        steps.append(("optimizer", cloned_optimizer))

        pipeline = Pipeline(steps)
        if self._tc_from_spec:
            pipeline._flowportfolio_holding_period = self._holding_period
            pipeline._flowportfolio_tc_from_spec = True
        return pipeline

    def build_nco(
        self,
        inner_estimator,
        outer_estimator,
        clusterer: str = "ward",
        random_state: int | None = 42,
    ) -> NestedClustersOptimization:
        """Assemble a Nested Clusters Optimisation ensemble.

        Linear group constraints and turnover limits cannot be enforced
        inside isolated sub-clusters and are rejected. Period-scaled management
        fees and a compatible prior (e.g. EmpiricalPrior, SyntheticData) are
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
        random_state : int or None, optional
            Non-negative integer seed for ``kmeans`` clustering reproducibility,
            or ``None`` for unseeded clustering. Default is ``42``.

        Returns
        -------
        skfolio.optimization.NestedClustersOptimization

        Raises
        ------
        TypeError
            If inner_estimator or outer_estimator is None.
        ValueError
            If clusterer is not one of the permitted values, if ``random_state``
            is invalid, if linear group constraints or turnover limits have been
            configured (they cannot be enforced inside isolated sub-clusters),
            or if the prior is a ticker-level view prior (BlackLitterman or
            EntropyPooling).
        """
        if inner_estimator is None:
            raise TypeError("inner_estimator must not be None.")
        if outer_estimator is None:
            raise TypeError("outer_estimator must not be None.")

        permitted_clusterers = ["ward", "complete", "single", "average", "kmeans"]
        if not isinstance(clusterer, str) or clusterer not in permitted_clusterers:
            raise ValueError(f"clusterer must be one of {permitted_clusterers}")

        if random_state is not None and (
            not isinstance(random_state, int)
            or isinstance(random_state, bool)
            or random_state < 0
        ):
            raise ValueError(
                f"random_state must be a non-negative integer or None, got {random_state!r}."
            )

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
