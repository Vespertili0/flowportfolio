from __future__ import annotations

import json
import logging
import math
import numbers
import os
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from skfolio import Population

from flowportfolio.core.experiment import ExperimentResult

logger = logging.getLogger(__name__)

_TRAJECTORY_RESERVED_COLS: frozenset[str] = frozenset(
    {"timestamp", "portfolio_name", "tag"}
)


def _validate_tz_aware_iso_timestamp(timestamp_str: str, source_file: Path) -> str:
    """Ensure ISO-8601 timestamp includes explicit timezone/UTC info (reject naive)."""
    stripped = timestamp_str.strip()
    try:
        dt = datetime.fromisoformat(stripped)
    except ValueError as exc:
        raise ValueError(
            f"Invalid ISO-8601 timestamp in snapshot file '{source_file}': {exc}"
        ) from exc
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        raise ValueError(
            f"Timezone-naive timestamp is not allowed (explicit 'Z' or UTC offset required) "
            f"in snapshot file '{source_file}': {stripped!r}"
        )
    return stripped


def _build_trajectory_record(
    ts: pd.Timestamp,
    p_data: dict[str, Any],
    warned_collisions: set[str],
) -> dict[str, Any]:
    """Build a single trajectory row while namespacing reserved-column ticker collisions."""
    record: dict[str, Any] = {
        "timestamp": ts,
        "portfolio_name": p_data.get("name"),
        "tag": p_data.get("tag"),
    }
    weights = p_data.get("weights")
    if isinstance(weights, dict):
        for key, val in weights.items():
            col_name = str(key)
            if col_name in _TRAJECTORY_RESERVED_COLS:
                safe_col = f"weight_{col_name}"
                if col_name not in warned_collisions:
                    logger.warning(
                        "Weight ticker '%s' collides with reserved trajectory "
                        "metadata column; storing under '%s'.",
                        col_name,
                        safe_col,
                    )
                    warned_collisions.add(col_name)
                col_name = safe_col
            record[col_name] = val
    return record


def _unpack_population_and_live_weights(
    population: Population | ExperimentResult,
    live_weights: dict[str, dict[str, float]] | None,
) -> tuple[Population, dict[str, dict[str, float]]]:
    """Resolve a ``Population`` and merged live-weights mapping from inputs."""
    result_lw: dict[str, dict[str, float]] | None = None
    if isinstance(population, ExperimentResult):
        pop = population.population
        result_lw = population.live_weights
    else:
        pop = population

    if not isinstance(pop, Population):
        raise TypeError("population must be a skfolio.Population instance.")
    if live_weights is not None and not isinstance(live_weights, dict):
        raise TypeError(
            "live_weights must be a dict mapping strategy tags to weight dicts or None."
        )

    merged: dict[str, dict[str, float]] = {}
    pop_latest = getattr(pop, "latest_weights_by_tag", None)
    if isinstance(pop_latest, dict):
        merged.update(pop_latest)
    if isinstance(result_lw, dict):
        merged.update(result_lw)
    if live_weights is not None:
        merged.update(live_weights)
    return pop, merged


def _cleanup_temp_file(tmp_name: str) -> None:
    """Best-effort removal of a temporary snapshot file after write failure."""
    if os.path.exists(tmp_name):
        try:
            os.remove(tmp_name)
        except OSError:
            pass


def _serialize_portfolio(
    port: Any,
    resolved_live_weights: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Extract weights and summary metrics from a single portfolio object."""
    tag = getattr(port, "tag", None)
    latest_w = resolved_live_weights.get(tag) if tag is not None else None
    weights = dict(latest_w) if isinstance(latest_w, dict) else None
    if weights is None and hasattr(port, "portfolios") and port.portfolios:
        terminal_port = port.portfolios[-1]
        weights = getattr(terminal_port, "weights_dict", None)
    if weights is None:
        weights = getattr(port, "weights_dict", None) or {}

    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe_val = (
            port.annualized_sharpe_ratio
            if hasattr(port, "annualized_sharpe_ratio")
            else port.sharpe_ratio
        )
        cvar_val = float(port.cvar)
        mdd_val = float(port.max_drawdown)

    return {
        "name": port.name,
        "tag": tag,
        "weights": weights,
        "sharpe": float(sharpe_val),
        "cvar": cvar_val,
        "max_drawdown": mdd_val,
    }


class PersistenceManager:
    """Manages state persistence for portfolio recommendations.

    Provides mechanisms to record, serialise, and reconstruct historical
    allocation advice, enabling version-controlled Git-Ops audit trails.

    Parameters
    ----------
    base_dir : str, Path, or None, optional
        Optional root directory boundary. When provided, all resolved snapshot
        and export paths (including symlink targets) must remain strictly
        within ``base_dir``.
    """

    def __init__(self, base_dir: str | Path | None = None) -> None:
        if base_dir is not None and not isinstance(base_dir, (str, Path)):
            raise TypeError("base_dir must be a str, Path, or None.")
        self._base_dir: Path | None = (
            Path(base_dir).resolve() if base_dir is not None else None
        )

    def _validate_path(
        self,
        raw_path: str | Path,
        *,
        traversal_error: str,
        container_dir: Path | None = None,
    ) -> Path:
        """Validate path traversal and canonical symlink containment."""
        path = Path(raw_path)
        if ".." in path.parts:
            raise ValueError(traversal_error)

        resolved = path.resolve()
        if self._base_dir is not None and not resolved.is_relative_to(self._base_dir):
            raise ValueError(
                f"Path '{raw_path}' escapes configured base_dir '{self._base_dir}'."
            )
        if container_dir is not None:
            resolved_container = container_dir.resolve()
            if not resolved.is_relative_to(resolved_container):
                raise ValueError(
                    f"Path '{raw_path}' escapes directory '{resolved_container}'."
                )
        return resolved

    def _sanitize_nans(self, data: Any) -> Any:
        if isinstance(data, dict):
            return {k: self._sanitize_nans(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._sanitize_nans(i) for i in data]
        if isinstance(data, (int, float, numbers.Real)) and not isinstance(data, bool):
            if not math.isfinite(float(data)):
                return None
            return float(data) if not isinstance(data, int) else data
        return data

    def save_snapshot(
        self,
        population: Population | ExperimentResult,
        filepath: str,
        *,
        live_weights: dict[str, dict[str, float]] | None = None,
    ) -> None:
        """Serialise a Population's portfolio summary statistics and weights to JSON.

        The JSON structure contains a UTC ISO-8601 timestamp and a list of
        portfolios with their name, tag, weights, sharpe, cvar, and max_drawdown.
        Creates parent directories if they do not exist.

        Parameters
        ----------
        population : Population or ExperimentResult
            The benchmarked portfolio population or immutable
            :class:`~flowportfolio.core.experiment.ExperimentResult`.
        filepath : str
            Destination JSON file path.
        live_weights : dict[str, dict[str, float]] or None, optional
            Explicit mapping of strategy tag to live terminal weights. Takes
            precedence over ``ExperimentResult.live_weights`` and historical
            Fold-N OOS weights.

        Raises
        ------
        TypeError
            If ``population`` is not a ``Population`` or ``ExperimentResult``,
            or if ``live_weights`` is not a dict or None.
        ValueError
            If ``population`` is empty, or if path traversal / symlink escape
            is detected.
        OSError
            If the directory cannot be created.
        RuntimeError
            If the snapshot file cannot be written.
        """
        pop, resolved_live_weights = _unpack_population_and_live_weights(
            population, live_weights
        )
        if len(pop) == 0:
            raise ValueError("population is empty.")

        path = Path(filepath)
        resolved_path = self._validate_path(
            path,
            traversal_error=f"Invalid filepath (path traversal detected): {filepath}",
        )

        try:
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise OSError(f"Cannot create directory {path.parent}") from e

        data = self._sanitize_nans(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "portfolios": [
                    _serialize_portfolio(port, resolved_live_weights) for port in pop
                ],
            }
        )
        fd: int | None = None
        tmp_name: str | None = None
        try:
            fd, tmp_name = tempfile.mkstemp(dir=resolved_path.parent, text=True)
            f = os.fdopen(fd, "w", encoding="utf-8")
            fd = None
            with f:
                json.dump(data, f, indent=4, allow_nan=False)
            os.replace(tmp_name, resolved_path)
        except OSError as e:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
            if tmp_name is not None:
                _cleanup_temp_file(tmp_name)
            raise RuntimeError(f"Failed to save snapshot to {path}") from e
        except Exception:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
            if tmp_name is not None:
                _cleanup_temp_file(tmp_name)
            raise

    def load_snapshot(self, filepath: str) -> dict[str, Any]:
        """Load a JSON snapshot dict from the specified filepath.

        Raises
        ------
        FileNotFoundError
            If filepath does not exist, with message:
            ``"Snapshot file not found: {filepath}"``.
        ValueError
            If the file exists but cannot be parsed as a valid JSON object,
            or if path traversal / symlink escape is detected.
        """
        resolved_path = self._validate_path(
            filepath,
            traversal_error=f"Invalid filepath (path traversal detected): {filepath}",
        )

        try:
            with resolved_path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Snapshot file not found: {filepath}") from None
        except OSError as e:
            raise RuntimeError(f"Failed to read snapshot from {filepath}") from e
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in snapshot file: {filepath}") from e

        if not isinstance(loaded, dict):
            raise ValueError(  # noqa: TRY004
                f"Snapshot file must contain a JSON object (dict): {filepath}"
            )
        return loaded

    def export_gitops_artifact(
        self,
        population: Population | ExperimentResult,
        export_dir: str,
        label: str | None = None,
        *,
        live_weights: dict[str, dict[str, float]] | None = None,
    ) -> str:
        """Export a Git-Ops artifact for the portfolio population.

        Generates a structured collision-resistant filename using the current
        ISO week, UTC microsecond timestamp, and short random suffix (e.g.,
        ``rebalance_2026_W33_20260812_101500_123456_a1b2c3.json``) unless a
        label is provided. Saves the snapshot using :meth:`save_snapshot` and
        returns the absolute path.

        Raises
        ------
        TypeError
            If ``population`` is not a ``Population`` or ``ExperimentResult``,
            or if ``label`` is not a string or None.
        ValueError
            If path traversal / symlink escape is detected or ``label`` is invalid.
        """
        pop, resolved_live_weights = _unpack_population_and_live_weights(
            population, live_weights
        )

        resolved_dir = self._validate_path(
            export_dir,
            traversal_error="Path traversal detected in export_dir.",
        )

        if label is not None:
            if not isinstance(label, str):
                raise TypeError("label must be a string or None.")
            # Sanitize label to prevent path traversal vulnerabilities
            safe_label = Path(label.replace("\\", "/")).name.strip()
            if not safe_label or safe_label.startswith("."):
                raise ValueError("Invalid label provided for Git-Ops artifact.")
            filename = f"{safe_label}.json"
        else:
            now = datetime.now(UTC)
            year, week, _ = now.isocalendar()
            ts = now.strftime("%Y%m%d_%H%M%S_%f")
            suffix = uuid.uuid4().hex[:6]
            filename = f"rebalance_{year}_W{week:02d}_{ts}_{suffix}.json"

        export_path = Path(export_dir) / filename
        resolved_export = self._validate_path(
            export_path,
            traversal_error="Path traversal detected in export_dir.",
            container_dir=resolved_dir,
        )
        self.save_snapshot(
            pop,
            str(resolved_export),
            live_weights=resolved_live_weights or None,
        )
        return str(resolved_export)

    def calculate_trajectory_history(
        self, snapshot_dir: str, include_custom_labels: bool = False
    ) -> pd.DataFrame:
        """Build a time-series DataFrame of weight recommendations from snapshots.

        Scans all ``.json`` files in ``snapshot_dir`` matching the
        ``rebalance_*.json`` pattern (or ``*.json`` if
        ``include_custom_labels=True``), parses all timestamps strictly via
        timezone-aware ISO-8601 UTC (rejecting naive timestamps), namespaces any
        ticker weight key that collides with reserved metadata columns, and
        fills schema-evolved absent ticker weights with explicit ``0.0``.

        Returns
        -------
        pd.DataFrame
            DataFrame with UTC ``DatetimeIndex`` (``timestamp``),
            ``portfolio_name``, ``tag``, and one float column per ticker.

        Raises
        ------
        ValueError
            If path traversal / symlink escape is detected, if no matching
            snapshot files are found, or if any snapshot has a missing,
            timezone-naive, or unparseable ISO-8601 timestamp.
        FileNotFoundError
            If ``snapshot_dir`` does not exist.
        """
        resolved_dir = self._validate_path(
            snapshot_dir,
            traversal_error="Path traversal detected in snapshot_dir.",
        )

        if not resolved_dir.exists() or not resolved_dir.is_dir():
            raise FileNotFoundError(f"Snapshot directory not found: {snapshot_dir}")

        pattern = "*.json" if include_custom_labels else "rebalance_*.json"
        json_files = sorted(resolved_dir.glob(pattern))
        if not json_files:
            raise ValueError(f"No rebalance_*.json snapshots found in: {snapshot_dir}")

        valid_files_data: list[tuple[Path, dict[str, Any]]] = []
        timestamps_to_parse: list[str] = []

        for file in json_files:
            safe_file = self._validate_path(
                file,
                traversal_error="Path traversal detected in snapshot_dir.",
                container_dir=resolved_dir,
            )
            data = self.load_snapshot(str(safe_file))
            timestamp_str = data.get("timestamp")
            if not isinstance(timestamp_str, str) or not timestamp_str.strip():
                raise ValueError(
                    f"Missing or non-string timestamp in snapshot file: {file}"
                )
            validated_ts = _validate_tz_aware_iso_timestamp(timestamp_str, file)
            valid_files_data.append((file, data))
            timestamps_to_parse.append(validated_ts)

        try:
            parsed_index = pd.to_datetime(
                timestamps_to_parse, format="ISO8601", utc=True
            )
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Failed to parse ISO-8601 timestamp in snapshot directory '{snapshot_dir}': {e}"
            ) from e

        records: list[dict[str, Any]] = []
        warned_collisions: set[str] = set()
        for (file_path, data), ts in zip(valid_files_data, parsed_index, strict=True):
            raw_portfolios = data.get("portfolios")
            if raw_portfolios is None:
                portfolios: list[Any] = []
            elif isinstance(raw_portfolios, list):
                portfolios = raw_portfolios
            else:
                raise ValueError(
                    f"Invalid 'portfolios' field (expected list) in snapshot file: {file_path}"
                )
            for p_data in portfolios:
                if not isinstance(p_data, dict):
                    continue
                records.append(_build_trajectory_record(ts, p_data, warned_collisions))

        if not records:
            return pd.DataFrame(
                columns=["portfolio_name", "tag"],
                index=pd.DatetimeIndex([], tz="UTC", name="timestamp"),
            )

        df = pd.DataFrame(records)
        ticker_cols = [
            col for col in df.columns if col not in _TRAJECTORY_RESERVED_COLS
        ]
        if ticker_cols:
            df[ticker_cols] = df[ticker_cols].fillna(0.0)
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        return df
