from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from skfolio import Population

logger = logging.getLogger(__name__)


class PersistenceManager:
    """Manages state persistence for portfolio recommendations.

    Provides mechanisms to record, serialise, and reconstruct historical
    allocation advice, enabling version-controlled Git-Ops audit trails.
    Designed for stateless execution within Prefect workflows.
    """

    def __init__(self) -> None:
        pass

    def save_snapshot(self, population: Population, filepath: str) -> None:
        """Serialise a Population's portfolio summary statistics and weights to JSON.

        The JSON structure will contain an ISO8601 timestamp and a list of
        portfolios with their name, tag, weights, sharpe, cvar, and max_drawdown.
        Creates parent directories if they do not exist.

        Raises:
            TypeError: If population is not a skfolio.Population instance.
            ValueError: If population is empty, or if path traversal is detected.
            OSError: If the directory cannot be created.
            RuntimeError: If the snapshot file cannot be written.
        """
        if not isinstance(population, Population):
            raise TypeError("population must be a skfolio.Population instance.")
        if len(population) == 0:
            raise ValueError("population is empty.")

        path = Path(filepath)

        # Prevent path traversal
        if ".." in path.parts:
            raise ValueError(f"Invalid filepath (path traversal detected): {filepath}")

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise OSError(f"Cannot create directory {path.parent}") from e

        portfolios_data = []
        for port in population:
            weights = None
            if hasattr(port, "portfolios") and port.portfolios:
                terminal_port = port.portfolios[-1]
                weights = getattr(terminal_port, "weights_dict", None)
            if weights is None:
                weights = getattr(port, "weights_dict", None) or {}

            portfolios_data.append(
                {
                    "name": port.name,
                    "tag": port.tag,
                    "weights": weights,
                    "sharpe": float(port.sharpe_ratio),
                    "cvar": float(port.cvar),
                    "max_drawdown": float(port.max_drawdown),
                }
            )

        data = {
            "timestamp": datetime.now(UTC).isoformat(),
            "portfolios": portfolios_data,
        }

        try:
            with path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
        except OSError as e:
            raise RuntimeError(f"Failed to save snapshot to {path}") from e

    def load_snapshot(self, filepath: str) -> dict:
        """Load a JSON snapshot dict from the specified filepath.

        Raises:
            FileNotFoundError: If filepath does not exist, with message:
                "Snapshot file not found: {filepath}"
            ValueError: If the file exists but cannot be parsed as valid JSON, with message:
                "Invalid JSON in snapshot file: {filepath}", or if path traversal is detected.
        """
        path = Path(filepath)

        # Prevent path traversal
        if ".." in path.parts:
            raise ValueError(f"Invalid filepath (path traversal detected): {filepath}")

        if not path.exists():
            raise FileNotFoundError(f"Snapshot file not found: {filepath}")

        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except OSError as e:
            raise RuntimeError(f"Failed to read snapshot from {filepath}") from e
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in snapshot file: {filepath}") from e

    def export_gitops_artifact(
        self, population: Population, export_dir: str, label: str | None = None
    ) -> str:
        """Export a Git-Ops artifact for the portfolio population.

        Generates a structured filename using the current ISO week (e.g.,
        rebalance_2026_W33.json) unless a label is provided. Saves the
        snapshot using save_snapshot and returns the absolute path.

        Raises:
            TypeError: If population is not a skfolio.Population instance.
        """
        if not isinstance(population, Population):
            raise TypeError("population must be a skfolio.Population instance.")

        if ".." in Path(export_dir).parts:
            raise ValueError("Path traversal detected in export_dir.")

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
            filename = f"rebalance_{year}_W{week:02d}.json"

        export_path = Path(export_dir) / filename
        self.save_snapshot(population, str(export_path))
        return str(export_path.resolve())

    def calculate_trajectory_history(self, snapshot_dir: str) -> pd.DataFrame:
        """Build a time-series DataFrame of weight recommendations from snapshots.

        Scans all .json files in snapshot_dir matching the rebalance_*.json
        pattern, parses each file, and extracts the timestamp and per-portfolio
        weights.

        Returns:
            pd.DataFrame: With timestamp (DatetimeIndex), portfolio_name, tag,
            and one column per ticker.

        Raises:
            ValueError: If path traversal is detected in snapshot_dir, or if no
                matching snapshot files are found.
            FileNotFoundError: If snapshot_dir does not exist.
        """
        dir_path = Path(snapshot_dir)

        # Prevent path traversal
        if ".." in dir_path.parts:
            raise ValueError("Path traversal detected in snapshot_dir.")

        if not dir_path.exists() or not dir_path.is_dir():
            raise FileNotFoundError(f"Snapshot directory not found: {snapshot_dir}")

        json_files = list(dir_path.glob("rebalance_*.json"))
        if not json_files:
            raise ValueError(f"No rebalance_*.json snapshots found in: {snapshot_dir}")

        records = []
        valid_files_data = []
        timestamps_to_parse = []

        for file in sorted(json_files):
            data = self.load_snapshot(str(file))
            timestamp_str = data.get("timestamp")
            if timestamp_str is None:
                continue

            orig_str = timestamp_str
            # Handle different isoformat styles gracefully
            if timestamp_str.endswith("Z"):
                timestamp_str = timestamp_str[:-1] + "+00:00"

            valid_files_data.append((file, timestamp_str, data, orig_str))
            timestamps_to_parse.append(timestamp_str)

        if timestamps_to_parse:
            parsed_series = pd.to_datetime(timestamps_to_parse, errors="coerce")
            for (file, timestamp_str, data, orig_str), ts in zip(
                valid_files_data, parsed_series
            ):
                if pd.isna(ts):
                    try:
                        pd.to_datetime(timestamp_str)
                    except (ValueError, TypeError) as e:
                        logger.warning(
                            f"Failed to parse timestamp {orig_str} in file {file}: {e}"
                        )
                    continue

                portfolios = data.get("portfolios") or []
                for p_data in portfolios:
                    if not isinstance(p_data, dict):
                        continue
                    record = {
                        "timestamp": ts,
                        "portfolio_name": p_data.get("name"),
                        "tag": p_data.get("tag"),
                    }
                    weights = p_data.get("weights")
                    if isinstance(weights, dict):
                        record.update(weights)
                    records.append(record)

        if not records:
            # Handle edge case where files matched but didn't contain valid data
            df = pd.DataFrame(columns=["timestamp", "portfolio_name", "tag"])
            df.set_index("timestamp", inplace=True)
            return df

        df = pd.DataFrame(records)
        df.set_index("timestamp", inplace=True)
        return df
