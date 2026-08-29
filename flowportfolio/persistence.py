from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from skfolio import Population


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
            ValueError: If population is empty.
            OSError: If the directory cannot be created.
        """
        if not isinstance(population, Population):
            raise TypeError("population must be a skfolio.Population instance.")
        if len(population) == 0:
            raise ValueError("population is empty.")

        path = Path(filepath)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise OSError(f"Cannot create directory {path.parent}") from e

        portfolios_data = []
        for port in population:
            portfolios_data.append(
                {
                    "name": port.name,
                    "tag": port.tag,
                    "weights": port.weights_dict,
                    "sharpe": float(port.sharpe_ratio),
                    "cvar": float(port.cvar),
                    "max_drawdown": float(port.max_drawdown),
                }
            )

        data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "portfolios": portfolios_data,
        }

        try:
            with path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
        except OSError as e:
            raise OSError(f"Cannot write file to {path}") from e

    def load_snapshot(self, filepath: str) -> dict:
        """Load a JSON snapshot dict from the specified filepath.

        Raises:
            FileNotFoundError: If filepath does not exist, with message:
                "Snapshot file not found: {filepath}"
            ValueError: If the file exists but cannot be parsed as valid JSON, with message:
                "Invalid JSON in snapshot file: {filepath}"
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Snapshot file not found: {filepath}")

        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
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

        if label:
            # Sanitize label to prevent path traversal vulnerabilities
            safe_label = Path(label).name
            filename = f"{safe_label}.json"
        else:
            now = datetime.now(timezone.utc)
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
            FileNotFoundError: If snapshot_dir does not exist.
            ValueError: If no matching snapshot files are found, with message:
                "No rebalance_*.json snapshots found in: {snapshot_dir}"
        """
        dir_path = Path(snapshot_dir)
        if not dir_path.exists() or not dir_path.is_dir():
            raise FileNotFoundError(f"Snapshot directory not found: {snapshot_dir}")

        json_files = list(dir_path.glob("rebalance_*.json"))
        if not json_files:
            raise ValueError(f"No rebalance_*.json snapshots found in: {snapshot_dir}")

        records = []
        for file in sorted(json_files):
            data = self.load_snapshot(str(file))
            timestamp_str = data.get("timestamp")
            if timestamp_str is None:
                continue

            try:
                # Handle different isoformat styles gracefully
                if timestamp_str.endswith("Z"):
                    timestamp_str = timestamp_str[:-1] + "+00:00"
                ts = pd.to_datetime(timestamp_str)
            except Exception as e:
                import warnings
                warnings.warn(f"Failed to parse timestamp {timestamp_str} in file {file}: {e}")
                continue

            for p_data in data.get("portfolios", []):
                record = {
                    "timestamp": ts,
                    "portfolio_name": p_data.get("name"),
                    "tag": p_data.get("tag"),
                }
                weights = p_data.get("weights", {})
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
