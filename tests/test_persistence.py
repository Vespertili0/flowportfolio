import json
import re
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from skfolio import Population, Portfolio

from flowportfolio.persistence import PersistenceManager


@pytest.fixture
def dummy_population():
    # Create a dummy population using some mock data
    df1 = pd.DataFrame(np.random.randn(10, 3) / 100, columns=["A", "B", "C"])
    df1.index = pd.date_range("2023-01-01", periods=10)
    p1 = Portfolio(X=df1, weights=np.array([0.3, 0.3, 0.4]), name="port1", tag="tag1")

    df2 = pd.DataFrame(np.random.randn(10, 3) / 100, columns=["A", "B", "C"])
    df2.index = pd.date_range("2023-01-01", periods=10)
    p2 = Portfolio(X=df2, weights=np.array([0.5, 0.5, 0.0]), name="port2", tag="tag2")

    return Population([p1, p2])


@pytest.fixture
def manager():
    return PersistenceManager()


def test_save_snapshot_success(manager, dummy_population, tmp_path):
    filepath = tmp_path / "test_snapshot.json"
    manager.save_snapshot(dummy_population, str(filepath))

    assert filepath.exists()

    with filepath.open("r") as f:
        data = json.load(f)

    assert "timestamp" in data
    assert "portfolios" in data
    assert len(data["portfolios"]) == 2

    port1_data = data["portfolios"][0]
    assert port1_data["name"] == "port1"
    assert port1_data["tag"] == "tag1"
    assert "A" in port1_data["weights"]
    assert "B" in port1_data["weights"]
    assert "C" in port1_data["weights"]
    assert "sharpe" in port1_data
    assert "cvar" in port1_data
    assert "max_drawdown" in port1_data


def test_save_snapshot_type_error(manager, tmp_path):
    filepath = tmp_path / "test.json"
    with pytest.raises(
        TypeError, match="population must be a skfolio.Population instance."
    ):
        manager.save_snapshot([], str(filepath))


def test_save_snapshot_value_error(manager, tmp_path):
    filepath = tmp_path / "test.json"
    empty_pop = Population([])
    with pytest.raises(ValueError, match="population is empty."):
        manager.save_snapshot(empty_pop, str(filepath))


def test_save_snapshot_mkdir_os_error(manager, dummy_population, tmp_path):
    # Simulate a file existing at the directory path, which causes mkdir to raise OSError
    conflict_path = tmp_path / "conflict"
    conflict_path.touch()
    filepath = conflict_path / "test.json"

    with pytest.raises(
        OSError, match=re.escape(f"Cannot create directory {conflict_path}")
    ):
        manager.save_snapshot(dummy_population, str(filepath))


def test_save_snapshot_os_error(manager, dummy_population, tmp_path):
    filepath = tmp_path / "test.json"

    with (
        patch("pathlib.Path.open", side_effect=OSError("Disk full")),
        pytest.raises(
            RuntimeError, match=re.escape(f"Failed to save snapshot to {filepath}")
        ),
    ):
        manager.save_snapshot(dummy_population, str(filepath))


def test_save_snapshot_path_traversal(manager, dummy_population, tmp_path):
    filepath = tmp_path / ".." / "test.json"
    with pytest.raises(
        ValueError,
        match=re.escape(f"Invalid filepath (path traversal detected): {filepath!s}"),
    ):
        manager.save_snapshot(dummy_population, str(filepath))


def test_load_snapshot_success(manager, dummy_population, tmp_path):
    filepath = tmp_path / "test_snapshot.json"
    manager.save_snapshot(dummy_population, str(filepath))

    data = manager.load_snapshot(str(filepath))
    assert "timestamp" in data
    assert "portfolios" in data


def test_load_snapshot_file_not_found(manager, tmp_path):
    filepath = tmp_path / "non_existent.json"
    with pytest.raises(
        FileNotFoundError, match=re.escape(f"Snapshot file not found: {filepath!s}")
    ):
        manager.load_snapshot(str(filepath))


def test_load_snapshot_path_traversal(manager, tmp_path):
    filepath = tmp_path / ".." / "test.json"
    with pytest.raises(
        ValueError,
        match=re.escape(f"Invalid filepath (path traversal detected): {filepath!s}"),
    ):
        manager.load_snapshot(str(filepath))


def test_load_snapshot_invalid_json(manager, tmp_path):
    filepath = tmp_path / "invalid.json"
    with filepath.open("w") as f:
        f.write("{invalid_json:")

    with pytest.raises(
        ValueError, match=re.escape(f"Invalid JSON in snapshot file: {filepath!s}")
    ):
        manager.load_snapshot(str(filepath))


def test_export_gitops_artifact_default_label(manager, dummy_population, tmp_path):
    path_str = manager.export_gitops_artifact(dummy_population, str(tmp_path))
    path = Path(path_str)

    now = datetime.now(UTC)
    year, week, _ = now.isocalendar()
    expected_filename = f"rebalance_{year}_W{week:02d}.json"

    assert path.name == expected_filename
    assert path.exists()


def test_export_gitops_artifact_with_label(manager, dummy_population, tmp_path):
    path_str = manager.export_gitops_artifact(
        dummy_population, str(tmp_path), label="custom_label"
    )
    path = Path(path_str)

    assert path.name == "custom_label.json"
    assert path.exists()


@pytest.mark.parametrize("invalid_label", [".", "..", ".hidden", "///", "", "..\\.."])
def test_export_gitops_artifact_invalid_label(
    manager, dummy_population, tmp_path, invalid_label
):
    with pytest.raises(
        ValueError, match="Invalid label provided for Git-Ops artifact."
    ):
        manager.export_gitops_artifact(
            dummy_population, str(tmp_path), label=invalid_label
        )


def test_export_gitops_artifact_label_type_error(manager, dummy_population, tmp_path):
    with pytest.raises(TypeError, match="label must be a string or None"):
        manager.export_gitops_artifact(dummy_population, str(tmp_path), label=123)


@pytest.mark.parametrize("invalid_dir", ["../foo", "foo/../../bar", "a/b/../c"])
def test_export_gitops_artifact_invalid_export_dir(
    manager, dummy_population, invalid_dir
):
    with pytest.raises(ValueError, match="Path traversal detected in export_dir."):
        manager.export_gitops_artifact(dummy_population, invalid_dir)


def test_calculate_trajectory_history_success(manager, dummy_population, tmp_path):
    # Save a few snapshots
    manager.export_gitops_artifact(dummy_population, str(tmp_path))
    # We use export_gitops_artifact because it prefixes with rebalance_
    # Let's manually create another file to simulate a different week

    # file 2
    file2 = tmp_path / "rebalance_2026_W34.json"
    manager.save_snapshot(dummy_population, str(file2))

    df = manager.calculate_trajectory_history(str(tmp_path))

    assert isinstance(df, pd.DataFrame)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert "portfolio_name" in df.columns
    assert "tag" in df.columns
    assert "A" in df.columns
    assert "B" in df.columns
    assert "C" in df.columns
    # We saved 2 files, each with 2 portfolios -> 4 rows
    assert len(df) == 4


def test_calculate_trajectory_history_dir_not_found(manager, tmp_path):
    bad_dir = tmp_path / "no_dir"
    with pytest.raises(
        FileNotFoundError,
        match=re.escape(f"Snapshot directory not found: {bad_dir!s}"),
    ):
        manager.calculate_trajectory_history(str(bad_dir))


def test_calculate_trajectory_history_path_traversal(manager, tmp_path):
    bad_dir = tmp_path / ".." / "snapshots"
    with pytest.raises(ValueError, match="Path traversal detected in snapshot_dir."):
        manager.calculate_trajectory_history(str(bad_dir))


def test_calculate_trajectory_history_no_files(manager, tmp_path):
    # directory exists, but no files
    with pytest.raises(
        ValueError,
        match=re.escape(f"No rebalance_*.json snapshots found in: {tmp_path!s}"),
    ):
        manager.calculate_trajectory_history(str(tmp_path))


def test_load_snapshot_os_error(manager, tmp_path, monkeypatch):
    filepath = tmp_path / "os_error.json"
    with filepath.open("w") as f:
        f.write("{}")

    def mock_open(*args, **kwargs):
        raise OSError("Mocked OSError")

    monkeypatch.setattr(Path, "open", mock_open)

    with pytest.raises(
        RuntimeError, match=re.escape(f"Failed to read snapshot from {filepath!s}")
    ):
        manager.load_snapshot(str(filepath))
