import inspect
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from skfolio import MultiPeriodPortfolio, Population, Portfolio

from flowportfolio import persistence as persistence_module
from flowportfolio.core.experiment import ExperimentResult
from flowportfolio.persistence import PersistenceManager


@pytest.fixture
def dummy_population():
    rng = np.random.default_rng(42)
    df1 = pd.DataFrame(rng.normal(0.001, 0.01, (10, 3)), columns=["A", "B", "C"])
    df1.index = pd.bdate_range("2023-01-02", periods=10)
    p1 = Portfolio(X=df1, weights=np.array([0.3, 0.3, 0.4]), name="port1", tag="tag1")

    df2 = pd.DataFrame(rng.normal(0.001, 0.01, (10, 3)), columns=["A", "B", "C"])
    df2.index = pd.bdate_range("2023-01-02", periods=10)
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


def test_save_snapshot_multi_period_portfolio(manager, tmp_path):
    filepath = tmp_path / "test_mpp_snapshot.json"

    rng = np.random.default_rng(7)
    idx1 = pd.bdate_range("2025-01-02", periods=10)
    idx2 = pd.bdate_range("2025-01-16", periods=10)
    sub_p1 = Portfolio(
        X=pd.DataFrame(
            rng.normal(0.001, 0.01, (10, 2)), columns=["A", "B"], index=idx1
        ),
        weights=np.array([0.5, 0.5]),
        name="fold1",
    )
    sub_p2 = Portfolio(
        X=pd.DataFrame(
            rng.normal(0.001, 0.01, (10, 2)), columns=["A", "B"], index=idx2
        ),
        weights=np.array([0.7, 0.3]),
        name="fold2",
    )
    mpp = MultiPeriodPortfolio(
        portfolios=[sub_p1, sub_p2], name="mpp_strat", tag="mpp_tag"
    )
    pop = Population([mpp])

    manager.save_snapshot(pop, str(filepath))

    with filepath.open("r") as f:
        data = json.load(f)

    assert len(data["portfolios"]) == 1
    p_data = data["portfolios"][0]
    assert p_data["name"] == "mpp_strat"
    assert p_data["weights"]["A"] == pytest.approx(0.7, abs=1e-12)
    assert p_data["weights"]["B"] == pytest.approx(0.3, abs=1e-12)


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
        patch("os.fdopen", side_effect=OSError("Disk full")),
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
    assert path.name.startswith(
        f"rebalance_{year}_W{week:02d}_"
    ) and path.name.endswith(".json")
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


def test_persistence_sanitizes_inf_and_loads_custom_labels(
    manager, dummy_population, tmp_path
):
    """Verify sanitisation of Inf and loading of snapshots with custom labels."""
    sanitized = manager._sanitize_nans(
        {"sharpe": float("inf"), "cvar": float("-inf"), "np_val": np.float64(np.nan)}
    )
    assert sanitized["sharpe"] is None
    assert sanitized["cvar"] is None
    assert sanitized["np_val"] is None

    custom_path = manager.export_gitops_artifact(
        dummy_population, str(tmp_path), label="custom_audit_run"
    )
    assert Path(custom_path).exists()

    with pytest.raises(ValueError, match=r"No rebalance_\*\.json snapshots found"):
        manager.calculate_trajectory_history(str(tmp_path), include_custom_labels=False)

    df = manager.calculate_trajectory_history(str(tmp_path), include_custom_labels=True)
    assert len(df) == 2
    assert "A" in df.columns


# ---------------------------------------------------------------------------
# Cycle 10 (F-05, F-09, F-10): Live Weights, Lossless ISO-8601, Schema & Paths
# ---------------------------------------------------------------------------


def test_save_snapshot_and_export_preserve_live_weights_across_population_rewrap(
    manager: PersistenceManager, tmp_path: Path
) -> None:
    """F-05 (O5): Explicit live_weights and ExperimentResult override stale Fold-N weights after Population(list(pop))."""
    rng = np.random.default_rng(11)
    fold1 = Portfolio(
        X=pd.DataFrame(
            rng.normal(0.001, 0.01, (10, 2)),
            columns=["A", "B"],
            index=pd.bdate_range("2025-01-02", periods=10),
        ),
        weights=np.array([0.9, 0.1]),
        name="fold_last",
    )
    mpp = MultiPeriodPortfolio(portfolios=[fold1], name="LiveStrat", tag="LiveStrat")
    raw_pop = Population([mpp])
    # Re-wrapping Population(list(pop)) strips any dynamic attributes
    rewrapped = Population(list(raw_pop))
    live_weights = {"LiveStrat": {"A": 0.2, "B": 0.8}}

    # 1. Explicit live_weights kwarg on save_snapshot
    snap_path = tmp_path / "live_snap.json"
    manager.save_snapshot(rewrapped, str(snap_path), live_weights=live_weights)
    saved = manager.load_snapshot(str(snap_path))
    assert saved["portfolios"][0]["weights"] == {"A": 0.2, "B": 0.8}

    # 2. Direct ExperimentResult passed to export_gitops_artifact
    result = ExperimentResult(population=rewrapped, live_weights=live_weights)
    artifact_path = manager.export_gitops_artifact(
        result, str(tmp_path), label="from_result"
    )
    exported = manager.load_snapshot(artifact_path)
    assert exported["portfolios"][0]["weights"] == {"A": 0.2, "B": 0.8}

    # 3. Invalid live_weights type raises TypeError
    with pytest.raises(TypeError, match="live_weights"):
        manager.save_snapshot(
            rewrapped,
            str(tmp_path / "bad.json"),
            live_weights=[("LiveStrat", {"A": 1.0})],  # type: ignore[arg-type]
        )


def test_calculate_trajectory_history_lossless_mixed_iso8601_and_schema_evolution(
    manager: PersistenceManager, tmp_path: Path
) -> None:
    """F-09 (O5): Mixed integer/microsecond/Z ISO-8601 timestamps preserve 100% of rows in UTC with 0.0 schema fill."""
    snapshots = [
        (
            "rebalance_2026_W01.json",
            {
                "timestamp": "2026-01-01T00:00:00+00:00",
                "portfolios": [
                    {"name": "p1", "tag": "S1", "weights": {"A": 0.6, "B": 0.4}}
                ],
            },
        ),
        (
            "rebalance_2026_W02.json",
            {
                "timestamp": "2026-01-02T00:00:00.123456+00:00",
                "portfolios": [
                    {"name": "p1", "tag": "S1", "weights": {"A": 0.5, "C": 0.5}}
                ],
            },
        ),
        (
            "rebalance_2026_W03.json",
            {
                "timestamp": "2026-01-03T00:00:00Z",
                "portfolios": [
                    {"name": "p1", "tag": "S1", "weights": {"B": 0.3, "C": 0.7}}
                ],
            },
        ),
    ]
    for fname, payload in snapshots:
        (tmp_path / fname).write_text(json.dumps(payload), encoding="utf-8")

    df = manager.calculate_trajectory_history(str(tmp_path))

    # All 3 mixed-precision snapshots MUST be preserved in UTC
    assert len(df) == 3
    assert df.index.tz is not None
    assert list(df.index) == list(
        pd.to_datetime(
            [
                "2026-01-01T00:00:00+00:00",
                "2026-01-02T00:00:00.123456+00:00",
                "2026-01-03T00:00:00+00:00",
            ],
            format="ISO8601",
            utc=True,
        )
    )

    # Absent tickers across schema evolution must be filled with explicit 0.0 (no NaNs)
    ticker_cols = ["A", "B", "C"]
    assert not df[ticker_cols].isna().any().any()
    assert df.iloc[0]["C"] == 0.0
    assert df.iloc[1]["B"] == 0.0
    assert df.iloc[2]["A"] == 0.0
    np.testing.assert_allclose(df[ticker_cols].sum(axis=1).to_numpy(), 1.0, atol=1e-12)


def test_calculate_trajectory_history_rejects_malformed_or_missing_timestamps(
    manager: PersistenceManager, tmp_path: Path
) -> None:
    """F-09.1: Malformed or missing timestamps must raise ValueError rather than silently dropping rows."""
    (tmp_path / "rebalance_2026_W01.json").write_text(
        json.dumps(
            {
                "timestamp": "not-an-iso-timestamp",
                "portfolios": [{"name": "p1", "tag": "S1", "weights": {"A": 1.0}}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="timestamp"):
        manager.calculate_trajectory_history(str(tmp_path))

    (tmp_path / "rebalance_2026_W01.json").write_text(
        json.dumps(
            {"portfolios": [{"name": "p1", "tag": "S1", "weights": {"A": 1.0}}]}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="timestamp"):
        manager.calculate_trajectory_history(str(tmp_path))


def test_persistence_symlink_escape_and_base_dir_containment(
    dummy_population: Population, tmp_path: Path
) -> None:
    """F-09.2: Symlinks escaping base_dir or snapshot_dir must raise ValueError; non-dict JSON raises ValueError."""
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()

    outside_file = outside / "secret.json"
    outside_file.write_text(
        json.dumps(
            {
                "timestamp": "2026-01-01T00:00:00+00:00",
                "portfolios": [{"name": "p1", "tag": "S1", "weights": {"A": 1.0}}],
            }
        ),
        encoding="utf-8",
    )

    # Symlink inside allowed dir pointing outside
    escaped_link = allowed / "rebalance_2026_W01.json"
    escaped_link.symlink_to(outside_file)

    # Even without base_dir, calculate_trajectory_history must reject symlinks escaping snapshot_dir
    mgr_default = PersistenceManager()
    with pytest.raises(ValueError, match="outside|traversal|escapes"):
        mgr_default.calculate_trajectory_history(str(allowed))

    # With explicit base_dir, symlink escape is blocked across save, load, export, trajectory
    mgr_scoped = PersistenceManager(base_dir=allowed)
    with pytest.raises(ValueError, match="outside|traversal|escapes"):
        mgr_scoped.load_snapshot(str(escaped_link))
    with pytest.raises(ValueError, match="outside|traversal|escapes"):
        mgr_scoped.save_snapshot(dummy_population, str(outside / "escape.json"))

    escaped_dir = allowed / "linked_out"
    escaped_dir.symlink_to(outside)
    with pytest.raises(ValueError, match="outside|traversal|escapes"):
        mgr_scoped.export_gitops_artifact(dummy_population, str(escaped_dir))

    # Non-dict JSON payload raises ValueError in load_snapshot
    list_json = allowed / "list_payload.json"
    list_json.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        mgr_scoped.load_snapshot(str(list_json))


def test_persistence_module_free_of_mock_name_and_slots_monkey_patch() -> None:
    """F-05 & F-10: persistence.py contains zero _mock_name or dead latest_weights_dict references."""
    source = inspect.getsource(persistence_module)
    assert "_mock_name" not in source
    assert "latest_weights_dict" not in source
