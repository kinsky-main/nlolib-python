import math
import sqlite3
import sys
import tempfile
from pathlib import Path

import nlolib
from nlolib import ExecutionOptions, PulseSpec, StorageOptions, propagate
from nlolib._executor import _DENSE_OUTPUT_MAX_BYTES, _COMPLEX_BYTES, PropagationExecutor
from nlolib._binding import NloPhysicsConfig, NloSimulationConfig
from nlolib._requests import _NormalizedPropagateRequest


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PYTHON_DIR = REPO_ROOT / "examples"
if str(EXAMPLE_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_PYTHON_DIR))

from backend.storage import ExampleRunDB

CPU = ExecutionOptions(backend_type="cpu")


def _table_exists(cur: sqlite3.Cursor, table_name: str) -> bool:
    row = cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def _gaussian_pulse(n: int, width: float) -> PulseSpec:
    return PulseSpec(
        samples=[complex(math.exp(-(((i - (n / 2.0)) / width) ** 2)), 0.0) for i in range(n)],
        delta_time=1.0 / n,
        pulse_period=1.0,
        frequency_grid=[complex(i, 0.0) for i in range(n)],
    )


def _propagate(pulse: PulseSpec, num_records: int, **kwargs):
    """A short identity-operator run; the storage path is what is under test."""
    options = dict(
        propagation_distance=0.1,
        starting_step_size=0.01,
        max_step_size=0.02,
        min_step_size=1e-4,
        error_tolerance=1e-6,
        num_records=num_records,
    )
    options.update(kwargs)
    return propagate(pulse, lambda A, w: 0.0, lambda A, I: 0.0, **options)


def _base_case(db_path: Path) -> None:
    num_records = 10
    result = _propagate(
        _gaussian_pulse(64, 10.0),
        num_records,
        storage=StorageOptions(sqlite_path=str(db_path), chunk_records=4),
        return_records=False,
    )
    storage = result.storage_result
    assert isinstance(storage, nlolib.StorageResult)
    assert storage.records_captured == num_records
    assert storage.records_spilled == num_records
    assert storage.chunks_written > 0
    assert storage.truncated is False
    assert result.records == []

    with sqlite3.connect(db_path) as con:
        cur = con.cursor()
        assert _table_exists(cur, "io_runs")
        assert _table_exists(cur, "io_record_chunks")
        assert cur.execute("SELECT COUNT(*) FROM io_runs").fetchone()[0] == 1
        assert cur.execute("SELECT COUNT(*) FROM io_record_chunks").fetchone()[0] == storage.chunks_written
        if _table_exists(cur, "io_final_output_fields"):
            assert cur.execute("SELECT COUNT(*) FROM io_final_output_fields").fetchone()[0] == 0


def _cap_case(db_path: Path) -> None:
    result = _propagate(
        _gaussian_pulse(64, 10.0),
        10,
        storage=StorageOptions(sqlite_path=str(db_path), chunk_records=2, sqlite_max_bytes=1024),
        return_records=False,
    )
    assert result.storage_result.truncated is True


def _final_output_logging_case(db_path: Path) -> None:
    n = 32
    _propagate(
        _gaussian_pulse(n, 8.0),
        4,
        storage=StorageOptions(sqlite_path=str(db_path), chunk_records=2,
                               log_final_output_field_to_db=True),
        return_records=False,
    )

    with sqlite3.connect(db_path) as con:
        cur = con.cursor()
        if _table_exists(cur, "io_final_output_fields"):
            row = cur.execute(
                "SELECT num_time_samples, length(payload) FROM io_final_output_fields LIMIT 1"
            ).fetchone()
            assert row is not None
            assert row[0] == n
            assert row[1] == n * 16


def _run_id_case(db_path: Path) -> None:
    result = _propagate(
        _gaussian_pulse(16, 4.0),
        2,
        storage=StorageOptions(sqlite_path=str(db_path), run_id="named-run"),
    )
    assert result.storage_result.run_id == "named-run"
    try:
        StorageOptions(sqlite_path=str(db_path), run_id="x" * 64)
        _propagate(
            _gaussian_pulse(16, 4.0),
            2,
            storage=StorageOptions(sqlite_path=str(db_path), run_id="x" * 64),
        )
        raise AssertionError("a 64-character run_id must be rejected")
    except ValueError as exc:
        assert "run_id" in str(exc)
    try:
        _propagate(
            _gaussian_pulse(16, 4.0),
            2,
            storage=StorageOptions(sqlite_path=str(db_path), cap_policy="drop"),
        )
        raise AssertionError("an unknown cap_policy must be rejected")
    except ValueError as exc:
        assert "cap_policy" in str(exc)


class _HugeInput:
    def __len__(self) -> int:
        return (_DENSE_OUTPUT_MAX_BYTES // _COMPLEX_BYTES) + 1

    def __iter__(self):
        raise AssertionError("dense output validation should run before input conversion")


def _dense_output_guard_case() -> None:
    executor = PropagationExecutor(object(), lambda: True)
    request = _NormalizedPropagateRequest(
        sim_cfg=NloSimulationConfig(),
        phys_cfg=NloPhysicsConfig(),
        keepalive=[],
        input_seq=_HugeInput(),  # type: ignore[arg-type]
        num_records=1,
        exec_options=None,
        storage=StorageOptions(sqlite_path="guard.db"),
        return_records=True,
        capture_step_history=False,
        step_history_capacity=0,
        record_z=None,
        progress_callback=None,
        meta_overrides={},
    )

    try:
        executor.execute(request)
        raise AssertionError("expected dense output guard to reject oversized request")
    except ValueError as exc:
        message = str(exc)
        assert "return_records=False" in message
        assert "SQLite storage" in message


def _example_db_tensor_reload_case(db_path: Path) -> None:
    db = ExampleRunDB(db_path)
    example_name = "test_python_storage_chunking"
    run_group = db.begin_group(example_name, "tensor-reload")
    case_key = "tensor"
    storage_kwargs = db.storage_kwargs(
        example_name=example_name,
        run_group=run_group,
        case_key=case_key,
        chunk_records=2,
    )

    nt = 4
    nx = 3
    ny = 2
    n = nt * nx * ny
    pulse = PulseSpec(
        samples=[complex(math.exp(-(((i % nt) - (nt / 2.0)) ** 2)), 0.0) for i in range(n)],
        delta_time=1.0 / nt,
        pulse_period=1.0,
        frequency_grid=[complex(i, 0.0) for i in range(nt)],
        tensor_nt=nt,
        tensor_nx=nx,
        tensor_ny=ny,
        potential_grid=[0j] * (nx * ny),
    )
    result = propagate(
        pulse,
        lambda A, wt: 0.0,
        lambda A, I, V: 1j * A * V,
        propagation_distance=0.05,
        starting_step_size=0.01,
        max_step_size=0.02,
        min_step_size=1e-4,
        error_tolerance=1e-6,
        num_records=3,
        exec_options=CPU,
        return_records=False,
        **storage_kwargs,
    )
    assert result.records == []
    db.save_case_from_result(
        example_name=example_name,
        run_group=run_group,
        case_key=case_key,
        result=result,
        meta={"nt": nt, "nx": nx, "ny": ny},
    )

    loaded = db.load_case(example_name=example_name, run_group=run_group, case_key=case_key)
    assert loaded.records.shape == (3, n)
    assert loaded.z_axis.shape == (3,)
    assert loaded.num_time_samples == n
    assert loaded.requested_records == 3
    assert loaded.records[-1].shape == (n,)


def _small_dense_return_case() -> None:
    n = 40
    result = _propagate(_gaussian_pulse(n, 7.0), 6, return_records=True)

    assert result.storage_result is None
    assert int(result.meta["records_written"]) == 6
    assert len(result.records) == 6
    assert len(result.final) == n


def main() -> None:
    if not nlolib.storage_is_available():
        print("test_python_storage_chunking: storage unavailable, skipping.")
        return

    tmp = Path(tempfile.mkdtemp(prefix="nlolib-storage-"))
    _base_case(tmp / "base.db")
    _cap_case(tmp / "cap.db")
    _final_output_logging_case(tmp / "final_output.db")
    _run_id_case(tmp / "run_id.db")
    _dense_output_guard_case()
    _example_db_tensor_reload_case(tmp / "example_tensor.db")
    _small_dense_return_case()
    print("test_python_storage_chunking: passed")


if __name__ == "__main__":
    main()
