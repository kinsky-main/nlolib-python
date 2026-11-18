"""
The public functions of the nlolib Python package.

Every function works on one process-wide library, loaded on first use or by an
explicit `load(path)`.
"""

from __future__ import annotations

import ctypes
from typing import Callable, Mapping, Sequence

from . import _enums
from ._binding import (
    LEGACY_NT_MAX,
    NLOLIB_STATUS_OK,
    NloPerfProfileSnapshot,
    NloRuntimeLimits,
    load_library,
)
from ._config import _normalize_pulse_spec, _prepare_sim_config
from ._executor import PropagationExecutor, _c_execution_options
from ._models import (
    ExecutionOptions,
    NlolibError,
    PerfProfileSnapshot,
    ProgressInfo,
    PropagationResult,
    PulseSpec,
    RuntimeLimits,
    StorageOptions,
)
from ._requests import _validate_exec_options, build_request

_LIB: ctypes.CDLL | None = None


def load(path=None) -> None:
    """Load the nlolib shared library.

    With `path`, that file is loaded. Without it, the library named by the
    NLOLIB_LIBRARY environment variable is loaded, or else the one next to the
    package. Calling it again replaces the loaded library. Raises OSError when
    the library cannot be found or loaded.
    """
    global _LIB
    _LIB = load_library(path)


def _lib() -> ctypes.CDLL:
    if _LIB is None:
        load()
    assert _LIB is not None
    return _LIB


def loaded_library_path() -> str:
    """Path of the loaded shared library, loading the default one if needed."""
    return str(_lib().loaded_path)


def _require(feature: str, symbol: str) -> ctypes.CDLL:
    lib = _lib()
    if not bool(getattr(lib, f"_has_{feature}", False)):
        raise NlolibError("not_implemented", f"{symbol} is unavailable in the loaded nlolib build")
    return lib


def _check(status: int, symbol: str) -> None:
    if int(status) != NLOLIB_STATUS_OK:
        raise NlolibError(_enums.status_name(int(status)), f"{symbol} failed")


def propagate(
    pulse: PulseSpec,
    linear_operator: Callable[..., object],
    nonlinear_operator: Callable[..., object],
    *,
    propagation_distance: float,
    preset: str = "balanced",
    starting_step_size: float | None = None,
    max_step_size: float | None = None,
    min_step_size: float | None = None,
    error_tolerance: float | None = None,
    num_records: int | None = None,
    record_z: Sequence[float] | None = None,
    return_records: bool = True,
    potential_operator: Callable[..., object] | None = None,
    linear_step_operator: Callable[..., object] | None = None,
    constant_bindings: Mapping[str, float] | None = None,
    nonlinear_model: str = "expr",
    nonlinear_gamma: float = 0.0,
    raman_fraction: float = 0.0,
    raman_tau1: float = 0.0122,
    raman_tau2: float = 0.0320,
    shock_omega0: float = 0.0,
    raman_response_time: Sequence[complex] | None = None,
    exec_options: ExecutionOptions | None = None,
    storage: StorageOptions | None = None,
    capture_step_history: bool = False,
    step_history_capacity: int = 200000,
    progress_callback: Callable[[ProgressInfo], bool | int | None] | None = None,
) -> PropagationResult:
    """Propagate `pulse` and return its recorded fields.

    Each operator is a function whose single-expression body is lowered once to
    the core's expression language; free names are captured from its scope and
    `constant_bindings` names values explicitly (a binding wins over a captured
    name). `linear_operator` is the linear factor, e.g.
    ``lambda A, w: 1j * beta2 * w * w``; `nonlinear_operator` is e.g.
    ``lambda A, I: 1j * gamma * A * I``. `linear_step_operator` replaces the
    per-step linear operator (default ``exp(h*D)``) and `potential_operator`
    sets the potential of a tensor run.

    Records: with `record_z`, exactly those positions, for any length
    including one; otherwise `num_records` (from the preset when unset)
    uniformly spaced positions from 0 to `propagation_distance`, and
    `num_records=1` returns the field at `propagation_distance` only.
    `result.final` is the last returned record, so it is the field at
    `propagation_distance` only when the last position is that distance.

    Raises TypeError when an operator is not a function, ValueError for a bad
    pulse, an untranslatable operator or a bad option value, NlolibError when
    the core returns a non-OK status, and PropagationAbortedError when
    `progress_callback` returns False.
    """
    request = build_request(
        pulse,
        linear_operator,
        nonlinear_operator,
        propagation_distance=propagation_distance,
        preset=preset,
        starting_step_size=starting_step_size,
        max_step_size=max_step_size,
        min_step_size=min_step_size,
        error_tolerance=error_tolerance,
        num_records=num_records,
        record_z=record_z,
        return_records=return_records,
        potential_operator=potential_operator,
        linear_step_operator=linear_step_operator,
        constant_bindings=constant_bindings,
        nonlinear_model=nonlinear_model,
        nonlinear_gamma=nonlinear_gamma,
        raman_fraction=raman_fraction,
        raman_tau1=raman_tau1,
        raman_tau2=raman_tau2,
        shock_omega0=shock_omega0,
        raman_response_time=raman_response_time,
        exec_options=exec_options,
        storage=storage,
        capture_step_history=capture_step_history,
        step_history_capacity=step_history_capacity,
        progress_callback=progress_callback,
    )
    return PropagationExecutor(_lib(), storage_is_available).execute(request)


def query_runtime_limits(
    *,
    pulse: PulseSpec | None = None,
    exec_options: ExecutionOptions | None = None,
) -> RuntimeLimits:
    """Solver limits for the backend `exec_options` selects.

    With `pulse`, the grid-dependent fields
    (`max_num_recorded_samples_in_memory`,
    `estimated_required_working_set_bytes`) are computed for its grid.
    """
    lib = _lib()
    exec_options = _validate_exec_options(exec_options)
    if not bool(getattr(lib, "_has_query_runtime_limits", False)):
        return RuntimeLimits(
            max_num_time_samples_runtime=int(LEGACY_NT_MAX),
            max_num_recorded_samples_in_memory=int(LEGACY_NT_MAX),
            max_num_recorded_samples_with_storage=int(LEGACY_NT_MAX),
            estimated_required_working_set_bytes=0,
            estimated_device_budget_bytes=0,
            storage_available=storage_is_available(),
        )

    sim_ptr = None
    config = None
    if pulse is not None:
        spec = _normalize_pulse_spec(pulse)
        # Placeholder step sizes: the limits depend on the grid, not on them.
        config = _prepare_sim_config(
            spec,
            propagation_distance=1.0,
            starting_step_size=1.0,
            max_step_size=1.0,
            min_step_size=1.0,
            error_tolerance=1.0,
        )
        sim_cfg = config.simulation_config
        # The library derives the grid-dependent fields only from a config that
        # states its sample count; a temporal config leaves time.nt at zero.
        if sim_cfg.time.nt == 0 and sim_cfg.tensor.nt == 0:
            sim_cfg.time.nt = int(config.num_time_samples)
        sim_ptr = ctypes.pointer(sim_cfg)
    c_options = _c_execution_options(exec_options)
    opts_ptr = ctypes.pointer(c_options) if c_options is not None else None
    out = NloRuntimeLimits()
    _check(lib.nlolib_query_runtime_limits(sim_ptr, None, opts_ptr, ctypes.pointer(out)),
           "nlolib_query_runtime_limits")
    return RuntimeLimits(
        max_num_time_samples_runtime=int(out.max_num_time_samples_runtime),
        max_num_recorded_samples_in_memory=int(out.max_num_recorded_samples_in_memory),
        max_num_recorded_samples_with_storage=int(out.max_num_recorded_samples_with_storage),
        estimated_required_working_set_bytes=int(out.estimated_required_working_set_bytes),
        estimated_device_budget_bytes=int(out.estimated_device_budget_bytes),
        storage_available=bool(out.storage_available),
    )


def storage_is_available() -> bool:
    """Whether the loaded library was built with SQLite storage."""
    lib = _lib()
    if not bool(getattr(lib, "_has_storage_is_available", False)):
        return False
    return bool(int(lib.nlolib_storage_is_available()))


def tensor_record_view(record: Sequence[complex], nt: int, nx: int, ny: int) -> list[list[list[complex]]]:
    """One flat tensor record as nested lists indexed ``[t][y][x]``.

    `record` is flattened time fastest, ``index = ((x * ny) + y) * nt + t``.
    """
    nt, nx, ny = int(nt), int(nx), int(ny)
    if nt <= 0 or nx <= 0 or ny <= 0:
        raise ValueError("nt, nx and ny must be > 0")
    if len(record) != nt * nx * ny:
        raise ValueError(f"len(record)={len(record)} does not equal nt*nx*ny={nt * nx * ny}")
    return [
        [[record[((x * ny) + y) * nt + t] for x in range(nx)] for y in range(ny)]
        for t in range(nt)
    ]


def set_log_file(path: str | None, *, append: bool = False) -> None:
    """Write runtime logs to `path` (None stops file logging)."""
    lib = _require("set_log_file", "nlolib_set_log_file")
    encoded = str(path).encode("utf-8") if path else None
    _check(lib.nlolib_set_log_file(encoded, int(bool(append))), "nlolib_set_log_file")


def set_log_buffer(capacity_bytes: int = 256 * 1024) -> None:
    """Keep runtime logs in an in-memory ring buffer of `capacity_bytes`."""
    lib = _require("set_log_buffer", "nlolib_set_log_buffer")
    _check(lib.nlolib_set_log_buffer(int(capacity_bytes)), "nlolib_set_log_buffer")


def clear_log_buffer() -> None:
    """Discard the buffered runtime logs."""
    lib = _require("clear_log_buffer", "nlolib_clear_log_buffer")
    _check(lib.nlolib_clear_log_buffer(), "nlolib_clear_log_buffer")


def read_log_buffer(*, consume: bool = True, max_bytes: int = 256 * 1024) -> str:
    """Buffered runtime logs as text, at most `max_bytes`; `consume` removes them."""
    lib = _require("read_log_buffer", "nlolib_read_log_buffer")
    if int(max_bytes) < 2:
        raise ValueError("max_bytes must be >= 2")
    out = ctypes.create_string_buffer(int(max_bytes))
    written = ctypes.c_size_t(0)
    _check(
        lib.nlolib_read_log_buffer(
            ctypes.cast(out, ctypes.c_void_p),
            ctypes.c_size_t(int(max_bytes)),
            ctypes.byref(written),
            int(bool(consume)),
        ),
        "nlolib_read_log_buffer",
    )
    return out.raw[: int(written.value)].decode("utf-8", errors="replace")


def set_log_level(level: str = "info") -> None:
    """Set the runtime log level: ``"error"``, ``"warn"``, ``"info"`` or ``"debug"``."""
    value = _enums.log_level(level)
    lib = _require("set_log_level", "nlolib_set_log_level")
    _check(lib.nlolib_set_log_level(value), "nlolib_set_log_level")


def set_progress_options(
    *,
    enabled: bool = True,
    milestone_percent: int = 5,
    emit_on_step_adjust: bool = False,
) -> None:
    """Configure the runtime progress display."""
    lib = _require("set_progress_options", "nlolib_set_progress_options")
    _check(
        lib.nlolib_set_progress_options(
            int(bool(enabled)), int(milestone_percent), int(bool(emit_on_step_adjust))
        ),
        "nlolib_set_progress_options",
    )


def set_progress_stream(stream_mode: str = "stderr") -> None:
    """Send progress lines to ``"stderr"``, ``"stdout"`` or ``"both"``."""
    value = _enums.progress_stream(stream_mode)
    lib = _require("set_progress_stream", "nlolib_set_progress_stream")
    _check(lib.nlolib_set_progress_stream(value), "nlolib_set_progress_stream")


def perf_profile_set_enabled(enabled: bool = True) -> None:
    """Enable or disable accumulation of the performance counters."""
    lib = _require("perf_profile", "nlolib_perf_profile_set_enabled")
    _check(lib.nlolib_perf_profile_set_enabled(int(bool(enabled))), "nlolib_perf_profile_set_enabled")


def perf_profile_is_enabled() -> bool:
    """Whether the performance counters are accumulating."""
    lib = _require("perf_profile", "nlolib_perf_profile_is_enabled")
    return bool(int(lib.nlolib_perf_profile_is_enabled()))


def perf_profile_reset() -> None:
    """Reset the performance counters to zero."""
    lib = _require("perf_profile", "nlolib_perf_profile_reset")
    _check(lib.nlolib_perf_profile_reset(), "nlolib_perf_profile_reset")


def perf_profile_read() -> PerfProfileSnapshot:
    """The current performance counters."""
    lib = _require("perf_profile", "nlolib_perf_profile_read")
    snapshot = NloPerfProfileSnapshot()
    _check(lib.nlolib_perf_profile_read(ctypes.pointer(snapshot)), "nlolib_perf_profile_read")
    return PerfProfileSnapshot(
        **{name: getattr(snapshot, name) for name, _ in NloPerfProfileSnapshot._fields_}
    )
