"""
Public data types and errors of the nlolib Python package.

The names and field order follow the shared binding vocabulary
(docs/binding_api_vocabulary.md in the core repository).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass
class PulseSpec:
    """Input field and the grids it lives on.

    A temporal run gives ``samples`` and ``delta_time``. A tensor run also sets
    ``tensor_nt``, ``tensor_nx`` and ``tensor_ny``, with ``samples`` flattened
    time fastest: ``index = ((x * ny) + y) * nt + t``.
    """

    samples: Sequence[complex]
    delta_time: float
    pulse_period: float | None = None
    frequency_grid: Sequence[complex] | None = None
    tensor_nt: int | None = None
    tensor_nx: int | None = None
    tensor_ny: int | None = None
    delta_x: float = 1.0
    delta_y: float = 1.0
    spatial_frequency_grid: Sequence[complex] | None = None
    potential_grid: Sequence[complex] | None = None
    wt_axis: Sequence[complex] | None = None
    kx_axis: Sequence[complex] | None = None
    ky_axis: Sequence[complex] | None = None


@dataclass
class ExecutionOptions:
    """Backend selection and device budget for one propagate call.

    ``backend_type`` is ``"cpu"``, ``"vulkan"`` or ``"auto"``; ``fft_backend``
    is ``"auto"``, ``"fftw"`` or ``"vkfft"``.
    """

    backend_type: str = "auto"
    fft_backend: str = "auto"
    device_heap_fraction: float = 0.70
    record_ring_target: int = 0
    forced_device_budget_bytes: int = 0


@dataclass
class StorageOptions:
    """Stream records to a SQLite database.

    ``run_id`` is at most 63 characters; ``cap_policy`` is ``"stop_writes"``
    or ``"fail"`` and applies when ``sqlite_max_bytes`` is reached.
    """

    sqlite_path: str
    run_id: str | None = None
    sqlite_max_bytes: int = 0
    chunk_records: int = 0
    cap_policy: str = "stop_writes"
    log_final_output_field_to_db: bool = False


@dataclass
class StepHistory:
    """Accepted solver steps, captured when ``capture_step_history`` is set."""

    step_index: list[int]
    z: list[float]
    step_size: list[float]
    next_step_size: list[float]
    error: list[float]
    dropped: int
    capacity: int


@dataclass
class StorageResult:
    """What a storage-backed propagate wrote to the database."""

    run_id: str
    records_captured: int
    records_spilled: int
    chunks_written: int
    db_size_bytes: int
    truncated: bool


@dataclass
class PropagationResult:
    """Result of `propagate`.

    ``records`` is a list of rows, one per recorded position in ``z_axis``.
    ``step_history`` is None unless ``capture_step_history`` was set, and
    ``storage_result`` is None unless ``storage`` was given. ``meta`` holds
    ``preset``, ``records_requested``, ``records_written``,
    ``backend_requested``, ``coupled`` and ``status``.
    """

    records: list[list[complex]]
    z_axis: list[float]
    final: list[complex]
    step_history: StepHistory | None
    storage_result: StorageResult | None
    meta: dict[str, Any]


@dataclass(frozen=True)
class ProgressInfo:
    """One progress event; ``event_type`` is ``"accepted"``, ``"rejected"`` or ``"finish"``."""

    event_type: str
    step_index: int
    reject_attempt: int
    z: float
    z_end: float
    percent: float
    step_size: float
    next_step_size: float
    error: float
    elapsed_seconds: float
    eta_seconds: float


@dataclass(frozen=True)
class RuntimeLimits:
    """Solver limits for the current backend; the C ``runtime_limits``."""

    max_num_time_samples_runtime: int
    max_num_recorded_samples_in_memory: int
    max_num_recorded_samples_with_storage: int
    estimated_required_working_set_bytes: int
    estimated_device_budget_bytes: int
    storage_available: bool


@dataclass(frozen=True)
class PerfProfileSnapshot:
    """Accumulated performance counters; the C ``nlo_perf_profile_snapshot``."""

    dispersion_ms: float
    nonlinear_ms: float
    dispersion_calls: int
    nonlinear_calls: int
    gpu_dispatch_count: int
    gpu_copy_count: int
    gpu_device_copy_count: int
    gpu_device_copy_bytes: int
    gpu_host_transfer_copy_count: int
    gpu_host_transfer_copy_bytes: int
    gpu_memory_pass_count: int
    gpu_memory_pass_bytes: int
    gpu_upload_count: int
    gpu_download_count: int
    gpu_upload_bytes: int
    gpu_download_bytes: int
    operator_jit_programs: int
    operator_jit_fallbacks: int


class NlolibError(RuntimeError):
    """A C call returned a non-OK status, or the loaded build lacks a feature.

    ``status`` is the status string, e.g. ``"invalid_argument"``.
    """

    def __init__(self, status: str, message: str):
        super().__init__(f"{message} (status={status})")
        self.status = status


class PropagationAbortedError(NlolibError):
    """The progress callback stopped the run; ``result`` holds what was recorded."""

    def __init__(self, result: PropagationResult):
        super().__init__("aborted", "propagate aborted by progress callback")
        self.result = result
