"""
Private normalization of a `propagate` call into the C-facing request.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from . import _enums
from ._binding import STORAGE_RUN_ID_MAX, NloPhysicsConfig, NloSimulationConfig
from ._config import (
    _is_tensor_run,
    _lower_operators,
    _normalize_pulse_spec,
    _prepare_sim_config,
    _require_callable,
    _solver_profile_defaults,
    _validate_record_z,
)
from ._models import ExecutionOptions, ProgressInfo, StorageOptions


@dataclass
class _NormalizedPropagateRequest:
    sim_cfg: NloSimulationConfig
    phys_cfg: NloPhysicsConfig
    keepalive: list[object]
    input_seq: Sequence[complex]
    num_records: int
    exec_options: ExecutionOptions | None
    storage: StorageOptions | None
    return_records: bool
    capture_step_history: bool
    step_history_capacity: int
    record_z: list[float] | None
    progress_callback: Callable[[ProgressInfo], bool | int | None] | None
    meta_overrides: dict[str, Any]


def _validate_exec_options(exec_options: object) -> ExecutionOptions | None:
    if exec_options is None:
        return None
    if not isinstance(exec_options, ExecutionOptions):
        raise ValueError("exec_options must be an ExecutionOptions")
    _enums.vector_backend(exec_options.backend_type)
    _enums.fft_backend(exec_options.fft_backend)
    if not 0.0 < float(exec_options.device_heap_fraction) <= 1.0:
        raise ValueError("exec_options.device_heap_fraction must be in (0, 1]")
    if int(exec_options.record_ring_target) < 0 or int(exec_options.forced_device_budget_bytes) < 0:
        raise ValueError("exec_options.record_ring_target and forced_device_budget_bytes must be >= 0")
    return exec_options


def _validate_storage(storage: object) -> StorageOptions | None:
    if storage is None:
        return None
    if not isinstance(storage, StorageOptions):
        raise ValueError("storage must be a StorageOptions")
    if not storage.sqlite_path:
        raise ValueError("storage.sqlite_path must be a non-empty path")
    if storage.run_id is not None and len(str(storage.run_id).encode("utf-8")) >= STORAGE_RUN_ID_MAX:
        raise ValueError(f"storage.run_id must be at most {STORAGE_RUN_ID_MAX - 1} characters")
    _enums.cap_policy(storage.cap_policy)
    if int(storage.sqlite_max_bytes) < 0 or int(storage.chunk_records) < 0:
        raise ValueError("storage.sqlite_max_bytes and storage.chunk_records must be >= 0")
    return storage


def build_request(
    pulse: Any,
    linear_operator: object,
    nonlinear_operator: object,
    *,
    propagation_distance: float,
    preset: str,
    starting_step_size: float | None,
    max_step_size: float | None,
    min_step_size: float | None,
    error_tolerance: float | None,
    num_records: int | None,
    record_z: Sequence[float] | None,
    return_records: bool,
    potential_operator: object,
    linear_step_operator: object,
    constant_bindings: Mapping[str, float] | None,
    nonlinear_model: str,
    nonlinear_gamma: float,
    raman_fraction: float,
    raman_tau1: float,
    raman_tau2: float,
    shock_omega0: float,
    raman_response_time: Sequence[complex] | None,
    exec_options: ExecutionOptions | None,
    storage: StorageOptions | None,
    capture_step_history: bool,
    step_history_capacity: int,
    progress_callback: Callable[[ProgressInfo], bool | int | None] | None,
) -> _NormalizedPropagateRequest:
    _require_callable("linear_operator", linear_operator)
    _require_callable("nonlinear_operator", nonlinear_operator)
    if potential_operator is not None:
        _require_callable("potential_operator", potential_operator)
    if linear_step_operator is not None:
        _require_callable("linear_step_operator", linear_step_operator)
    if progress_callback is not None and not callable(progress_callback):
        raise ValueError("progress_callback must be callable")

    pulse_spec = _normalize_pulse_spec(pulse)
    tensor_run = _is_tensor_run(pulse_spec)
    if potential_operator is not None and pulse_spec.potential_grid is not None:
        raise ValueError(
            "potential_operator and pulse.potential_grid are both set; the core uses only "
            "the grid, so give one of them"
        )

    model = _enums.nonlinear_model(nonlinear_model)
    exec_options = _validate_exec_options(exec_options)
    storage = _validate_storage(storage)

    profile = _solver_profile_defaults(preset, float(propagation_distance))
    if starting_step_size is not None:
        profile["starting_step_size"] = float(starting_step_size)
    if max_step_size is not None:
        profile["max_step_size"] = float(max_step_size)
    if min_step_size is not None:
        profile["min_step_size"] = float(min_step_size)
    if error_tolerance is not None:
        profile["error_tolerance"] = float(error_tolerance)

    resolved_record_z: list[float] | None = None
    if record_z is not None:
        resolved_record_z = [float(v) for v in record_z]
        _validate_record_z(resolved_record_z, float(propagation_distance))
        if num_records is not None and int(num_records) != len(resolved_record_z):
            raise ValueError(
                f"num_records={num_records} disagrees with len(record_z)={len(resolved_record_z)}; "
                "leave num_records unset"
            )
        records = len(resolved_record_z)
    else:
        records = int(num_records) if num_records is not None else int(profile["num_records"])
    if records <= 0:
        raise ValueError("num_records must be > 0")

    if int(step_history_capacity) < 0:
        raise ValueError("step_history_capacity must be >= 0")
    if capture_step_history and int(step_history_capacity) <= 0:
        raise ValueError("step_history_capacity must be > 0 when capture_step_history is set")

    expressions, constants = _lower_operators(
        {
            "linear_operator": linear_operator,
            "linear_step_operator": linear_step_operator,
            "potential_operator": potential_operator,
            "nonlinear_operator": nonlinear_operator,
        },
        tensor_run=tensor_run,
        constant_bindings=constant_bindings,
    )

    config = _prepare_sim_config(
        pulse_spec,
        propagation_distance=float(propagation_distance),
        starting_step_size=float(profile["starting_step_size"]),
        max_step_size=float(profile["max_step_size"]),
        min_step_size=float(profile["min_step_size"]),
        error_tolerance=float(profile["error_tolerance"]),
        expressions=expressions,
        constants=constants,
        nonlinear_model=model,
        nonlinear_gamma=float(nonlinear_gamma),
        raman_fraction=float(raman_fraction),
        raman_tau1=float(raman_tau1),
        raman_tau2=float(raman_tau2),
        shock_omega0=float(shock_omega0),
        raman_response_time=raman_response_time,
    )
    coupled = bool(
        tensor_run and (int(pulse_spec.tensor_nx) > 1 or int(pulse_spec.tensor_ny) > 1)  # type: ignore[arg-type]
    )
    return _NormalizedPropagateRequest(
        sim_cfg=config.simulation_config,
        phys_cfg=config.physics_config,
        keepalive=config.keepalive,
        input_seq=pulse_spec.samples,
        num_records=records,
        exec_options=exec_options,
        storage=storage,
        return_records=bool(return_records),
        capture_step_history=bool(capture_step_history),
        step_history_capacity=int(step_history_capacity) if capture_step_history else 0,
        record_z=resolved_record_z,
        progress_callback=progress_callback,
        meta_overrides={"preset": preset, "coupled": coupled},
    )
