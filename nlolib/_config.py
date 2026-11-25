"""
Private builder of the C simulation and physics configs.

`propagate` and `query_runtime_limits` build their C structs here from a
`PulseSpec`, the solver settings and the operator callables, which are lowered
to the C expression text for the one field the core compiles for the run type.
"""

from __future__ import annotations

import ctypes
import re
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from . import _enums
from ._binding import (
    RUNTIME_OPERATOR_CONSTANTS_MAX,
    TENSOR_LAYOUT_XYT_T_FAST,
    NloComplex,
    NloPhysicsConfig,
    NloProgressInfo,
    NloSimulationConfig,
    NloStorageResult,
    make_complex_array,
)
from ._models import ProgressInfo, PulseSpec, StepHistory, StorageResult
from ._runtime_expr import (
    RUNTIME_CONTEXT_DISPERSION,
    RUNTIME_CONTEXT_DISPERSION_FACTOR,
    RUNTIME_CONTEXT_LINEAR,
    RUNTIME_CONTEXT_LINEAR_FACTOR,
    RUNTIME_CONTEXT_NONLINEAR,
    RUNTIME_CONTEXT_POTENTIAL,
    translate_callable,
)


def _default_frequency_grid(num_time_samples: int, delta_time: float) -> list[complex]:
    two_pi = 2.0 * 3.141592653589793
    omega_step = two_pi / (float(num_time_samples) * delta_time)
    positive_limit = (num_time_samples - 1) // 2
    out: list[complex] = [0j] * num_time_samples
    for i in range(num_time_samples):
        if i <= positive_limit:
            omega = float(i) * omega_step
        else:
            omega = -(float(num_time_samples - i) * omega_step)
        out[i] = complex(omega, 0.0)
    return out


def _normalize_pulse_spec(pulse: PulseSpec) -> PulseSpec:
    if not isinstance(pulse, PulseSpec):
        raise ValueError(f"pulse must be a PulseSpec, got {type(pulse).__name__}")
    spec = pulse
    if spec.samples is None or len(spec.samples) == 0:
        raise ValueError("pulse.samples must be non-empty")
    if not spec.delta_time or float(spec.delta_time) <= 0.0:
        raise ValueError("pulse.delta_time must be > 0")
    tensor_fields = (spec.tensor_nt, spec.tensor_nx, spec.tensor_ny)
    if any(value is not None for value in tensor_fields):
        _validate_coupled_pulse_spec(spec)
    return spec


def _is_tensor_run(pulse: PulseSpec) -> bool:
    return pulse.tensor_nt is not None


def _validate_coupled_pulse_spec(pulse: PulseSpec) -> None:
    if pulse.tensor_nt is None or int(pulse.tensor_nt) <= 0:
        raise ValueError("pulse.tensor_nt must be > 0 for tensor runs")
    if pulse.tensor_nx is None or int(pulse.tensor_nx) <= 0:
        raise ValueError("pulse.tensor_nx must be > 0 for tensor runs")
    if pulse.tensor_ny is None or int(pulse.tensor_ny) <= 0:
        raise ValueError("pulse.tensor_ny must be > 0 for tensor runs")

    nt = int(pulse.tensor_nt)
    nx = int(pulse.tensor_nx)
    ny = int(pulse.tensor_ny)
    total = nt * nx * ny
    if len(pulse.samples) != total:
        raise ValueError("len(pulse.samples) must equal pulse.tensor_nt * pulse.tensor_nx * pulse.tensor_ny")

    xy = nx * ny
    if pulse.spatial_frequency_grid is not None and len(pulse.spatial_frequency_grid) not in {xy, total}:
        raise ValueError(
            "pulse.spatial_frequency_grid length must match pulse.tensor_nx*pulse.tensor_ny "
            "(or full-volume length)"
        )
    if pulse.potential_grid is not None and len(pulse.potential_grid) not in {xy, total}:
        raise ValueError(
            "pulse.potential_grid length must match pulse.tensor_nx*pulse.tensor_ny "
            "(or full-volume length)"
        )


def _solver_profile_defaults(profile: str, propagation_distance: float) -> dict[str, float | int]:
    if propagation_distance <= 0.0:
        raise ValueError("propagation_distance must be > 0")

    if profile == "balanced":
        return {
            "starting_step_size": propagation_distance / 200.0,
            "max_step_size": propagation_distance / 25.0,
            "min_step_size": propagation_distance / 20000.0,
            "error_tolerance": 1e-6,
            "num_records": 128,
        }
    if profile == "fast":
        return {
            "starting_step_size": propagation_distance / 120.0,
            "max_step_size": propagation_distance / 12.0,
            "min_step_size": propagation_distance / 4000.0,
            "error_tolerance": 5e-6,
            "num_records": 64,
        }
    if profile == "accuracy":
        return {
            "starting_step_size": propagation_distance / 400.0,
            "max_step_size": propagation_distance / 50.0,
            "min_step_size": propagation_distance / 80000.0,
            "error_tolerance": 1e-7,
            "num_records": 192,
        }
    raise ValueError(f"preset must be one of 'fast', 'balanced', 'accuracy', got {profile!r}")


def _require_callable(name: str, operator: object) -> Callable[..., object]:
    if isinstance(operator, (str, bytes)):
        raise TypeError(
            f"{name} must be a callable, not a string; expression strings are not "
            "accepted (write e.g. lambda A, w: 1j * beta2 * w * w)"
        )
    if not callable(operator):
        raise TypeError(f"{name} must be a callable, got {type(operator).__name__}")
    return operator  # type: ignore[return-value]


def _decode_storage_run_id(raw: bytes) -> str:
    return raw.split(b"\x00", 1)[0].decode("utf-8", errors="replace")


def _storage_result(storage_result: NloStorageResult) -> StorageResult:
    return StorageResult(
        run_id=_decode_storage_run_id(bytes(storage_result.run_id)),
        records_captured=int(storage_result.records_captured),
        records_spilled=int(storage_result.records_spilled),
        chunks_written=int(storage_result.chunks_written),
        db_size_bytes=int(storage_result.db_size_bytes),
        truncated=bool(storage_result.truncated),
    )


def _step_history(step_events: ctypes.Array, count: int, dropped: int, capacity: int) -> StepHistory:
    n = max(0, int(count))
    events = [step_events[i] for i in range(n)]
    return StepHistory(
        step_index=[int(event.step_index) for event in events],
        z=[float(event.z_current) for event in events],
        step_size=[float(event.step_size) for event in events],
        next_step_size=[float(event.next_step_size) for event in events],
        error=[float(event.error) for event in events],
        dropped=int(dropped),
        capacity=int(capacity),
    )


def _progress_info_from_struct(info: NloProgressInfo) -> ProgressInfo:
    return ProgressInfo(
        event_type=_enums.progress_event_name(int(info.event_type)),
        step_index=int(info.step_index),
        reject_attempt=int(info.reject_attempt),
        z=float(info.z),
        z_end=float(info.z_end),
        percent=float(info.percent),
        step_size=float(info.step_size),
        next_step_size=float(info.next_step_size),
        error=float(info.error),
        elapsed_seconds=float(info.elapsed_seconds),
        eta_seconds=float(info.eta_seconds),
        accepted_steps=int(info.accepted_steps),
        rejected_steps=int(info.rejected_steps),
        min_step_size=float(info.min_step_size),
        max_step_size=float(info.max_step_size),
    )


def _validate_record_z(z_values: Sequence[float], distance: float) -> None:
    if len(z_values) <= 0:
        raise ValueError("record_z must be non-empty")
    prev = float(z_values[0])
    if prev < 0.0 or prev > float(distance):
        raise ValueError("record_z values must be within [0, propagation_distance]")
    for i in range(1, len(z_values)):
        current = float(z_values[i])
        if current < prev:
            raise ValueError("record_z must be monotonic nondecreasing")
        if current < 0.0 or current > float(distance):
            raise ValueError("record_z values must be within [0, propagation_distance]")
        prev = current


def _shift_constant_indices(expression: str, offset: int) -> str:
    if offset == 0:
        return expression

    def repl(match: re.Match[str]) -> str:
        return f"c{int(match.group(1)) + offset}"

    return re.sub(r"\bc(\d+)\b", repl, expression)


# The C field and translation context each operator lowers to, by run type.
# None: the core does not compile that operator for that run type.
_OPERATOR_SLOTS: dict[str, dict[str, tuple[str, str] | None]] = {
    "linear_operator": {
        "temporal": ("dispersion_factor_expr", RUNTIME_CONTEXT_DISPERSION_FACTOR),
        "tensor": ("linear_factor_expr", RUNTIME_CONTEXT_LINEAR_FACTOR),
    },
    "linear_step_operator": {
        "temporal": ("dispersion_expr", RUNTIME_CONTEXT_DISPERSION),
        "tensor": ("linear_expr", RUNTIME_CONTEXT_LINEAR),
    },
    "potential_operator": {
        "temporal": None,
        "tensor": ("potential_expr", RUNTIME_CONTEXT_POTENTIAL),
    },
    "nonlinear_operator": {
        "temporal": ("nonlinear_expr", RUNTIME_CONTEXT_NONLINEAR),
        "tensor": ("nonlinear_expr", RUNTIME_CONTEXT_NONLINEAR),
    },
}


def _lower_operators(
    operators: Mapping[str, object],
    *,
    tensor_run: bool,
    constant_bindings: Mapping[str, float] | None,
) -> tuple[dict[str, str], list[float]]:
    """Lower each given operator; return {C field: expression} and the constants."""
    bindings = None
    if constant_bindings is not None:
        if not isinstance(constant_bindings, Mapping):
            raise ValueError("constant_bindings must be a mapping of name to value")
        bindings = dict(constant_bindings)
    run_type = "tensor" if tensor_run else "temporal"
    expressions: dict[str, str] = {}
    constants: list[float] = []
    for name, operator in operators.items():
        if operator is None:
            continue
        fn = _require_callable(name, operator)
        slot = _OPERATOR_SLOTS[name][run_type]
        if slot is None:
            raise ValueError(f"{name} is only used by tensor runs; set pulse.tensor_nt/nx/ny")
        c_field, context = slot
        translated = translate_callable(fn, context, constant_bindings=bindings)
        expressions[c_field] = _shift_constant_indices(translated.expression, len(constants))
        constants.extend(translated.constants)
    if len(constants) > RUNTIME_OPERATOR_CONSTANTS_MAX:
        raise ValueError(
            f"the operators use {len(constants)} constants; at most "
            f"{RUNTIME_OPERATOR_CONSTANTS_MAX} are supported"
        )
    return expressions, constants


@dataclass
class _PreparedSimConfig:
    simulation_config: NloSimulationConfig
    physics_config: NloPhysicsConfig
    keepalive: list[object]
    #: Total sample count this config was prepared for. The C config does not
    #: carry it for a temporal run (`time.nt` stays zero, so the solver infers
    #: the layout from the count passed to propagate).
    num_time_samples: int = 0


def _complex_pointer(values: Sequence[complex], keepalive: list[object]):
    array = make_complex_array(values)
    keepalive.append(array)
    return ctypes.cast(array, ctypes.POINTER(NloComplex))


def _prepare_sim_config(
    pulse: PulseSpec,
    *,
    propagation_distance: float,
    starting_step_size: float,
    max_step_size: float,
    min_step_size: float,
    error_tolerance: float,
    expressions: Mapping[str, str] | None = None,
    constants: Sequence[float] = (),
    nonlinear_model: int = 0,
    nonlinear_gamma: float = 0.0,
    raman_fraction: float = 0.0,
    raman_tau1: float = 0.0122,
    raman_tau2: float = 0.0320,
    shock_omega0: float = 0.0,
    raman_response_time: Sequence[complex] | None = None,
) -> _PreparedSimConfig:
    """Build the C simulation and physics configs for one normalized pulse."""
    num_time_samples = len(pulse.samples)
    tensor_run = _is_tensor_run(pulse)
    resolved_nt = int(pulse.tensor_nt) if tensor_run else num_time_samples
    resolved_nx = int(pulse.tensor_nx) if tensor_run else 1
    resolved_ny = int(pulse.tensor_ny) if tensor_run else 1

    pulse_period = (
        float(pulse.pulse_period)
        if pulse.pulse_period is not None
        else float(pulse.delta_time) * float(resolved_nt)
    )
    frequency_grid = (
        list(pulse.frequency_grid)
        if pulse.frequency_grid is not None
        else _default_frequency_grid(resolved_nt, float(pulse.delta_time))
    )
    if len(frequency_grid) not in {resolved_nt, num_time_samples}:
        raise ValueError(
            "pulse.frequency_grid length must match the temporal sample count "
            "(or the full sample count for full-volume grids)"
        )

    sim_cfg = NloSimulationConfig()
    physics_cfg = NloPhysicsConfig()
    keepalive: list[object] = []

    sim_cfg.propagation.propagation_distance = float(propagation_distance)
    sim_cfg.propagation.starting_step_size = float(starting_step_size)
    sim_cfg.propagation.max_step_size = float(max_step_size)
    sim_cfg.propagation.min_step_size = float(min_step_size)
    sim_cfg.propagation.error_tolerance = float(error_tolerance)

    sim_cfg.tensor.nt = resolved_nt if tensor_run else 0
    sim_cfg.tensor.nx = resolved_nx if tensor_run else 0
    sim_cfg.tensor.ny = resolved_ny if tensor_run else 0
    sim_cfg.tensor.layout = int(TENSOR_LAYOUT_XYT_T_FAST)

    sim_cfg.time.nt = 0
    sim_cfg.time.pulse_period = pulse_period
    sim_cfg.time.delta_time = float(pulse.delta_time)
    sim_cfg.time.wt_axis = None
    sim_cfg.frequency.frequency_grid = _complex_pointer(frequency_grid, keepalive)

    sim_cfg.spatial.nx = resolved_nx if tensor_run else 0
    sim_cfg.spatial.ny = resolved_ny if tensor_run else 0
    sim_cfg.spatial.delta_x = float(pulse.delta_x)
    sim_cfg.spatial.delta_y = float(pulse.delta_y)
    sim_cfg.spatial.kx_axis = None
    sim_cfg.spatial.ky_axis = None

    if pulse.wt_axis is not None:
        if len(pulse.wt_axis) != resolved_nt:
            raise ValueError("pulse.wt_axis length must match the temporal sample count")
        sim_cfg.time.wt_axis = _complex_pointer(pulse.wt_axis, keepalive)
    if pulse.kx_axis is not None:
        if len(pulse.kx_axis) != resolved_nx:
            raise ValueError("pulse.kx_axis length must match pulse.tensor_nx")
        sim_cfg.spatial.kx_axis = _complex_pointer(pulse.kx_axis, keepalive)
    if pulse.ky_axis is not None:
        if len(pulse.ky_axis) != resolved_ny:
            raise ValueError("pulse.ky_axis length must match pulse.tensor_ny")
        sim_cfg.spatial.ky_axis = _complex_pointer(pulse.ky_axis, keepalive)

    potential_grid = pulse.potential_grid
    spatial_frequency_grid = pulse.spatial_frequency_grid
    if tensor_run:
        if potential_grid is not None and len(potential_grid) == resolved_nx * resolved_ny:
            # A static XY potential is broadcast along t.
            potential_grid = [complex(value) for value in potential_grid for _ in range(resolved_nt)]
    else:
        if spatial_frequency_grid is not None and len(spatial_frequency_grid) != num_time_samples:
            raise ValueError("pulse.spatial_frequency_grid length must match len(pulse.samples)")
        if potential_grid is not None and len(potential_grid) != num_time_samples:
            raise ValueError("pulse.potential_grid length must match len(pulse.samples)")
    sim_cfg.spatial.spatial_frequency_grid = (
        _complex_pointer(spatial_frequency_grid, keepalive)
        if spatial_frequency_grid is not None
        else None
    )
    sim_cfg.spatial.potential_grid = (
        _complex_pointer(potential_grid, keepalive) if potential_grid is not None else None
    )

    runtime = physics_cfg.runtime
    for c_field in (
        "linear_factor_expr",
        "linear_expr",
        "potential_expr",
        "dispersion_factor_expr",
        "dispersion_expr",
        "nonlinear_expr",
    ):
        text = (expressions or {}).get(c_field)
        if text:
            encoded = text.encode("utf-8")
            keepalive.append(encoded)
            setattr(runtime, c_field, ctypes.c_char_p(encoded))
        else:
            setattr(runtime, c_field, None)
    runtime.nonlinear_model = int(nonlinear_model)
    runtime.nonlinear_gamma = float(nonlinear_gamma)
    runtime.raman_fraction = float(raman_fraction)
    runtime.raman_tau1 = float(raman_tau1)
    runtime.raman_tau2 = float(raman_tau2)
    runtime.shock_omega0 = float(shock_omega0)
    if raman_response_time is not None:
        runtime.raman_response_time = _complex_pointer(raman_response_time, keepalive)
        runtime.raman_response_len = len(raman_response_time)
    else:
        runtime.raman_response_time = None
        runtime.raman_response_len = 0
    runtime.num_constants = len(constants)
    for i, constant in enumerate(constants):
        runtime.constants[i] = float(constant)

    return _PreparedSimConfig(
        simulation_config=sim_cfg,
        physics_config=physics_cfg,
        keepalive=keepalive,
        num_time_samples=int(num_time_samples),
    )
