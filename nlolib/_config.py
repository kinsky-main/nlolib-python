"""
Simulation config preparation and runtime operator normalization.
"""

from __future__ import annotations

import ctypes
import re
from dataclasses import dataclass
from typing import Mapping, Sequence

from ._binding import (
    NONLINEAR_MODEL_EXPR,
    RUNTIME_OPERATOR_CONSTANTS_MAX,
    TENSOR_LAYOUT_XYT_T_FAST,
    NloComplex,
    NloPhysicsConfig,
    NloProgressInfo,
    NloSimulationConfig,
    NloStepEvent,
    NloStorageResult,
    make_complex_array,
)
from ._models import OperatorSpec, ProgressInfo, PulseSpec, RuntimeOperators
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


def _normalize_pulse_spec(pulse: PulseSpec | Mapping[str, object]) -> PulseSpec:
    if isinstance(pulse, PulseSpec):
        spec = pulse
    elif isinstance(pulse, Mapping):
        if "samples" not in pulse or "delta_time" not in pulse:
            raise ValueError("pulse mapping must define 'samples' and 'delta_time'")
        for legacy_key in ("time_nt", "spatial_nx", "spatial_ny"):
            if pulse.get(legacy_key) is not None:
                raise ValueError(
                    f"pulse.{legacy_key} has been removed; use tensor_nt/tensor_nx/tensor_ny instead"
                )
        spec = PulseSpec(
            samples=pulse["samples"],  # type: ignore[arg-type]
            delta_time=float(pulse["delta_time"]),  # type: ignore[arg-type]
            pulse_period=float(pulse["pulse_period"]) if pulse.get("pulse_period") is not None else None,
            frequency_grid=pulse.get("frequency_grid"),  # type: ignore[arg-type]
            tensor_nt=int(pulse["tensor_nt"]) if pulse.get("tensor_nt") is not None else None,
            tensor_nx=int(pulse["tensor_nx"]) if pulse.get("tensor_nx") is not None else None,
            tensor_ny=int(pulse["tensor_ny"]) if pulse.get("tensor_ny") is not None else None,
            tensor_layout=int(pulse.get("tensor_layout", TENSOR_LAYOUT_XYT_T_FAST)),
            delta_x=float(pulse.get("delta_x", 1.0)),
            delta_y=float(pulse.get("delta_y", 1.0)),
            spatial_frequency_grid=pulse.get("spatial_frequency_grid"),  # type: ignore[arg-type]
            potential_grid=pulse.get("potential_grid"),  # type: ignore[arg-type]
        )
    else:
        raise ValueError("pulse must be a PulseSpec or mapping")
    if len(spec.samples) == 0:
        raise ValueError("pulse.samples must be non-empty")
    if spec.delta_time <= 0.0:
        raise ValueError("pulse.delta_time must be > 0")
    if spec.tensor_nt is not None and int(spec.tensor_nt) <= 0:
        raise ValueError("pulse.tensor_nt must be > 0 when provided")
    return spec


def _validate_coupled_pulse_spec(pulse: PulseSpec) -> None:
    if pulse.tensor_nt is None or int(pulse.tensor_nt) <= 0:
        raise ValueError("pulse.tensor_nt must be > 0 for coupled tensor simulations")
    if pulse.tensor_nx is None or int(pulse.tensor_nx) <= 0:
        raise ValueError("pulse.tensor_nx must be > 0 for coupled tensor simulations")
    if pulse.tensor_ny is None or int(pulse.tensor_ny) <= 0:
        raise ValueError("pulse.tensor_ny must be > 0 for coupled tensor simulations")

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
            "records": 128,
        }
    if profile == "fast":
        return {
            "starting_step_size": propagation_distance / 120.0,
            "max_step_size": propagation_distance / 12.0,
            "min_step_size": propagation_distance / 4000.0,
            "error_tolerance": 5e-6,
            "records": 64,
        }
    if profile == "accuracy":
        return {
            "starting_step_size": propagation_distance / 400.0,
            "max_step_size": propagation_distance / 50.0,
            "min_step_size": propagation_distance / 80000.0,
            "error_tolerance": 1e-7,
            "records": 192,
        }
    raise ValueError(f"unsupported preset '{profile}'")


def _reject_string_operator(name: str, operator: object) -> None:
    if isinstance(operator, (str, bytes)):
        raise TypeError(
            f"{name} must be a callable, not a string; expression strings are no longer "
            "accepted (write e.g. lambda A, w: 1j * beta2 * w * w)"
        )


def _coerce_operator_spec(name: str, operator: object) -> OperatorSpec:
    _reject_string_operator(name, operator)
    if isinstance(operator, OperatorSpec):
        spec = operator
    elif callable(operator):
        spec = OperatorSpec(fn=operator)
    elif isinstance(operator, Mapping):
        if "expr" in operator:
            raise TypeError(f"{name}: 'expr' is no longer accepted; pass a callable as 'fn'")
        spec = OperatorSpec(fn=operator.get("fn"), params=operator.get("params"))  # type: ignore[arg-type]
    else:
        raise TypeError(f"{name} must be a callable or an OperatorSpec, got {type(operator).__name__}")
    _reject_string_operator(f"{name}.fn", spec.fn)
    if not callable(spec.fn):
        raise TypeError(f"{name}.fn must be callable")
    if spec.params is not None and not isinstance(spec.params, Mapping):
        raise TypeError(f"{name}.params must be a mapping of name to value")
    return spec


def _resolve_operator_spec(
    name: str,
    operator: object,
) -> tuple[object, Mapping[str, float] | None]:
    spec = _coerce_operator_spec(name, operator)
    params = spec.params
    bindings = {key: float(value) for key, value in params.items()} if params is not None else None
    return spec.fn, bindings


def _decode_storage_run_id(raw: bytes) -> str:
    return raw.split(b"\x00", 1)[0].decode("utf-8", errors="replace")


def _storage_result_to_meta(storage_result: NloStorageResult) -> dict[str, object]:
    return {
        "run_id": _decode_storage_run_id(bytes(storage_result.run_id)),
        "records_captured": int(storage_result.records_captured),
        "records_spilled": int(storage_result.records_spilled),
        "chunks_written": int(storage_result.chunks_written),
        "db_size_bytes": int(storage_result.db_size_bytes),
        "truncated": bool(storage_result.truncated),
    }


def _step_events_to_meta(step_events: ctypes.Array, count: int) -> dict[str, object]:
    n = max(0, int(count))
    step_index: list[int] = [0] * n
    z: list[float] = [0.0] * n
    step_size: list[float] = [0.0] * n
    next_step_size: list[float] = [0.0] * n
    error: list[float] = [0.0] * n
    for i in range(n):
        event = step_events[i]
        step_index[i] = int(event.step_index)
        z[i] = float(event.z_current)
        step_size[i] = float(event.step_size)
        next_step_size[i] = float(event.next_step_size)
        error[i] = float(event.error)
    return {
        "step_index": step_index,
        "z": z,
        "step_size": step_size,
        "next_step_size": next_step_size,
        "error": error,
    }


def _progress_info_from_struct(info: NloProgressInfo) -> ProgressInfo:
    return ProgressInfo(
        event_type=int(info.event_type),
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
    )


def _validate_explicit_record_z(z_values: Sequence[float], distance: float) -> None:
    if len(z_values) <= 0:
        raise ValueError("t_eval must be non-empty")
    prev = float(z_values[0])
    if prev < 0.0 or prev > float(distance):
        raise ValueError("t_eval values must be within [0, propagation_distance]")
    for i in range(1, len(z_values)):
        current = float(z_values[i])
        if current < prev:
            raise ValueError("t_eval must be monotonic nondecreasing")
        if current < 0.0 or current > float(distance):
            raise ValueError("t_eval values must be within [0, propagation_distance]")
        prev = current


def _shift_constant_indices(expression: str, offset: int) -> str:
    if offset == 0:
        return expression

    def repl(match: re.Match[str]) -> str:
        return f"c{int(match.group(1)) + offset}"

    return re.sub(r"\bc(\d+)\b", repl, expression)


@dataclass
class PreparedSimConfig:
    simulation_config: NloSimulationConfig
    physics_config: NloPhysicsConfig
    keepalive: list[object]
    #: Total sample count this config was prepared for. The C `simulation_config`
    #: does not carry it: `time.nt` is left at zero for a 1-D run, which the
    #: library reads as "infer the layout from the sample count passed to
    #: propagate", and its frequency grid is a bare pointer with no length. So a
    #: caller that needs the count after the fact has nowhere to read it, which
    #: is why `query_runtime_limits` needs this.
    num_time_samples: int = 0

    @property
    def sim_ptr(self) -> ctypes.POINTER(NloSimulationConfig):
        return ctypes.pointer(self.simulation_config)

    @property
    def physics_ptr(self) -> ctypes.POINTER(NloPhysicsConfig):
        return ctypes.pointer(self.physics_config)


def prepare_sim_config(
    num_time_samples: int,
    *,
    propagation_distance: float,
    starting_step_size: float,
    max_step_size: float,
    min_step_size: float,
    error_tolerance: float,
    pulse_period: float,
    delta_time: float,
    tensor_nt: int | None = None,
    tensor_nx: int | None = None,
    tensor_ny: int | None = None,
    tensor_layout: int = TENSOR_LAYOUT_XYT_T_FAST,
    frequency_grid: Sequence[complex],
    wt_axis: Sequence[complex] | None = None,
    delta_x: float = 1.0,
    delta_y: float = 1.0,
    kx_axis: Sequence[complex] | None = None,
    ky_axis: Sequence[complex] | None = None,
    spatial_frequency_grid: Sequence[complex] | None = None,
    potential_grid: Sequence[complex] | None = None,
    runtime: RuntimeOperators | None = None,
) -> PreparedSimConfig:
    """
    Build a fully-initialized sim_config plus keepalive storage.
    """
    if num_time_samples <= 0:
        raise ValueError("num_time_samples must be > 0")

    tensor_mode = tensor_nt is not None and int(tensor_nt) > 0
    resolved_nt = int(num_time_samples)
    resolved_nx = 1
    resolved_ny = 1
    if tensor_mode:
        if tensor_nx is None or tensor_ny is None:
            raise ValueError("tensor_nx and tensor_ny are required when tensor_nt is set")
        resolved_nt = int(tensor_nt)
        resolved_nx = int(tensor_nx)
        resolved_ny = int(tensor_ny)
        if resolved_nt <= 0 or resolved_nx <= 0 or resolved_ny <= 0:
            raise ValueError("tensor_nt/tensor_nx/tensor_ny must be positive")
        if (resolved_nt * resolved_nx * resolved_ny) != int(num_time_samples):
            raise ValueError("tensor_nt * tensor_nx * tensor_ny must match num_time_samples")

    if len(frequency_grid) not in {resolved_nt, num_time_samples}:
        raise ValueError(
            "frequency_grid length must match resolved nt (or num_time_samples for full-volume grids)"
        )

    sim_cfg = NloSimulationConfig()
    physics_cfg = NloPhysicsConfig()
    keepalive: list[object] = []

    sim_cfg.propagation.propagation_distance = float(propagation_distance)
    sim_cfg.propagation.starting_step_size = float(starting_step_size)
    sim_cfg.propagation.max_step_size = float(max_step_size)
    sim_cfg.propagation.min_step_size = float(min_step_size)
    sim_cfg.propagation.error_tolerance = float(error_tolerance)

    if tensor_mode:
        sim_cfg.tensor.nt = resolved_nt
        sim_cfg.tensor.nx = resolved_nx
        sim_cfg.tensor.ny = resolved_ny
        sim_cfg.tensor.layout = int(tensor_layout)
    else:
        sim_cfg.tensor.nt = 0
        sim_cfg.tensor.nx = 0
        sim_cfg.tensor.ny = 0
        sim_cfg.tensor.layout = int(TENSOR_LAYOUT_XYT_T_FAST)

    sim_cfg.time.nt = 0
    sim_cfg.time.pulse_period = float(pulse_period)
    sim_cfg.time.delta_time = float(delta_time)
    sim_cfg.time.wt_axis = None

    freq_arr = make_complex_array(frequency_grid)
    keepalive.append(freq_arr)
    sim_cfg.frequency.frequency_grid = ctypes.cast(freq_arr, ctypes.POINTER(NloComplex))

    if tensor_mode:
        sim_cfg.spatial.nx = resolved_nx
        sim_cfg.spatial.ny = resolved_ny
    else:
        sim_cfg.spatial.nx = 0
        sim_cfg.spatial.ny = 0
    sim_cfg.spatial.delta_x = float(delta_x)
    sim_cfg.spatial.delta_y = float(delta_y)
    sim_cfg.spatial.kx_axis = None
    sim_cfg.spatial.ky_axis = None

    axis_nt = resolved_nt
    if wt_axis is not None:
        if len(wt_axis) != axis_nt:
            raise ValueError("wt_axis length must match resolved nt")
        wt_arr = make_complex_array(wt_axis)
        keepalive.append(wt_arr)
        sim_cfg.time.wt_axis = ctypes.cast(wt_arr, ctypes.POINTER(NloComplex))
    if kx_axis is not None:
        if len(kx_axis) != resolved_nx:
            raise ValueError("kx_axis length must match resolved nx")
        kx_arr = make_complex_array(kx_axis)
        keepalive.append(kx_arr)
        sim_cfg.spatial.kx_axis = ctypes.cast(kx_arr, ctypes.POINTER(NloComplex))
    if ky_axis is not None:
        if len(ky_axis) != resolved_ny:
            raise ValueError("ky_axis length must match resolved ny")
        ky_arr = make_complex_array(ky_axis)
        keepalive.append(ky_arr)
        sim_cfg.spatial.ky_axis = ctypes.cast(ky_arr, ctypes.POINTER(NloComplex))

    resolved_potential_grid = potential_grid
    if tensor_mode:
        xy_points = resolved_nx * resolved_ny
        if spatial_frequency_grid is not None and len(spatial_frequency_grid) not in {xy_points, num_time_samples}:
            raise ValueError(
                "spatial_frequency_grid length must match tensor_nx*tensor_ny "
                "(or num_time_samples for full-volume grids)"
            )
        if potential_grid is not None:
            potential_len = len(potential_grid)
            if potential_len == xy_points:
                resolved_potential_grid = [complex(value) for value in potential_grid for _ in range(resolved_nt)]
            elif potential_len == num_time_samples:
                resolved_potential_grid = potential_grid
            else:
                raise ValueError(
                    "potential_grid length must match num_time_samples "
                    "(or tensor_nx*tensor_ny for static XY broadcast) when tensor mode is enabled"
                )
    else:
        if spatial_frequency_grid is not None and len(spatial_frequency_grid) != num_time_samples:
            raise ValueError("spatial_frequency_grid length must match num_time_samples for temporal runs")
        if potential_grid is not None and len(potential_grid) != num_time_samples:
            raise ValueError("potential_grid length must match num_time_samples for temporal runs")

    if spatial_frequency_grid is not None:
        spatial_arr = make_complex_array(spatial_frequency_grid)
        keepalive.append(spatial_arr)
        sim_cfg.spatial.spatial_frequency_grid = ctypes.cast(spatial_arr, ctypes.POINTER(NloComplex))
    else:
        sim_cfg.spatial.spatial_frequency_grid = None

    if resolved_potential_grid is not None:
        potential_arr = make_complex_array(resolved_potential_grid)
        keepalive.append(potential_arr)
        sim_cfg.spatial.potential_grid = ctypes.cast(potential_arr, ctypes.POINTER(NloComplex))
    else:
        sim_cfg.spatial.potential_grid = None

    if runtime is not None:
        constants = [float(value) for value in runtime.constants]
        constant_bindings = runtime.constant_bindings
        auto_capture = bool(runtime.auto_capture_constants)

        def lower(name: str, fn: object, context: str) -> str | None:
            if fn is None:
                return None
            _reject_string_operator(f"runtime.{name}", fn)
            if not callable(fn):
                raise TypeError(f"runtime.{name} must be callable")
            translated = translate_callable(
                fn,
                context,
                constant_bindings=constant_bindings,
                auto_capture=auto_capture,
            )
            shifted = _shift_constant_indices(translated.expression, len(constants))
            constants.extend(translated.constants)
            return shifted

        physics_cfg.runtime.nonlinear_model = int(runtime.nonlinear_model)
        physics_cfg.runtime.nonlinear_gamma = float(runtime.nonlinear_gamma)
        physics_cfg.runtime.raman_fraction = float(runtime.raman_fraction)
        physics_cfg.runtime.raman_tau1 = float(runtime.raman_tau1)
        physics_cfg.runtime.raman_tau2 = float(runtime.raman_tau2)
        physics_cfg.runtime.shock_omega0 = float(runtime.shock_omega0)
        if runtime.raman_response_time is not None:
            raman_response_arr = make_complex_array(runtime.raman_response_time)
            keepalive.append(raman_response_arr)
            physics_cfg.runtime.raman_response_time = ctypes.cast(
                raman_response_arr,
                ctypes.POINTER(NloComplex),
            )
            physics_cfg.runtime.raman_response_len = len(runtime.raman_response_time)
        else:
            physics_cfg.runtime.raman_response_time = None
            physics_cfg.runtime.raman_response_len = 0

        if runtime.linear_factor_fn is not None and runtime.dispersion_factor_fn is not None:
            raise ValueError("runtime.linear_factor_fn and runtime.dispersion_factor_fn are aliases; provide only one")
        if runtime.linear_fn is not None and runtime.dispersion_fn is not None:
            raise ValueError("runtime.linear_fn and runtime.dispersion_fn are aliases; provide only one")

        if runtime.linear_factor_fn is not None:
            linear_factor_expr = lower("linear_factor_fn", runtime.linear_factor_fn, RUNTIME_CONTEXT_LINEAR_FACTOR)
        else:
            linear_factor_expr = lower(
                "dispersion_factor_fn",
                runtime.dispersion_factor_fn,
                RUNTIME_CONTEXT_DISPERSION_FACTOR,
            )
        if runtime.linear_fn is not None:
            linear_expr = lower("linear_fn", runtime.linear_fn, RUNTIME_CONTEXT_LINEAR)
        else:
            linear_expr = lower("dispersion_fn", runtime.dispersion_fn, RUNTIME_CONTEXT_DISPERSION)
        potential_expr = lower("potential_fn", runtime.potential_fn, RUNTIME_CONTEXT_POTENTIAL)
        nonlinear_expr = lower("nonlinear_fn", runtime.nonlinear_fn, RUNTIME_CONTEXT_NONLINEAR)

        if len(constants) > RUNTIME_OPERATOR_CONSTANTS_MAX:
            raise ValueError(
                "runtime.constants plus captured callable constants exceed "
                f"{RUNTIME_OPERATOR_CONSTANTS_MAX}"
            )

        if linear_factor_expr:
            linear_factor_bytes = linear_factor_expr.encode("utf-8")
            keepalive.append(linear_factor_bytes)
            physics_cfg.runtime.linear_factor_expr = ctypes.c_char_p(linear_factor_bytes)
            physics_cfg.runtime.dispersion_factor_expr = ctypes.c_char_p(linear_factor_bytes)
        else:
            physics_cfg.runtime.linear_factor_expr = None
            physics_cfg.runtime.dispersion_factor_expr = None

        if linear_expr:
            linear_bytes = linear_expr.encode("utf-8")
            keepalive.append(linear_bytes)
            physics_cfg.runtime.linear_expr = ctypes.c_char_p(linear_bytes)
            physics_cfg.runtime.dispersion_expr = ctypes.c_char_p(linear_bytes)
        else:
            physics_cfg.runtime.linear_expr = None
            physics_cfg.runtime.dispersion_expr = None

        if potential_expr:
            potential_bytes = potential_expr.encode("utf-8")
            keepalive.append(potential_bytes)
            physics_cfg.runtime.potential_expr = ctypes.c_char_p(potential_bytes)
        else:
            physics_cfg.runtime.potential_expr = None

        if nonlinear_expr:
            nonlin_bytes = nonlinear_expr.encode("utf-8")
            keepalive.append(nonlin_bytes)
            physics_cfg.runtime.nonlinear_expr = ctypes.c_char_p(nonlin_bytes)
        else:
            physics_cfg.runtime.nonlinear_expr = None

        physics_cfg.runtime.num_constants = len(constants)
        for i, constant in enumerate(constants):
            physics_cfg.runtime.constants[i] = float(constant)
    else:
        physics_cfg.runtime.linear_factor_expr = None
        physics_cfg.runtime.linear_expr = None
        physics_cfg.runtime.potential_expr = None
        physics_cfg.runtime.dispersion_factor_expr = None
        physics_cfg.runtime.dispersion_expr = None
        physics_cfg.runtime.nonlinear_expr = None
        physics_cfg.runtime.nonlinear_model = int(NONLINEAR_MODEL_EXPR)
        physics_cfg.runtime.nonlinear_gamma = 0.0
        physics_cfg.runtime.raman_fraction = 0.0
        physics_cfg.runtime.raman_tau1 = 0.0
        physics_cfg.runtime.raman_tau2 = 0.0
        physics_cfg.runtime.shock_omega0 = 0.0
        physics_cfg.runtime.raman_response_time = None
        physics_cfg.runtime.raman_response_len = 0
        physics_cfg.runtime.num_constants = 0

    return PreparedSimConfig(
        simulation_config=sim_cfg,
        physics_config=physics_cfg,
        keepalive=keepalive,
        num_time_samples=int(num_time_samples),
    )
