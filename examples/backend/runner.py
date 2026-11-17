"""Reusable runner utilities for Python examples."""

from __future__ import annotations

import math
import sys
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_API_DIR = REPO_ROOT
if str(PYTHON_API_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_API_DIR))


def centered_time_grid(num_samples: int, delta_time: float) -> np.ndarray:
    """Return a centered time axis with sample spacing ``delta_time``."""
    return (np.arange(num_samples, dtype=np.float64) - 0.5 * float(num_samples - 1)) * delta_time


@dataclass
class SimulationOptions:
    backend: str = "auto"
    fft_backend: str = "auto"
    device_heap_fraction: float = 0.70
    record_ring_target: int = 0
    forced_device_budget_bytes: int = 0

    def backend_type(self) -> str:
        return str(self.backend).strip().lower()

    def to_exec_options(self, nlo):
        return nlo.ExecutionOptions(
            backend_type=self.backend_type(),
            fft_backend=str(self.fft_backend).strip().lower(),
            device_heap_fraction=float(self.device_heap_fraction),
            record_ring_target=int(self.record_ring_target),
            forced_device_budget_bytes=int(self.forced_device_budget_bytes),
        )


@dataclass
class ExampleOperators:
    """Operators and nonlinear-model settings an example hands to the runner.

    Any operator left unset falls back to the runner's default for the run.
    """

    linear_operator: Callable[..., object] | None = None
    nonlinear_operator: Callable[..., object] | None = None
    linear_step_operator: Callable[..., object] | None = None
    potential_operator: Callable[..., object] | None = None
    constant_bindings: dict[str, float] | None = None
    nonlinear_model: str | None = None
    nonlinear_gamma: float | None = None
    raman_fraction: float | None = None
    raman_tau1: float | None = None
    raman_tau2: float | None = None
    shock_omega0: float | None = None
    raman_response_time: Any | None = None


_PROPAGATE_SETTINGS = (
    "linear_step_operator",
    "potential_operator",
    "constant_bindings",
    "nonlinear_model",
    "nonlinear_gamma",
    "raman_fraction",
    "raman_tau1",
    "raman_tau2",
    "shock_omega0",
    "raman_response_time",
)


@dataclass
class TemporalSimulationConfig:
    """
    Example convenience config mapped to operator-default runtime constants.

    When ``runtime`` is omitted, this helper uses the quadratic GLSE mapping
    ``c0 = beta2 / 2``, ``c1 = alpha / 2`` and ``c2 = gamma``.
    """

    gamma: float
    beta2: float
    dt: float
    z_final: float
    num_time_samples: int
    alpha: float = 0.0
    pulse_period: float | None = None
    omega: np.ndarray | None = None
    starting_step_size: float = 1e-4
    max_step_size: float = 1e-2
    min_step_size: float = 1e-6
    error_tolerance: float = 1e-8
    runtime: ExampleOperators | None = None
    honor_solver_controls: bool = False

    def resolved_pulse_period(self) -> float:
        if self.pulse_period is not None:
            return float(self.pulse_period)
        return float(self.num_time_samples) * float(self.dt)

    def resolved_omega(self) -> np.ndarray:
        if self.omega is not None:
            omega = np.asarray(self.omega, dtype=np.float64)
            if int(omega.size) != int(self.num_time_samples):
                raise ValueError("omega grid size must match num_time_samples.")
            return omega
        return 2.0 * math.pi * np.fft.fftfreq(int(self.num_time_samples), d=float(self.dt))


class NloExampleRunner:
    def __init__(self, library_path: str | None = None):
        try:
            import nlolib as nlo
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "nlolib is not available. Ensure "
                "PYTHONPATH includes the repo's python/ directory."
            ) from exc

        self.nlo = nlo
        if library_path is not None:
            os.environ["NLOLIB_LIBRARY"] = str(library_path)
        self.api = nlo
        self.last_result: Any | None = None

    @property
    def last_meta(self) -> dict[str, Any]:
        return dict(self.last_result.meta) if self.last_result is not None else {}

    @staticmethod
    def _settings(runtime_cfg: Any) -> dict[str, Any]:
        if runtime_cfg is None:
            return {}
        settings: dict[str, Any] = {}
        for name in _PROPAGATE_SETTINGS:
            value = getattr(runtime_cfg, name, None)
            if value is not None:
                settings[name] = value
        return settings

    def _propagate(self, pulse, linear_operator, nonlinear_operator, **kwargs):
        try:
            result = self.api.propagate(pulse, linear_operator, nonlinear_operator, **kwargs)
        except self.api.PropagationAbortedError as exc:
            self.last_result = exc.result
            raise
        self.last_result = result
        return result

    def propagate_temporal_records(
        self,
        field0: np.ndarray,
        sim_cfg: TemporalSimulationConfig,
        num_records: int,
        exec_options: SimulationOptions | None = None,
        *,
        capture_step_history: bool = False,
        step_history_capacity: int = 0,
        progress_callback: Callable[[Any], bool | int | None] | None = None,
        storage: Any | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if num_records <= 0:
            raise ValueError("num_records must be positive.")
        if int(step_history_capacity) < 0:
            raise ValueError("step_history_capacity must be >= 0.")
        if capture_step_history and int(step_history_capacity) <= 0:
            raise ValueError("step_history_capacity must be > 0 when capture_step_history is enabled.")

        field = np.asarray(field0, dtype=np.complex128).reshape(-1)
        n = int(field.size)
        if int(sim_cfg.num_time_samples) != n:
            raise ValueError("field length must match sim_cfg.num_time_samples.")

        omega = sim_cfg.resolved_omega()
        runtime_cfg = sim_cfg.runtime

        options = exec_options if exec_options is not None else SimulationOptions()
        opts = options.to_exec_options(self.nlo)

        pulse = self.nlo.PulseSpec(
            samples=field.tolist(),
            delta_time=float(sim_cfg.dt),
            pulse_period=float(sim_cfg.resolved_pulse_period()),
            frequency_grid=[complex(float(om), 0.0) for om in omega],
        )
        if getattr(runtime_cfg, "linear_operator", None) is not None:
            linear_operator = runtime_cfg.linear_operator
        else:
            beta2_scale = 0.5 * float(sim_cfg.beta2)
            loss = 0.5 * float(sim_cfg.alpha)
            linear_operator = lambda A, w: ((1.0j * beta2_scale) * (w * w) - loss)  # noqa: E731
        if getattr(runtime_cfg, "nonlinear_operator", None) is not None:
            nonlinear_operator = runtime_cfg.nonlinear_operator
        else:
            gamma = float(sim_cfg.gamma)
            nonlinear_operator = lambda A, I, V: (1.0j * A) * (gamma * I + V)  # noqa: E731
        result = self._propagate(
            pulse,
            linear_operator,
            nonlinear_operator,
            propagation_distance=float(sim_cfg.z_final),
            num_records=int(num_records),
            exec_options=opts,
            starting_step_size=float(sim_cfg.starting_step_size),
            max_step_size=float(sim_cfg.max_step_size),
            min_step_size=float(sim_cfg.min_step_size),
            error_tolerance=float(sim_cfg.error_tolerance),
            capture_step_history=bool(capture_step_history),
            step_history_capacity=int(step_history_capacity) if capture_step_history else 200000,
            progress_callback=progress_callback,
            storage=storage,
            **self._settings(runtime_cfg),
        )
        z_records = np.asarray(result.z_axis, dtype=np.float64)
        records = np.asarray(result.records, dtype=np.complex128).reshape(int(num_records), n)
        return z_records, records

    def propagate_tensor3d_records(
        self,
        field0_tfast: np.ndarray,
        nt: int,
        nx: int,
        ny: int,
        num_records: int,
        propagation_distance: float,
        starting_step_size: float,
        max_step_size: float,
        min_step_size: float,
        error_tolerance: float,
        delta_x: float,
        delta_y: float,
        delta_time: float = 1.0,
        pulse_period: float | None = None,
        frequency_grid: np.ndarray | None = None,
        potential_grid: np.ndarray | None = None,
        runtime: ExampleOperators | None = None,
        exec_options: SimulationOptions | None = None,
        storage: Any | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if num_records <= 0:
            raise ValueError("num_records must be positive.")

        nt_resolved = int(nt)
        nx_resolved = int(nx)
        ny_resolved = int(ny)
        if nt_resolved <= 0 or nx_resolved <= 0 or ny_resolved <= 0:
            raise ValueError("nt, nx, and ny must be positive.")

        n_total = nt_resolved * nx_resolved * ny_resolved
        field = np.asarray(field0_tfast, dtype=np.complex128).reshape(-1)
        if int(field.size) != n_total:
            raise ValueError("field0_tfast length must equal nt * nx * ny.")

        freq_values = np.zeros(nt_resolved, dtype=np.complex128)
        if frequency_grid is not None:
            freq_values = np.asarray(frequency_grid, dtype=np.complex128).reshape(-1)
            if int(freq_values.size) != nt_resolved:
                raise ValueError("frequency_grid length must equal nt.")

        potential_values = None
        if potential_grid is not None:
            potential_values = np.asarray(potential_grid, dtype=np.complex128).reshape(-1)
            if int(potential_values.size) == (nx_resolved * ny_resolved):
                potential_values = np.tile(potential_values, nt_resolved)
            if int(potential_values.size) != n_total:
                raise ValueError("potential_grid length must equal nt*nx*ny (or nx*ny for static XY broadcast).")

        options = exec_options if exec_options is not None else SimulationOptions()
        opts = options.to_exec_options(self.nlo)

        pulse = self.nlo.PulseSpec(
            samples=field.tolist(),
            delta_time=float(delta_time),
            pulse_period=float(pulse_period) if pulse_period is not None else float(nt_resolved) * float(delta_time),
            tensor_nt=nt_resolved,
            tensor_nx=nx_resolved,
            tensor_ny=ny_resolved,
            frequency_grid=freq_values.tolist(),
            delta_x=float(delta_x),
            delta_y=float(delta_y),
            potential_grid=(None if potential_values is None else potential_values.tolist()),
        )
        if getattr(runtime, "linear_operator", None) is not None:
            linear_operator = runtime.linear_operator
        else:
            linear_operator = lambda A, wt, kx, ky: 0.0  # noqa: E731
        if getattr(runtime, "nonlinear_operator", None) is not None:
            nonlinear_operator = runtime.nonlinear_operator
        else:
            nonlinear_operator = lambda A, I, V: 0.0  # noqa: E731
        result = self._propagate(
            pulse,
            linear_operator,
            nonlinear_operator,
            propagation_distance=float(propagation_distance),
            num_records=int(num_records),
            starting_step_size=float(starting_step_size),
            max_step_size=float(max_step_size),
            min_step_size=float(min_step_size),
            error_tolerance=float(error_tolerance),
            exec_options=opts,
            storage=storage,
            **self._settings(runtime),
        )
        records = np.asarray(
            result.records,
            dtype=np.complex128,
        ).reshape(int(num_records), n_total)
        z_records = np.asarray(result.z_axis, dtype=np.float64)
        return z_records, records
