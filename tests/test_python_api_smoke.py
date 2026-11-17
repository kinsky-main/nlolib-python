import math
import os
import tempfile

import nlolib
from nlolib import (
    ExecutionOptions,
    NlolibError,
    PropagationAbortedError,
    PulseSpec,
    StepHistory,
    propagate,
)

CPU = ExecutionOptions(backend_type="cpu")
AUTO = ExecutionOptions(backend_type="auto")
VULKAN = ExecutionOptions(backend_type="vulkan")


def _max_abs_diff(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def _l2_norm(field):
    return sum((v.real * v.real) + (v.imag * v.imag) for v in field) ** 0.5


def _identity(pulse, num_records, exec_options, **kwargs):
    """Zero linear and nonlinear operators: the field must come back unchanged."""
    options = dict(
        propagation_distance=0.02,
        starting_step_size=1e-3,
        max_step_size=2e-3,
        min_step_size=1e-5,
        error_tolerance=1e-7,
        num_records=num_records,
        exec_options=exec_options,
    )
    options.update(kwargs)
    return propagate(pulse, lambda A, w: 0.0, lambda A, I: 0.0, **options)


def _expect(exc_type, substring, func):
    try:
        func()
    except exc_type as exc:
        assert substring in str(exc), f"missing '{substring}' in '{exc}'"
        return exc
    raise AssertionError(f"expected {exc_type.__name__} containing '{substring}'")


def _runtime_logging() -> None:
    nlolib.set_log_buffer(128 * 1024)
    nlolib.set_log_level("info")
    nlolib.set_progress_options(enabled=True, milestone_percent=20, emit_on_step_adjust=True)
    nlolib.set_progress_stream("both")
    nlolib.set_progress_stream("stderr")
    nlolib.clear_log_buffer()
    _expect(ValueError, "level", lambda: nlolib.set_log_level(2))
    _expect(ValueError, "stream_mode", lambda: nlolib.set_progress_stream("stdin"))

    log_dir = tempfile.mkdtemp(prefix="nlolib-log-")
    log_path = os.path.join(log_dir, "run.log")
    nlolib.set_log_file(log_path, append=False)
    nlolib.set_log_file(None)
    assert os.path.isfile(log_path)
    assert os.path.isfile(nlolib.loaded_library_path())
    print("test_python_api_smoke: runtime log API configured.")


def main():
    _runtime_logging()

    limits = nlolib.query_runtime_limits(exec_options=CPU)
    assert isinstance(limits, nlolib.RuntimeLimits)
    assert int(limits.max_num_time_samples_runtime) > 0
    assert int(limits.max_num_recorded_samples_with_storage) > 0

    n = 128
    pulse_zero = PulseSpec(samples=[0j] * n, delta_time=0.001, pulse_period=1.0,
                           frequency_grid=[0j] * n)
    out = _identity(pulse_zero, 4, CPU)
    assert len(out.records) == 4
    assert len(out.records[0]) == n
    log_text = nlolib.read_log_buffer(consume=True)
    assert "propagate request" in log_text
    assert "field_size" in log_text
    assert out.meta == {
        "preset": "balanced",
        "records_requested": 4,
        "records_written": 4,
        "backend_requested": "cpu",
        "coupled": False,
        "status": "ok",
    }, out.meta
    assert out.step_history is None and out.storage_result is None
    print("test_python_api_smoke: CPU 1D propagation returned expected record-major shape.")

    gaussian = [complex(math.exp(-((i - 0.5 * n) / 18.0) ** 2), 0.0) for i in range(n)]
    pulse = PulseSpec(samples=gaussian, delta_time=0.001, pulse_period=1.0, frequency_grid=[0j] * n)
    identity_records = _identity(pulse, 3, AUTO).records
    baseline_norm = _l2_norm(identity_records[0])
    final_norm = _l2_norm(identity_records[-1])
    rel_drift = abs(final_norm - baseline_norm) / max(baseline_norm, 1e-12)
    assert rel_drift <= 1e-6, f"AUTO identity propagation drift too large: {rel_drift}"
    print("test_python_api_smoke: AUTO identity propagation preserved field norm.")

    explicit_vulkan_records = _identity(pulse, 3, VULKAN).records
    assert len(explicit_vulkan_records) == 3
    assert len(explicit_vulkan_records[0]) == n
    print("test_python_api_smoke: explicit VULKAN defaults now resolve a usable backend.")

    step_history = _identity(pulse, 3, AUTO, capture_step_history=True,
                             step_history_capacity=20000).step_history
    assert isinstance(step_history, StepHistory)
    assert isinstance(step_history.z, list)
    assert isinstance(step_history.step_size, list)
    assert isinstance(step_history.next_step_size, list)
    assert isinstance(step_history.dropped, int)
    assert step_history.capacity == 20000
    assert len(step_history.z) > 0
    assert step_history.dropped >= 0
    print("test_python_api_smoke: optional step history capture returned structured telemetry.")

    events = []

    def abort_on_first_progress(info):
        events.append(info.event_type)
        return False

    exc = _expect(
        PropagationAbortedError,
        "aborted",
        lambda: _identity(pulse, 3, AUTO, capture_step_history=True, step_history_capacity=128,
                          progress_callback=abort_on_first_progress),
    )
    assert isinstance(exc, NlolibError) and exc.status == "aborted"
    assert events and all(event in {"accepted", "rejected", "finish"} for event in events), events
    assert exc.result.meta["status"] == "aborted"
    assert int(exc.result.meta["records_written"]) >= 0
    print("test_python_api_smoke: progress callback can abort propagation cleanly.")

    nx = 16
    ny = 8
    nxy = nx * ny
    pulse_2d = PulseSpec(samples=[0j] * nxy, delta_time=0.001, pulse_period=1.0,
                         frequency_grid=[0j], tensor_nt=1, tensor_nx=nx, tensor_ny=ny,
                         potential_grid=[1 + 0j] * nxy)
    out_2d = _identity(pulse_2d, 1, CPU)
    assert len(out_2d.records) == 1
    assert len(out_2d.records[0]) == nxy
    assert out_2d.z_axis == [0.02]
    assert out_2d.meta["coupled"] is True
    print("test_python_api_smoke: tensor 2D propagation returned expected shape.")

    nt, nx3, ny3 = 4, 4, 2
    n3 = nt * nx3 * ny3
    pulse_3d = PulseSpec(samples=[complex(i, 0.0) for i in range(n3)], delta_time=0.001,
                         pulse_period=1.0, frequency_grid=[0j] * nt, tensor_nt=nt,
                         tensor_nx=nx3, tensor_ny=ny3, potential_grid=[0j] * n3)
    out_3d = _identity(pulse_3d, 1, CPU)
    assert len(out_3d.records) == 1
    assert len(out_3d.records[0]) == n3
    view = nlolib.tensor_record_view(pulse_3d.samples, nt, nx3, ny3)
    assert len(view) == nt and len(view[0]) == ny3 and len(view[0][0]) == nx3
    for t in range(nt):
        for y in range(ny3):
            for x in range(nx3):
                assert view[t][y][x] == pulse_3d.samples[((x * ny3) + y) * nt + t]
    _expect(ValueError, "nt*nx*ny", lambda: nlolib.tensor_record_view([0j] * 5, nt, nx3, ny3))
    print("test_python_api_smoke: explicit 3+1D layout propagation returned expected shape.")

    _expect(
        ValueError,
        "len(pulse.samples) must equal pulse.tensor_nt * pulse.tensor_nx * pulse.tensor_ny",
        lambda: _identity(PulseSpec(samples=[0j] * nxy, delta_time=0.001, frequency_grid=[0j, 0j],
                                    tensor_nt=2, tensor_nx=nx, tensor_ny=ny), 1, CPU),
    )
    print("test_python_api_smoke: inconsistent tensor shape rejected as expected.")

    sim_result = propagate(
        pulse_zero,
        lambda A, w: 1j * -0.5 * w * w,
        lambda A, I: 1j * A * I,
        propagation_distance=0.02,
        preset="balanced",
        exec_options=CPU,
    )
    assert len(sim_result.records) == 128
    assert len(sim_result.records[0]) == n
    assert len(sim_result.z_axis) == 128
    assert _max_abs_diff(sim_result.final, sim_result.records[-1]) == 0.0
    print("test_python_api_smoke: propagate returned the preset's dense output.")

    beta2, gamma = 99.0, 99.0  # captured, but constant_bindings wins
    explicit_result = propagate(
        pulse_zero,
        lambda A, w: 1j * beta2 * w * w,
        lambda A, I: 1j * gamma * A * I,
        constant_bindings={"beta2": -0.5, "gamma": 1.0},
        propagation_distance=0.02,
        num_records=16,
        exec_options=CPU,
    )
    assert len(explicit_result.records) == 16
    assert len(explicit_result.records[0]) == n
    print("test_python_api_smoke: constant_bindings path returned expected shape.")

    record_z = [0.0, 0.005, 0.02]
    at_z = _identity(pulse, None, CPU, record_z=record_z)
    assert at_z.z_axis == record_z and len(at_z.records) == 3
    # One explicit position returns the field there, not the final field, and
    # `final` is that last returned record. A dispersive run, so positions
    # differ: the single-position record must match the same position of a
    # three-position run.
    chirped = PulseSpec(
        samples=[g * complex(math.cos(40.0 * (i - 0.5 * n) / n), math.sin(40.0 * (i - 0.5 * n) / n))
                 for i, g in enumerate(gaussian)],
        delta_time=0.001,
        pulse_period=n * 0.001,
    )
    dispersive = dict(propagation_distance=0.02, error_tolerance=1e-10, exec_options=CPU)
    three = propagate(chirped, lambda A, w: 1j * 2.0e-4 * w * w, lambda A, I: 0.0,
                      record_z=[0.0, 0.005, 0.02], **dispersive)
    one = propagate(chirped, lambda A, w: 1j * 2.0e-4 * w * w, lambda A, I: 0.0,
                    record_z=[0.005], **dispersive)
    assert one.z_axis == [0.005] and len(one.records) == 1, one.z_axis
    assert one.final == one.records[0]
    at_005 = _max_abs_diff(one.records[0], three.records[1])
    vs_end = _max_abs_diff(one.records[0], three.records[2])
    assert at_005 <= 1e-8, at_005
    assert vs_end >= 1e-3, f"the 0.005 record is the final field: {vs_end}"
    _expect(ValueError, "disagrees with len(record_z)",
            lambda: _identity(pulse, 2, CPU, record_z=record_z))
    print("test_python_api_smoke: record_z returns exactly the requested positions.")

    nt_c, nx_c, ny_c = 4, 4, 2
    n_c = nt_c * nx_c * ny_c
    pulse_coupled = PulseSpec(samples=[0j] * n_c, delta_time=0.001, pulse_period=1.0,
                              tensor_nt=nt_c, tensor_nx=nx_c, tensor_ny=ny_c,
                              frequency_grid=[0j] * nt_c)
    coupled_result = propagate(
        pulse_coupled,
        lambda A, wt, kx, ky: 1j * (beta2 * wt * wt + beta_t * (kx * kx + ky * ky)),
        lambda A, I, V: 1j * A * (gamma * I + V),
        potential_operator=lambda x, y: 0.0 * (x * x + y * y),
        constant_bindings={"beta2": 0.0, "beta_t": 0.0, "gamma": 0.0},
        propagation_distance=0.01,
        num_records=2,
        exec_options=CPU,
    )
    assert len(coupled_result.records) == 2
    assert len(coupled_result.records[0]) == n_c
    assert coupled_result.meta["coupled"] is True
    print("test_python_api_smoke: coupled tensor propagate returned expected shape.")

    _expect(ValueError, "only used by tensor runs",
            lambda: _identity(pulse, 2, CPU, potential_operator=lambda x, y: x))
    _expect(ValueError, "potential_operator and pulse.potential_grid",
            lambda: _identity(pulse_3d, 2, CPU, potential_operator=lambda x, y: x))
    print("test_python_api_smoke: potential_operator rejected where the core would ignore it.")

    dense_record_count = 257
    dense_history_result = _identity(pulse, dense_record_count, CPU, capture_step_history=True,
                                     step_history_capacity=20000)
    dense_steps = [float(v) for v in dense_history_result.step_history.step_size]
    assert len(dense_steps) > 0
    dense_spacing = 0.02 / float(dense_record_count - 1)
    assert max(dense_steps) > (4.0 * dense_spacing)
    assert len(dense_history_result.step_history.z) < dense_record_count
    assert len(dense_history_result.records) == dense_record_count
    print("test_python_api_smoke: adaptive step sizes are no longer capped by record spacing.")

    fixed = dict(propagation_distance=0.01, starting_step_size=1e-3, max_step_size=1e-3,
                 min_step_size=1e-3, error_tolerance=1e-7)
    fixed_aligned = _identity(pulse, 11, CPU, capture_step_history=True,
                              step_history_capacity=128, **fixed)
    assert len(fixed_aligned.records) == 11
    assert len(fixed_aligned.z_axis) == 11
    assert int(fixed_aligned.meta["records_written"]) == 11
    fixed_history = fixed_aligned.step_history
    assert len(fixed_history.step_size) > 0
    assert all(abs(v) <= 1e-15 for v in fixed_history.error)
    for a, b in zip(fixed_history.step_size, fixed_history.next_step_size):
        assert abs(a - b) <= 1e-15

    fixed_oversubscribed = _identity(pulse, 64, CPU, **fixed)
    assert len(fixed_oversubscribed.records) == 11
    assert len(fixed_oversubscribed.z_axis) == 11
    assert int(fixed_oversubscribed.meta["records_requested"]) == 64
    assert int(fixed_oversubscribed.meta["records_written"]) == 11
    print("test_python_api_smoke: fixed-step oversubscribed records are clamped to step-aligned samples.")

    # Unknown option names are Python's own TypeError; removed spellings included.
    for removed in ("records", "t_eval", "sqlite_path", "output", "first_step", "tensor_nt"):
        _expect(TypeError, removed, lambda: _identity(pulse, 2, CPU, **{removed: 1}))  # noqa: B023
    _expect(TypeError, "propagation_distance",
            lambda: propagate(pulse, lambda A, w: 0.0, lambda A, I: 0.0))
    print("test_python_api_smoke: unknown and removed option names rejected with TypeError.")

    # Every enum-style option is a lowercase string; anything else is a ValueError.
    for bad in (
        dict(exec_options=ExecutionOptions(backend_type="gpu")),
        dict(exec_options=ExecutionOptions(backend_type=0)),
        dict(exec_options=ExecutionOptions(fft_backend="cufft")),
        dict(nonlinear_model="raman"),
        dict(preset="slow"),
    ):
        _expect(ValueError, "", lambda: _identity(pulse, 2, CPU, **bad) if "exec_options" not in bad
                else _identity(pulse, 2, bad["exec_options"]))  # noqa: B023
    print("test_python_api_smoke: unknown enum strings rejected with ValueError.")

    for label, operator in (
        ("preset string", "gvd"),
        ("expression string", "i*beta2*w*w"),
        ("bytes", b"i*w"),
        ("mapping", {"fn": lambda A, w: 0.0}),
        ("number", 0.0),
    ):
        _expect(TypeError, "linear_operator",
                lambda: propagate(pulse, operator, lambda A, I: 0.0, propagation_distance=0.02,  # noqa: B023
                                  exec_options=CPU))
    _expect(TypeError, "nonlinear_operator",
            lambda: propagate(pulse, lambda A, w: 0.0, "i*A*I", propagation_distance=0.02))
    _expect(TypeError, "linear_step_operator",
            lambda: _identity(pulse, 2, CPU, linear_step_operator="exp(h*D)"))
    _expect(TypeError, "nonlinear_operator",
            lambda: propagate(pulse, lambda A, w: 0.0, propagation_distance=0.02))
    print("test_python_api_smoke: string and missing operators rejected with TypeError.")

    _expect(ValueError, "unknown identifier 'w'",
            lambda: propagate(pulse, lambda A, w: 0.0, lambda A, I: w * A,  # noqa: F821
                              propagation_distance=0.02))
    _expect(ValueError, "pulse must be a PulseSpec",
            lambda: propagate({"samples": gaussian, "delta_time": 0.001}, lambda A, w: 0.0,
                              lambda A, I: 0.0, propagation_distance=0.02))
    _expect(ValueError, "pulse.delta_time", lambda: _identity(PulseSpec(samples=gaussian, delta_time=0.0), 2, CPU))
    print("test_python_api_smoke: untranslatable operators and bad pulses rejected with ValueError.")


if __name__ == "__main__":
    main()
