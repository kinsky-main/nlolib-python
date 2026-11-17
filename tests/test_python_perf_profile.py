import math

import nlolib


def main() -> None:
    nlolib.perf_profile_reset()
    nlolib.perf_profile_set_enabled(True)
    assert nlolib.perf_profile_is_enabled() is True

    n = 64
    dt = 0.01
    omega = [complex(v, 0.0) for v in (2.0 * math.pi * i / (n * dt) for i in range(n))]
    field0 = [complex(math.exp(-((idx - 0.5 * n) / 10.0) ** 2), 0.0) for idx in range(n)]
    nlolib.propagate(
        nlolib.PulseSpec(samples=field0, delta_time=dt, frequency_grid=omega),
        lambda A, w: 1j * 0.02 * w * w,
        lambda A, I: 1j * A * (0.01 * I),
        propagation_distance=0.02,
        starting_step_size=0.01,
        max_step_size=0.01,
        min_step_size=0.01,
        error_tolerance=1e-9,
        num_records=1,
        exec_options=nlolib.ExecutionOptions(backend_type="cpu"),
    )
    snapshot = nlolib.perf_profile_read()
    nlolib.perf_profile_set_enabled(False)

    assert isinstance(snapshot, nlolib.PerfProfileSnapshot)
    assert snapshot.dispersion_ms >= 0.0
    assert snapshot.nonlinear_ms >= 0.0
    assert snapshot.dispersion_calls > 0
    assert snapshot.nonlinear_calls > 0
    assert math.isfinite(snapshot.dispersion_ms)
    assert math.isfinite(snapshot.nonlinear_ms)
    print("test_python_perf_profile: perf profile APIs returned finite counters after propagation.")


if __name__ == "__main__":
    main()
