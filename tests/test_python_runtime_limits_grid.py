"""The grid-dependent runtime limits must be non-zero for a pulse.

`nlolib_query_runtime_limits` derives `estimated_required_working_set_bytes` and
`max_num_recorded_samples_in_memory` from a config that states its own sample
count. The config built for propagation leaves `time.nt` at zero, because the
solver is told the sample count as an argument to propagate and infers the
layout from it, so both fields came back zero -- which made the documented
`min(wanted, max_num_recorded_samples_in_memory)` recipe yield zero records,
and propagate then rejects zero. `query_runtime_limits(pulse=...)` stamps the
count onto the config it builds for the query.
"""

import math

import nlolib

N = 512
DT = 0.002


CPU = nlolib.ExecutionOptions(backend_type="cpu")


def _pulse(num_time_samples=N):
    two_pi = 2.0 * math.pi
    step = two_pi / (float(num_time_samples) * DT)
    grid = [
        complex(i * step if i <= num_time_samples // 2
                else -(num_time_samples - i) * step, 0.0)
        for i in range(num_time_samples)
    ]
    field = [
        complex(math.exp(-(((i - num_time_samples / 2) * DT) / 0.05) ** 2), 0.0)
        for i in range(num_time_samples)
    ]
    return nlolib.PulseSpec(samples=field, delta_time=DT, frequency_grid=grid)


def _limits(num_time_samples=N):
    return nlolib.query_runtime_limits(pulse=_pulse(num_time_samples), exec_options=CPU)


def test_grid_dependent_limits_are_populated() -> None:
    limits = _limits()

    working_set = int(limits.estimated_required_working_set_bytes)
    in_memory = int(limits.max_num_recorded_samples_in_memory)

    assert working_set > 0, (
        "estimated_required_working_set_bytes is 0; the config's sample count "
        "did not reach nlolib_query_runtime_limits"
    )
    assert in_memory > 0, (
        "max_num_recorded_samples_in_memory is 0, so the documented "
        "min(wanted, max_num_recorded_samples_in_memory) recipe yields 0"
    )
    # One working vector per sample is the floor; the solver holds several.
    assert working_set >= N * 16, f"working set {working_set} is below one vector"

    # It must say something: equal to the storage cap would mean the field
    # carries no information, which is what it used to do.
    with_storage = int(limits.max_num_recorded_samples_with_storage)
    assert in_memory < with_storage, (
        f"in-memory capacity {in_memory} equals the storage cap {with_storage}, "
        "so it is not derived from memory at all"
    )


def test_in_memory_capacity_shrinks_as_records_grow() -> None:
    """Capacity is host memory over record size, so a 4x grid holds ~4x fewer."""
    small = int(_limits(N).max_num_recorded_samples_in_memory)
    large = int(_limits(16 * N).max_num_recorded_samples_in_memory)
    assert 0 < large < small, f"capacity did not shrink with record size: {small} -> {large}"
    ratio = small / large
    assert 8.0 <= ratio <= 32.0, f"expected roughly 16x fewer records, got {ratio:.1f}x"


def test_limits_scale_with_the_grid() -> None:
    """A field pinned to the grid, not a constant that happens to be non-zero."""
    small = int(_limits(N).estimated_required_working_set_bytes)
    large = int(_limits(4 * N).estimated_required_working_set_bytes)
    assert large == 4 * small, f"working set did not scale with nt: {small} -> {large}"


def test_query_does_not_mutate_the_callers_pulse() -> None:
    pulse = _pulse()
    before = (list(pulse.samples), list(pulse.frequency_grid), pulse.pulse_period)
    nlolib.query_runtime_limits(pulse=pulse, exec_options=CPU)
    assert (list(pulse.samples), list(pulse.frequency_grid), pulse.pulse_period) == before


def test_propagate_still_works_after_a_limits_query() -> None:
    pulse = _pulse()
    limits = nlolib.query_runtime_limits(pulse=pulse, exec_options=CPU)
    records = min(4, int(limits.max_num_recorded_samples_in_memory))
    assert records > 0

    result = nlolib.propagate(
        pulse,
        lambda A, w: 1j * -0.01 * w * w,
        lambda A, I: 1j * 1.0 * A * I,
        propagation_distance=0.05,
        starting_step_size=1e-3,
        max_step_size=1e-3,
        min_step_size=1e-3,
        error_tolerance=1e-10,
        num_records=records,
        exec_options=CPU,
    )
    assert len(result.records) == records


def test_storage_ceiling_is_not_a_memory_figure() -> None:
    """The streaming ceiling must not move with the grid, or with host memory.

    max_num_recorded_samples_in_memory describes in-memory recording, where
    every record stays resident. A run streaming to the snapshot store keeps
    only a ring window resident, so its ceiling is
    max_num_recorded_samples_with_storage -- a precision ceiling. If that one
    ever started tracking memory, a streaming run would be told it can record
    far less than it can.
    """
    caps = [int(_limits(n).max_num_recorded_samples_with_storage) for n in (N, 16 * N)]
    assert caps[0] == caps[1], f"storage ceiling moved with the grid: {caps}"

    in_memory = int(_limits(N).max_num_recorded_samples_in_memory)
    assert caps[0] > in_memory, (
        "the storage ceiling is no higher than the in-memory capacity, so "
        "streaming would be constrained by host memory it does not use"
    )


def main() -> None:
    test_grid_dependent_limits_are_populated()
    test_in_memory_capacity_shrinks_as_records_grow()
    test_limits_scale_with_the_grid()
    test_storage_ceiling_is_not_a_memory_figure()
    test_query_does_not_mutate_the_callers_pulse()
    test_propagate_still_works_after_a_limits_query()
    print("test_python_runtime_limits_grid: grid-dependent runtime limits populated.")


if __name__ == "__main__":
    main()
