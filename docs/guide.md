# Python Binding Guide

The Python package exposes the user-facing `nlolib` wrapper around the native
shared library. Its surface is the one shared by the Python, MATLAB and Julia
bindings, described in the core repository's
[binding API vocabulary](../../docs/binding_api_vocabulary.md).

## Minimal Math-to-API Example

This example uses the quadratic GLSE mapping

\f[
D(\omega)=i\left(\frac{\beta_2}{2}\right)\omega^2-\frac{\alpha}{2},
\qquad
N(A)=i\gamma A|A|^2
\f]

Each operator is written as a Python callable. Its parameter names select the
runtime symbols (`A`, `w`, `I`, ...), and the scalars it uses are captured from
the enclosing scope:

```python
import math
import nlolib

beta2 = -0.02
alpha = 0.10
gamma = 1.20

n, dt = 1024, 0.01
field0 = [complex(math.exp(-(((i - n / 2) * dt) / 0.2) ** 2), 0.0) for i in range(n)]

result = nlolib.propagate(
    nlolib.PulseSpec(samples=field0, delta_time=dt),
    lambda A, w: 1j * (0.5 * beta2) * w * w - 0.5 * alpha,
    lambda A, I: 1j * gamma * A * I,
    propagation_distance=1.0,
    num_records=128,
    exec_options=nlolib.ExecutionOptions(backend_type="auto"),
)
```

`result.records[k]` is the field at `result.z_axis[k]`. With `record_z` the
positions are exactly the ones given, for any length including one; otherwise
they are `num_records` uniform positions from 0 to `propagation_distance`.
`result.final` is the last returned record, `result.records[-1]`, which is the
field at `propagation_distance` only when the last position is that distance.

The callables are lowered once, at setup, to the core's runtime operator
language. A temporal run lowers `linear_operator` to the core's dispersion
factor and a tensor run (`tensor_nt`, `tensor_nx`, `tensor_ny` set on the
pulse) to its linear factor; `w` and `wt` name the same axis in both. Operator
expression strings are not part of the Python API, and passing one raises
`TypeError`.

## Public API

Functions: `load`, `loaded_library_path`, `propagate`, `query_runtime_limits`,
`storage_is_available`, `tensor_record_view`, `set_log_file`, `set_log_buffer`,
`clear_log_buffer`, `read_log_buffer`, `set_log_level`, `set_progress_options`,
`set_progress_stream`, `perf_profile_set_enabled`, `perf_profile_is_enabled`,
`perf_profile_reset`, `perf_profile_read`.

Types: `PulseSpec`, `ExecutionOptions`, `StorageOptions`, `PropagationResult`,
`StepHistory`, `StorageResult`, `ProgressInfo`, `RuntimeLimits`,
`PerfProfileSnapshot`; errors `NlolibError` (a non-OK status, or a feature the
loaded build lacks; `.status` is the status string) and
`PropagationAbortedError` (the progress callback stopped the run; `.result`).

`nlolib.__version__` is the package version, from the installed distribution
metadata or, in a source checkout, from `pyproject.toml`.

## Upgrading from 1.x

2.0.0 moves the package to the API shared with the MATLAB and Julia bindings.
It is not backwards compatible.

- Operators are Python functions only. A string expression raises
  `TypeError`. Free names in the body are captured as constants;
  `constant_bindings` names values explicitly.
- There is one package-level `nlolib.propagate(pulse, linear_operator,
  nonlinear_operator, ...)` and no client class. `nlolib.load(path)` selects a
  specific shared library.
- Enum-style values are lowercase strings (`backend_type="vulkan"`,
  `cap_policy="fail"`, `level="debug"`, `meta["status"] == "ok"`). The integer
  constants are gone.
- `num_records` sets how many evenly spaced records come back, and `record_z`
  gives explicit positions. A `record_z` with one position returns the field at
  that position. `result.final` is the last returned record, `records[-1]`.
- The in-memory record limit is no longer a fixed 2 GiB check in the binding.
  The core derives it from the memory available on the host;
  `query_runtime_limits(pulse=pulse).max_num_recorded_samples_in_memory` reports it,
  and a run above it raises `NlolibError` with status `"invalid_argument"`.
  Give `storage=StorageOptions(...)` to record more.

| 1.x | 2.0.0 |
|---|---|
| `NLolib()` and its methods, `NLolib.set_log_file` | package functions: `nlolib.propagate`, `nlolib.set_log_file`, ... |
| `records=`, `num_recorded_samples=` | `num_records` |
| `t_eval`, `explicit_record_z` | `record_z` |
| `output`, `output_mode` | removed; `num_records` / `record_z` decide |
| `linear_factor_fn`, `nonlinear_fn`, `potential_fn`, `linear_fn` | `linear_operator`, `nonlinear_operator`, `potential_operator`, `linear_step_operator` |
| `dispersion_factor_fn`, `dispersion_fn` | removed; the run type picks the C context |
| `auto_capture_constants`, raw `constants`, `OperatorSpec` | removed; pass the function, use `constant_bindings` |
| `tensor_layout` | removed; the C API has one layout |
| `t_span`, `first_step`, `max_step`, `min_step`, `rtol`, `dense_output`, `events` | removed |
| `sqlite_path`, `run_id`, `sqlite_max_bytes`, `chunk_records`, `cap_policy`, `log_final_output_field_to_db` keywords | `storage=StorageOptions(...)` |
| `PulseSpec` fields repeated as `propagate` keywords | set them on the pulse |
| a dict, mapping or NamedTuple as `pulse` | `PulseSpec` |
| `prepare_sim_config`, `PreparedSimConfig`, `RuntimeOperators`, the low-level `propagate` forms | private; `propagate` takes `potential_operator` and `linear_step_operator`, `PulseSpec` takes `wt_axis`, `kx_axis`, `ky_axis` |
| `PropagateResult` (`status`, `message`, `t_events`, `y_events`, `sol`) | `PropagationResult`; status in `meta["status"]` |
| `meta["step_history"]`, `meta["storage_result"]` | `result.step_history`, `result.storage_result` |
| `meta` keys `output`, `records`, `message`, `storage_enabled`, `records_returned` | removed |
| `query_runtime_limits(config, ...)` | `query_runtime_limits(pulse=..., exec_options=...)` |
| `NloExecutionOptions`, `default_execution_options` | `ExecutionOptions` |
| `NloStorageOptions`, `default_storage_options` | `StorageOptions` |
| `NloRuntimeLimits`, `NloPerfProfileSnapshot`, `NloStorageResult` | `RuntimeLimits`, `PerfProfileSnapshot`, `StorageResult` |
| `NloComplex`, `NloTensor3dDesc`, `NloVkBackendConfig`, `make_complex_array`, `complex_array_to_list` | private |
| `NLO_VECTOR_BACKEND_*`, `NLO_FFT_BACKEND_*` and the other integer constants | lowercase strings |
| `NT_MAX`, `STORAGE_RUN_ID_MAX` | `query_runtime_limits`; a `run_id` over 63 characters is a `ValueError` |
| `RuntimeError` on a status failure | `NlolibError` (a `RuntimeError` subclass, with `.status`) |
| (new) | `loaded_library_path`, `tensor_record_view`, `StepHistory` |

## Checking Whether the GPU JIT Ran

On the Vulkan backend each operator program is compiled to a single GPU kernel
at runtime, and falls back to the interpreter when it cannot be. The fallback is
silent: results stay correct, only slower. Two counters say which happened.

```python
nlolib.perf_profile_reset()
nlolib.propagate(pulse, linear, nonlinear, propagation_distance=1.0,
                 exec_options=nlolib.ExecutionOptions(backend_type="auto"))

snapshot = nlolib.perf_profile_read()
print(snapshot.operator_jit_programs)   # 3: one kernel per active operator program
print(snapshot.operator_jit_fallbacks)  # 0: nothing was left on the interpreter
```

Unlike the timing counters in the same snapshot, these two need no
`perf_profile_set_enabled(True)`, so a release build can be checked as it
stands. Both stay zero wherever the JIT is never reached -- the CPU backend, or
a build configured `ENABLE_VULKAN_BACKEND=OFF` -- which is not the same as a
program falling back. See [GPU backend behaviour](../../docs/gpu_backend.md).

## Build-Tree Usage

After a core build with this binding checked out (see
[development.md](development.md)), from the binding directory:

```powershell
python -m pip install -r examples\requirements.txt
$env:PYTHONPATH="$PWD"
$env:NLOLIB_LIBRARY="$PWD/Release/nlolib.dll"
python examples/second_order_soliton.py
```

```bash
python3 -m pip install -r examples/requirements.txt
export PYTHONPATH="$PWD"
export NLOLIB_LIBRARY="$PWD/libnlolib.so"
python3 examples/second_order_soliton.py
```

## Related Reference

- @ref python_binding
