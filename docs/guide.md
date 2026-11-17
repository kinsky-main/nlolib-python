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
