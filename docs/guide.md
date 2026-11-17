# Python Binding Guide

The Python package exposes the user-facing `nlolib` wrapper around the native
shared library.

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
import nlolib

beta2 = -0.02
alpha = 0.10
gamma = 1.20

runtime = nlolib.RuntimeOperators(
    dispersion_factor_fn=lambda A, w: 1j * (0.5 * beta2) * w * w - 0.5 * alpha,
    nonlinear_fn=lambda A, I: 1j * gamma * A * I,
)
```

The callables are lowered to the core's runtime operator language internally.
Operator expression strings are not part of the Python API, and passing one
raises `TypeError`.

## Public API Entry Points

- `nlolib.NLolib`
- `nlolib.RuntimeOperators`
- `nlolib.PulseSpec`
- `nlolib.OperatorSpec`
- `nlolib.propagate`
- `nlolib.query_runtime_limits`
- `nlolib.perf_profile_read`, `nlolib.perf_profile_reset`,
  `nlolib.perf_profile_set_enabled`, `nlolib.perf_profile_is_enabled`
- `nlolib.__version__` -- the package version, from the installed distribution
  metadata or, in a source checkout, from `pyproject.toml`

## Checking Whether the GPU JIT Ran

On the Vulkan backend each operator program is compiled to a single GPU kernel
at runtime, and falls back to the interpreter when it cannot be. The fallback is
silent: results stay correct, only slower. Two counters say which happened.

```python
api = nlolib.NLolib()
api.perf_profile_reset()
api.propagate(cfg, field0, 1,
              nlolib.default_execution_options(nlolib.NLO_VECTOR_BACKEND_AUTO))

snapshot = api.perf_profile_read()
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
python examples/second_order_soliton_rk4ip.py
```

```bash
python3 -m pip install -r examples/requirements.txt
export PYTHONPATH="$PWD"
export NLOLIB_LIBRARY="$PWD/libnlolib.so"
python3 examples/second_order_soliton_rk4ip.py
```

## Related Reference

- @ref python_binding
