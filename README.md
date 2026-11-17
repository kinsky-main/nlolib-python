# nlolib

Python package for the nlolib nonlinear-optics runtime: a split-step (RK4IP)
propagator for pulses and beams in nonlinear media, with a CPU backend and an
optional Vulkan GPU backend.

```bash
pip install nlolib
```

Wheels are published for Linux x86-64 and Windows x86-64 and work on any
CPython 3.9 or later. The native library ships inside the wheel, so nothing is
compiled on install.

Full documentation, including the Python guide and API reference:
https://kinsky-main.github.io/nlo-rk4ip-code/

## Quick start

Operators are plain Python callables. This propagates a fundamental soliton of
the nonlinear Schrödinger equation, which keeps its shape:

```python
import math
import nlolib

beta2 = -1.0  # group-velocity dispersion (anomalous)
gamma = 1.0   # Kerr coefficient

n, dt = 512, 0.05
t = [(i - n // 2) * dt for i in range(n)]
pulse = nlolib.PulseSpec(samples=[complex(1.0 / math.cosh(ti), 0.0) for ti in t], delta_time=dt)

result = nlolib.propagate(
    pulse,
    lambda A, w: 1j * 0.5 * beta2 * w * w,  # linear operator D(w)
    lambda A, I: 1j * gamma * A * I,        # nonlinear operator N(A), I = |A|^2
    propagation_distance=2.0,
    num_records=3,
)
print(result.z_axis)                      # [0.0, 1.0, 2.0]
print(max(abs(a) for a in result.final))  # ~1.0
```

`result.records[k]` is the field at `result.z_axis[k]`. Each operator is a
`lambda` or `def` whose body is a single expression. Its parameters name the
symbols it uses: `A` field, `w` angular frequency, `I` intensity `|A|^2`, `V`
potential, and `kx`/`ky` for tensor runs. Free names are captured from the
enclosing scope; `constant_bindings={"beta2": -1.0}` names values explicitly.
Expression strings are not accepted; passing one raises `TypeError`.

Options are keyword arguments: solver settings (`preset`, `error_tolerance`,
step sizes), records (`num_records`, or `record_z` for exact positions), the
backend (`exec_options=nlolib.ExecutionOptions(backend_type="cpu")`) and SQLite
storage (`storage=nlolib.StorageOptions(sqlite_path="run.sqlite")`). Every
enum-style value is a lowercase string.

## CPU and GPU

- The CPU backend works out of the box.
- The Vulkan backend is used when a Vulkan driver is present; without one the
  CPU backend runs unchanged.
- On Vulkan, operator programs are compiled to single GPU kernels at runtime
  and fall back to an interpreter silently when they cannot be. After a run,
  `nlolib.perf_profile_read()` reports which happened:

  ```python
  snapshot = nlolib.perf_profile_read()
  snapshot.operator_jit_programs   # kernels compiled; 0 on CPU
  snapshot.operator_jit_fallbacks  # programs left on the interpreter
  ```

## Examples

The [`examples/`](https://github.com/kinsky-main/nlolib-python/tree/main/examples)
directory of the source repository holds the example set shared with the
MATLAB and Julia bindings -- `quickstart`, `linear_drift`, `spm`,
`second_order_soliton`, `conservation_checks`, `raman_scattering`,
`tensor_dispersion_3d`, `transverse_grin_beam`, `spatiotemporal_kerr_bullet`,
`grin_soliton_fission`, `high_order_grin_soliton` and `sqlite_storage` -- each
with its numbers in one parameter block at the top. `tensor_backend_scaling*`,
`fixed_step_soliton_convergence` and `two_mode_linear_beating` are on-demand
extras. They need the extra packages in `examples/requirements.txt`.

## Links

- Source and issues: https://github.com/kinsky-main/nlolib-python
- Core C library: https://github.com/kinsky-main/nlo-rk4ip-code
- Working on the package itself: [docs/development.md](https://github.com/kinsky-main/nlolib-python/blob/main/docs/development.md)

## License

GPL-2.0-or-later.
