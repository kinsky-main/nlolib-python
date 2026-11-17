"""
Runtime operator callable example using the unified example runner backend.
"""

from __future__ import annotations

import argparse

import numpy as np
from backend.app_base import ExampleAppBase
from backend.runner import centered_time_grid
from backend.storage import ExampleRunDB
import nlolib


def _run(args: argparse.Namespace) -> None:
    db = ExampleRunDB(args.db_path)
    example_name = "runtime_callable_operator_rk4ip"
    case_key = "default"

    if args.replot:
        run_group = db.resolve_replot_group(example_name, args.run_group)
        loaded = db.load_case(example_name=example_name, run_group=run_group, case_key=case_key)
        z_records = np.asarray(loaded.z_axis, dtype=np.float64)
        records = np.asarray(loaded.records, dtype=np.complex128)
    else:
        run_group = db.begin_group(example_name, args.run_group)
        n = 1024
        dt = 0.01
        beta2 = 0.05
        scale = beta2 / 2.0
        z_final = 1.0
        t = centered_time_grid(n, dt)
        field0 = np.exp(-((t / 0.20) ** 2)) * np.exp((-1.0j) * 12.0 * t)
        pulse = nlolib.PulseSpec(
            samples=field0.astype(np.complex128).tolist(),
            delta_time=dt,
            pulse_period=n * dt,
        )
        storage_kwargs = db.storage_kwargs(
            example_name=example_name,
            run_group=run_group,
            case_key=case_key,
            chunk_records=2,
        )
        result = nlolib.propagate(
            pulse,
            lambda A, w: (1.0j * scale) * (w * w),
            lambda A, I: (1.0j * 0.0) * I,
            propagation_distance=z_final,
            num_records=2,
            error_tolerance=1e-7,
            exec_options=nlolib.ExecutionOptions(backend_type="auto", fft_backend="vkfft"),
            **storage_kwargs,
        )
        z_records = np.asarray(result.z_axis, dtype=np.float64)
        records = np.asarray(result.records, dtype=np.complex128)
        db.save_case(
            example_name=example_name,
            run_group=run_group,
            case_key=case_key,
            run_id=result.storage_result.run_id,
            meta={},
        )

    print(f"runtime callable example completed (run_group={run_group}).")
    print(f"z records: {z_records}")
    print(f"initial power={np.sum(np.abs(records[0]) ** 2):.6e}")
    print(f"final power={np.sum(np.abs(records[-1]) ** 2):.6e}")


class RuntimeCallableOperatorApp(ExampleAppBase):
    example_slug = "runtime_callable_operator"
    description = "Runtime callable operator example with DB-backed run/replot."

    def run(self) -> None:
        _run(self.args)


def main() -> None:
    RuntimeCallableOperatorApp.from_cli().run()


if __name__ == "__main__":
    main()
