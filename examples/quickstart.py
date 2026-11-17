"""
Quick start: a chirped Gaussian under pure dispersion, two records.

The shortest complete propagate call: a pulse, a linear and a nonlinear
operator written as plain functions, and the "balanced" preset. Dispersion
alone conserves power, so the initial and final powers agree.
"""

from __future__ import annotations

import argparse

import numpy as np
from backend.app_base import ExampleAppBase
from backend.metrics import report_metric
from backend.runner import centered_time_grid
import nlolib

# nlolib-example-parameters: begin
n = 1024
dt = 0.01
sigma = 0.2
chirp = 12.0
beta2 = 0.05
gamma = 0.0
z_final = 1.0
num_records = 2
# nlolib-example-parameters: end


def _run(args: argparse.Namespace) -> float:
    t = centered_time_grid(n, dt)
    field0 = np.exp(-((t / sigma) ** 2)) * np.exp((-1.0j) * chirp * t)
    pulse = nlolib.PulseSpec(samples=field0.astype(np.complex128).tolist(), delta_time=dt)

    result = nlolib.propagate(
        pulse,
        lambda A, w: 1j * (0.5 * beta2) * w * w,
        lambda A, I: 1j * gamma * A * I,
        propagation_distance=z_final,
        num_records=num_records,
        preset="balanced",
        exec_options=nlolib.ExecutionOptions(backend_type="auto"),
    )
    records = np.asarray(result.records, dtype=np.complex128)
    initial_power = float(np.sum(np.abs(records[0]) ** 2))
    final_power = float(np.sum(np.abs(records[-1]) ** 2))

    print(f"quickstart completed: z records {result.z_axis}")
    report_metric("initial power", initial_power)
    report_metric("final power", final_power)
    return abs(final_power - initial_power) / initial_power


class QuickstartApp(ExampleAppBase):
    example_slug = "quickstart"
    description = "Chirped Gaussian under pure dispersion: the shortest propagate call."

    def run(self) -> float:
        return _run(self.args)


def main() -> float:
    return QuickstartApp.from_cli().run()


if __name__ == "__main__":
    main()
