"""
Higher-order soliton fission in a graded-index (GRIN) fibre, (t, x, y).

A 4th-order soliton (by launch power) enters the fundamental mode of a GRIN
multimode fibre. The field A(t, x, y, z), in sqrt(W/um^2), obeys

    dA/dz = i (beta2/2 w^2 + beta3/6 w^3 - (kx^2 + ky^2)/(2 k0)) A
          + i (gamma |A|^2 + V(x, y)) A

with the parabolic GRIN potential V = -(k0 g^2 / 2)(x^2 + y^2). The fibre
holds the beam in its fundamental mode while, in time, third-order dispersion
splits the input into fundamental solitons and sheds a dispersive wave on the
blue side. Units: um, fs, W.

The operator's wt is the FFT frequency (+2 pi k / T); with the envelope
convention exp(-i w t) the physical offset is w = -wt, so the odd (beta3)
term changes sign in the operator.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from backend.app_base import ExampleAppBase
from backend.metrics import report_metric
from backend.plotting import plot_intensity_colormap_vs_propagation
import nlolib

# nlolib-example-parameters: begin
lambda0 = 1.55
n0 = 1.45
beta2 = -0.028
beta3 = 0.3
n2 = 2.6e-08
soliton_order = 4
t0 = 30.0
nt = 1024
dt = 2.5
nx = 24
ny = 24
dx = 2.0
t_launch = -400.0
num_records = 48
error_tolerance = 1e-06
# nlolib-example-parameters: end
# Derived: g_grin = sqrt(2 * 0.01) / 25 (1% index step, 25 um core),
# z_final = 1.2 t0^2 / |beta2|, step sizes z_final / 500, / 50 and / 1e5.


def _run(args: argparse.Namespace) -> float:
    k0 = 2.0 * math.pi * n0 / lambda0
    g_grin = math.sqrt(2.0 * 0.01) / 25.0
    gamma = 2.0 * math.pi * n2 / lambda0
    w0 = math.sqrt(2.0 / (k0 * g_grin))  # fundamental GRIN mode radius
    ld = t0 * t0 / abs(beta2)
    gamma_eff = gamma / (math.pi * w0 * w0)
    p0 = soliton_order**2 / (gamma_eff * ld)
    z_final = 1.2 * ld
    print(f"mode radius {w0:.1f} um, LD = {ld / 1e3:.1f} mm, P0 = {p0 / 1e6:.2f} MW, z = {z_final / 1e3:.1f} mm")

    t = (np.arange(nt) - nt / 2) * dt
    x = (np.arange(nx) - nx / 2) * dx
    y = (np.arange(ny) - ny / 2) * dx
    xx, yy = np.meshgrid(x, y, indexing="ij")  # (x, y)
    mode = np.exp(-(xx**2 + yy**2) / w0**2)
    peak_amp = math.sqrt(2.0 * p0 / (math.pi * w0 * w0))
    temporal = peak_amp / np.cosh((t - t_launch) / t0)
    field0 = (mode[:, :, None] * temporal[None, None, :]).astype(np.complex128)  # (x, y, t)
    potential = (-0.5 * k0 * g_grin**2 * (xx**2 + yy**2)).astype(np.complex128)

    b2 = beta2 / 2.0
    b3 = beta3 / 6.0
    bt = -1.0 / (2.0 * k0)
    pulse = nlolib.PulseSpec(
        samples=field0.reshape(-1).tolist(),
        delta_time=dt,
        pulse_period=nt * dt,
        tensor_nt=nt,
        tensor_nx=nx,
        tensor_ny=ny,
        delta_x=dx,
        delta_y=dx,
        potential_grid=potential.reshape(-1).tolist(),
    )
    result = nlolib.propagate(
        pulse,
        lambda A, wt, kx, ky: 1j * (b2 * wt * wt - b3 * wt * wt * wt + bt * (kx * kx + ky * ky)),
        lambda A, I, V: 1j * A * (gamma * I + V),
        propagation_distance=z_final,
        num_records=num_records,
        preset="balanced",
        error_tolerance=error_tolerance,
        starting_step_size=z_final / 500.0,
        max_step_size=z_final / 50.0,
        min_step_size=z_final / 1e5,
        capture_step_history=True,
        exec_options=nlolib.ExecutionOptions(backend_type="auto"),
    )
    steps = result.step_history
    print(
        f"{len(steps.step_size)} accepted steps, step {min(steps.step_size):.2f} to "
        f"{max(steps.step_size):.2f} um, max local error {max(steps.error):.1e}"
    )

    z_mm = np.asarray(result.z_axis, dtype=np.float64) / 1e3
    fields = np.asarray(result.records, dtype=np.complex128).reshape(len(z_mm), nx, ny, nt)
    power_t = (np.abs(fields) ** 2).sum(axis=(1, 2)) * dx * dx  # W
    spectrum = (np.abs(np.fft.fft(fields, axis=3)) ** 2).sum(axis=(1, 2))
    fluence_x = (np.abs(fields[:, :, ny // 2, :]) ** 2).sum(axis=2)
    w = -2.0 * math.pi * np.fft.fftfreq(nt, d=dt)
    order = np.argsort(w)

    energy_drift = abs(power_t[-1].sum() / power_t[0].sum() - 1.0)
    peak_in = float(power_t[0].max()) / 1e6
    peak_out = float(power_t[-1].max()) / 1e6

    def db(values: np.ndarray) -> np.ndarray:
        return np.maximum(10.0 * np.log10(values / values.max() + 1e-12), -40.0) + 40.0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = [
        plot_intensity_colormap_vs_propagation(
            t, z_mm, db(power_t), output_dir / "grin_soliton_fission_power_db.png",
            x_label="t (fs)", y_label="z (mm)", colorbar_label="Power (dB + 40)",
        ),
        plot_intensity_colormap_vs_propagation(
            w[order], z_mm, db(spectrum[:, order]), output_dir / "grin_soliton_fission_spectrum_db.png",
            x_label=r"$\omega - \omega_0$ (rad/fs)", y_label="z (mm)", colorbar_label="Spectrum (dB + 40)",
        ),
        plot_intensity_colormap_vs_propagation(
            x, z_mm, fluence_x, output_dir / "grin_soliton_fission_fluence_x.png",
            x_label=r"x ($\mu$m)", y_label="z (mm)",
        ),
    ]

    report_metric("energy drift", energy_drift)
    report_metric("peak power in", peak_in)
    report_metric("peak power out", peak_out)
    report_metric("accepted steps", len(steps.step_size))
    print("(peak powers in MW)")
    for path in saved:
        if path is not None:
            print(f"saved plot: {path}")
    return energy_drift


class GrinSolitonFissionApp(ExampleAppBase):
    example_slug = "grin_soliton_fission"
    description = "Fourth-order soliton fission in a GRIN multimode fibre."

    def run(self) -> float:
        return _run(self.args)


def main() -> float:
    return GrinSolitonFissionApp.from_cli().run()


if __name__ == "__main__":
    main()
