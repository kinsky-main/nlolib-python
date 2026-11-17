"""
Spatiotemporal Gaussian packet in a bulk Kerr medium with anomalous dispersion.

    dA/dz = i (b2 wt^2 + beta_t (kx^2 + ky^2)) A + i gamma |A|^2 A

with b2 = beta2 / 2 (anomalous) and beta_t = -1/(2 k0) (diffraction). The waist
and duration are chosen so the Rayleigh range equals the dispersion length
(zR = k0 w0^2 / 2 = T0^2 / |beta2| = l0): the packet spreads equally in time
and space, the regime where a light bullet can form.

Two runs on the same grid: a linear one (gamma = 0), checked against the
Gaussian laws T0 sqrt(1 + (z/l0)^2) and w0 sqrt(1 + (z/l0)^2), and a Kerr one
in which self-focusing arrests part of the spread. Lengths in um, times in fs;
the envelope has unit peak amplitude.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from backend.app_base import ExampleAppBase
from backend.metrics import report_metric
from backend.plotting import plot_intensity_colormap_vs_propagation, plot_two_curve_comparison
import nlolib

# nlolib-example-parameters: begin
lambda0 = 1.55
n0 = 1.444
beta2 = -0.028
l0 = 100000.0
nt = 32
nx = 64
ny = 64
dt = 20.0
dx = 32.0
dy = 32.0
num_records = 48
# nlolib-example-parameters: end
# Derived: z_final = 1.5 l0, gamma_kerr = 0.6 / l0, beta_t = -1 / (2 k0),
# k0 = 2 pi n0 / lambda0, w0 = sqrt(2 l0 / k0), T0 = sqrt(l0 |beta2|).


def _axis(count: int, step: float) -> np.ndarray:
    return (np.arange(count) - 0.5 * (count - 1)) * step


def _run_case(gamma_kerr: float, field0: np.ndarray, omega: np.ndarray, z_final: float,
              beta_t: float) -> dict[str, np.ndarray]:
    b2 = 0.5 * beta2
    pulse = nlolib.PulseSpec(
        samples=field0.reshape(-1).tolist(),
        delta_time=dt,
        pulse_period=nt * dt,
        frequency_grid=[complex(w, 0.0) for w in omega],
        tensor_nt=nt,
        tensor_nx=nx,
        tensor_ny=ny,
        delta_x=dx,
        delta_y=dy,
    )
    result = nlolib.propagate(
        pulse,
        lambda A, wt, kx, ky: 1j * (b2 * wt * wt + beta_t * (kx * kx + ky * ky)),
        lambda A, I: 1j * gamma_kerr * A * I,
        propagation_distance=z_final,
        num_records=num_records,
        preset="accuracy",
        exec_options=nlolib.ExecutionOptions(backend_type="auto"),
    )
    z_axis = np.asarray(result.z_axis, dtype=np.float64)
    # Records are flat with t fastest, then y, then x: (x, y, t) in C order.
    intensity = np.abs(np.asarray(result.records, dtype=np.complex128).reshape(len(z_axis), nx, ny, nt)) ** 2
    t = _axis(nt, dt)
    x = _axis(nx, dx)
    energy = intensity.sum(axis=(1, 2, 3))
    sigma_t = np.sqrt((intensity * (t**2)[None, None, None, :]).sum(axis=(1, 2, 3)) / energy)
    sigma_x = np.sqrt((intensity * (x**2)[None, :, None, None]).sum(axis=(1, 2, 3)) / energy)
    return {
        "z": z_axis,
        "power_t": intensity.sum(axis=(1, 2)),
        "fluence_x": intensity.sum(axis=(2, 3)),
        "sigma_t": sigma_t,
        "sigma_x": sigma_x,
        "peak": intensity.max(axis=(1, 2, 3)),
        "energy_drift": abs(energy[-1] - energy[0]) / max(energy[0], 1e-30),
    }


def _run(args: argparse.Namespace) -> float:
    k0 = 2.0 * math.pi * n0 / lambda0
    w0 = math.sqrt(2.0 * l0 / k0)
    t0 = math.sqrt(l0 * abs(beta2))
    z_final = 1.5 * l0
    gamma_kerr = 0.6 / l0
    beta_t = -1.0 / (2.0 * k0)
    print(f"w0 = {w0:.1f} um, T0 = {t0:.1f} fs, zR = LD = {l0 * 1e-3:.1f} mm, z_end = {z_final * 1e-3:.1f} mm")

    t = _axis(nt, dt)
    x = _axis(nx, dx)
    y = _axis(ny, dy)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    spatial = np.exp(-((xx**2 + yy**2) / w0**2))
    temporal = np.exp(-(t**2) / (2.0 * t0**2))
    field0 = (spatial[:, :, None] * temporal[None, None, :]).astype(np.complex128)  # (x, y, t)
    omega = 2.0 * math.pi * np.fft.fftfreq(nt, d=dt)

    linear = _run_case(0.0, field0, omega, z_final, beta_t)
    kerr = _run_case(gamma_kerr, field0, omega, z_final, beta_t)
    z_axis = kerr["z"]

    # Second moments as 1/e amplitude widths, to read against the Gaussian laws.
    growth = np.sqrt(1.0 + (linear["z"] / l0) ** 2)
    width_error = float(np.max(np.abs(math.sqrt(2.0) * linear["sigma_t"] - t0 * growth)) / t0)
    radius_error = float(np.max(np.abs(2.0 * linear["sigma_x"] - w0 * growth)) / w0)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    z_mm = z_axis * 1e-3
    saved = [
        plot_intensity_colormap_vs_propagation(
            t, z_mm, kerr["power_t"], output_dir / "kerr_bullet_power_vs_time.png",
            x_label="t (fs)", y_label="z (mm)",
        ),
        plot_intensity_colormap_vs_propagation(
            x, z_mm, kerr["fluence_x"], output_dir / "kerr_bullet_fluence_vs_x.png",
            x_label=r"x ($\mu$m)", y_label="z (mm)",
        ),
        plot_two_curve_comparison(
            z_mm, math.sqrt(2.0) * kerr["sigma_t"], math.sqrt(2.0) * linear["sigma_t"],
            output_dir / "kerr_bullet_pulse_width.png",
            label_a="Kerr", label_b="linear", x_label="z (mm)", y_label="Pulse width T (fs)",
        ),
        plot_two_curve_comparison(
            z_mm, 2.0 * kerr["sigma_x"], 2.0 * linear["sigma_x"],
            output_dir / "kerr_bullet_beam_radius.png",
            label_a="Kerr", label_b="linear", x_label="z (mm)", y_label=r"Beam radius w ($\mu$m)",
        ),
    ]

    print(f"final peak intensity: linear {linear['peak'][-1]:.3f}, Kerr {kerr['peak'][-1]:.3f} (input 1)")
    print(f"energy drift: linear {linear['energy_drift']:.3e}, Kerr {kerr['energy_drift']:.3e}")
    report_metric("linear-run width error vs analytic Gaussian", width_error)
    report_metric("linear-run radius error vs analytic Gaussian", radius_error)
    report_metric("energy drift", kerr["energy_drift"])
    for path in saved:
        if path is not None:
            print(f"saved plot: {path}")
    return max(width_error, radius_error)


class SpatiotemporalKerrBulletApp(ExampleAppBase):
    example_slug = "spatiotemporal_kerr_bullet"
    description = "Spatiotemporal Gaussian packet in bulk Kerr media, linear vs Kerr."

    def run(self) -> float:
        return _run(self.args)


def main() -> float:
    return SpatiotemporalKerrBulletApp.from_cli().run()


if __name__ == "__main__":
    main()
