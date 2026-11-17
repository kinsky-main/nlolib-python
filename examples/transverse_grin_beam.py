"""
Off-axis Gaussian beam in a parabolic-index (GRIN) medium, x-y only.

A purely transverse run (tensor_nt = 1): the field is an x-y sheet propagated
along z by

    dA/dz = i beta_t (kx^2 + ky^2) A + i V(x, y) A

with beta_t = -1/(2 k0) (diffraction) and V = -(k0 g^2 / 2)(x^2 + y^2), the
GRIN lens, supplied as a potential grid. A width-matched Gaussian launched at
rest off-axis oscillates rigidly about the axis with pitch 2 pi / g, so the
numerical centroid is checked against x_offset cos(g z). Lengths in um.
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
lambda0 = 1.03
n0 = 1.45
g_grin = 0.002
nx = 128
ny = 128
dx = 1.5
dy = 1.5
x_offset = 20.0
num_records = 96
# nlolib-example-parameters: end
# Derived: k0 = 2 pi n0 / lambda0, z_final = 2 pi / g_grin,
# w_matched = sqrt(2 / (k0 g_grin)), beta_t = -1 / (2 k0).


def _run(args: argparse.Namespace) -> float:
    k0 = 2.0 * math.pi * n0 / lambda0
    z_final = 2.0 * math.pi / g_grin
    w_matched = math.sqrt(2.0 / (k0 * g_grin))
    beta_t = -1.0 / (2.0 * k0)
    grin_depth = -0.5 * k0 * g_grin * g_grin
    print(f"matched waist = {w_matched:.2f} um, pitch period = {z_final:.1f} um")

    x = (np.arange(nx) - 0.5 * (nx - 1)) * dx
    y = (np.arange(ny) - 0.5 * (ny - 1)) * dy
    xx, yy = np.meshgrid(x, y, indexing="ij")  # (nx, ny): x slowest, y fastest
    field0 = np.exp(-(((xx - x_offset) ** 2 + yy**2) / w_matched**2)).astype(np.complex128)
    potential = (grin_depth * (xx**2 + yy**2)).astype(np.complex128)

    # The tensor layout is t fastest, then y, then x: with tensor_nt = 1 an
    # (nx, ny) array flattens in C order.
    pulse = nlolib.PulseSpec(
        samples=field0.reshape(-1).tolist(),
        delta_time=1.0,
        pulse_period=1.0,
        frequency_grid=[0j],
        tensor_nt=1,
        tensor_nx=nx,
        tensor_ny=ny,
        delta_x=dx,
        delta_y=dy,
        potential_grid=potential.reshape(-1).tolist(),
    )
    result = nlolib.propagate(
        pulse,
        lambda A, wt, kx, ky: 1j * beta_t * (kx * kx + ky * ky),
        lambda A, I, V: 1j * A * V,
        propagation_distance=z_final,
        num_records=num_records,
        preset="accuracy",
        exec_options=nlolib.ExecutionOptions(backend_type="auto"),
    )
    z_axis = np.asarray(result.z_axis, dtype=np.float64)
    records = np.asarray(result.records, dtype=np.complex128).reshape(len(z_axis), nx, ny)
    intensity = np.abs(records) ** 2

    power = intensity.sum(axis=(1, 2))
    power_drift = abs(power[-1] - power[0]) / max(power[0], 1e-12)
    centroid_x = (intensity * xx[None, :, :]).sum(axis=(1, 2)) / np.maximum(power, 1e-30)
    centroid_analytic = x_offset * np.cos(g_grin * z_axis)
    centroid_error = float(np.max(np.abs(centroid_x - centroid_analytic)))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = [
        plot_intensity_colormap_vs_propagation(
            x,
            z_axis,
            intensity[:, :, ny // 2],
            output_dir / "transverse_grin_beam_xz_intensity.png",
            x_label=r"x ($\mu$m)",
            y_label=r"Propagation distance z ($\mu$m)",
        ),
        plot_two_curve_comparison(
            z_axis,
            centroid_x,
            centroid_analytic,
            output_dir / "transverse_grin_beam_centroid.png",
            label_a="nlolib centroid",
            label_b=r"$x_0 \cos(g z)$",
            x_label=r"Propagation distance z ($\mu$m)",
            y_label=r"Beam centroid x ($\mu$m)",
        ),
    ]

    print(f"propagated {z_final:.1f} um on a {nx}x{ny} grid ({len(z_axis)} records)")
    report_metric("power drift", power_drift)
    report_metric("max centroid deviation from x_offset cos(g_grin z)", centroid_error)
    for path in saved:
        if path is not None:
            print(f"saved plot: {path}")
    return centroid_error


class TransverseGrinBeamApp(ExampleAppBase):
    example_slug = "transverse_grin_beam"
    description = "Off-axis Gaussian beam oscillating in a GRIN medium (x-y only)."

    def run(self) -> float:
        return _run(self.args)


def main() -> float:
    return TransverseGrinBeamApp.from_cli().run()


if __name__ == "__main__":
    main()
