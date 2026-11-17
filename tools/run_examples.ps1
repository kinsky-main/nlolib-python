
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

# The shared example set (the same in the MATLAB and Julia bindings). The
# on-demand examples -- tensor_backend_scaling*, fixed_step_soliton_convergence,
# two_mode_linear_beating -- are not run here.
$pythonExamples = @(
    "quickstart.py",
    "linear_drift.py",
    "spm.py",
    "second_order_soliton.py",
    "conservation_checks.py",
    "raman_scattering.py",
    "tensor_dispersion_3d.py",
    "transverse_grin_beam.py",
    "spatiotemporal_kerr_bullet.py",
    "grin_soliton_fission.py",
    "high_order_grin_soliton.py",
    "sqlite_storage.py"
)

foreach ($example in $pythonExamples) {
    python "$repoRoot\examples\$example"
}