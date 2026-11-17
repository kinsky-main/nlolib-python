import importlib.util
import pathlib
import re

import nlolib

ROOT = pathlib.Path(__file__).resolve().parents[1]


# The shared binding surface (docs/binding_api_vocabulary.md in the core
# repository): 17 functions, 9 data types and 2 errors, the same in every binding.
PUBLIC = {
    "load", "loaded_library_path", "propagate", "query_runtime_limits",
    "storage_is_available", "tensor_record_view", "set_log_file",
    "set_log_buffer", "clear_log_buffer", "read_log_buffer", "set_log_level",
    "set_progress_options", "set_progress_stream", "perf_profile_set_enabled",
    "perf_profile_is_enabled", "perf_profile_reset", "perf_profile_read",
    "PulseSpec", "ExecutionOptions", "StorageOptions", "PropagationResult",
    "StepHistory", "StorageResult", "ProgressInfo", "RuntimeLimits",
    "PerfProfileSnapshot", "NlolibError", "PropagationAbortedError",
}

# Removed with the move to the shared surface; none may come back as a
# package attribute either.
REMOVED = {
    "NLolib", "OperatorSpec", "RuntimeOperators", "PropagateResult",
    "PreparedSimConfig", "prepare_sim_config", "default_execution_options",
    "default_storage_options", "translate_callable", "NT_MAX",
    "STORAGE_RUN_ID_MAX", "TENSOR_LAYOUT_XYT_T_FAST", "VECTOR_BACKEND_CPU",
    "VECTOR_BACKEND_AUTO", "NLO_VECTOR_BACKEND_CPU", "FFT_BACKEND_AUTO",
    "NLOLIB_STATUS_OK", "NLOLIB_LOG_LEVEL_INFO", "PROGRESS_EVENT_ACCEPTED",
    "NloComplex", "NloExecutionOptions", "NloRuntimeLimits",
    "NloPerfProfileSnapshot", "NloStorageOptions", "NloStorageResult",
    "make_complex_array", "complex_array_to_list",
}


def test_package_surface() -> None:
    assert set(nlolib.__all__) == PUBLIC, sorted(set(nlolib.__all__) ^ PUBLIC)
    assert len(nlolib.__all__) == len(PUBLIC), "duplicate entries in __all__"
    for name in PUBLIC:
        assert hasattr(nlolib, name), name
    for name in REMOVED:
        assert not hasattr(nlolib, name), f"{name} is still a package attribute"
    assert issubclass(nlolib.NlolibError, RuntimeError)
    assert issubclass(nlolib.PropagationAbortedError, nlolib.NlolibError)
    assert importlib.util.find_spec("nlolib.translate") is None
    assert importlib.util.find_spec("nlolib._client") is None
    assert importlib.util.find_spec("nlolib._wrapper") is None


def test_version_attribute() -> None:
    """nlolib.__version__ must name the version this tree packages.

    pyproject.toml is the source of truth for the Python package version; this
    fails if the attribute stops tracking it.
    """
    assert hasattr(nlolib, "__version__")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'(?m)^version\s*=\s*"([0-9]+\.[0-9]+\.[0-9]+)"', pyproject)
    assert match, "could not parse the package version from pyproject.toml"
    assert nlolib.__version__ == match.group(1), (
        f"__version__ is {nlolib.__version__}, expected {match.group(1)}"
    )
    # Never exported: `from nlolib import *` must not rebind a caller's own.
    assert "__version__" not in nlolib.__all__


def main() -> None:
    test_package_surface()
    test_version_attribute()
    print("test_python_package_surface: package exports validated.")


if __name__ == "__main__":
    main()
