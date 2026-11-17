"""User-facing Python API for nlolib."""

import re
from importlib import metadata
from pathlib import Path

from ._api import (
    clear_log_buffer,
    load,
    loaded_library_path,
    perf_profile_is_enabled,
    perf_profile_read,
    perf_profile_reset,
    perf_profile_set_enabled,
    propagate,
    query_runtime_limits,
    read_log_buffer,
    set_log_buffer,
    set_log_file,
    set_log_level,
    set_progress_options,
    set_progress_stream,
    storage_is_available,
    tensor_record_view,
)
from ._models import (
    ExecutionOptions,
    NlolibError,
    PerfProfileSnapshot,
    ProgressInfo,
    PropagationAbortedError,
    PropagationResult,
    PulseSpec,
    RuntimeLimits,
    StepHistory,
    StorageOptions,
    StorageResult,
)


def _detect_version() -> str:
    """Report this package's version without storing a copy of it.

    ``pyproject.toml`` is the source of truth for the Python package version,
    which is released independently of the core library. Deriving the
    attribute from it rather than assigning a literal here means there is no
    second copy to drift.

    A source checkout has ``pyproject.toml`` next to the package directory; it
    is read first because an installed nlolib can shadow a checkout on
    ``sys.path``, and the installed distribution metadata would then name a
    version this code is not. The ``name`` check keeps an unrelated
    ``pyproject.toml`` from answering.
    """
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    try:
        text = pyproject.read_text(encoding="utf-8")
    except OSError:
        text = ""
    if 'name = "nlolib"' in text:
        match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', text)
        if match:
            return match.group(1)
    try:
        return metadata.version("nlolib")
    except metadata.PackageNotFoundError:
        return "unknown"


# Deliberately not in __all__: `from nlolib import *` should not rebind the
# caller's own __version__.
__version__ = _detect_version()

__all__ = [
    "load",
    "loaded_library_path",
    "propagate",
    "query_runtime_limits",
    "storage_is_available",
    "tensor_record_view",
    "set_log_file",
    "set_log_buffer",
    "clear_log_buffer",
    "read_log_buffer",
    "set_log_level",
    "set_progress_options",
    "set_progress_stream",
    "perf_profile_set_enabled",
    "perf_profile_is_enabled",
    "perf_profile_reset",
    "perf_profile_read",
    "PulseSpec",
    "ExecutionOptions",
    "StorageOptions",
    "PropagationResult",
    "StepHistory",
    "StorageResult",
    "ProgressInfo",
    "RuntimeLimits",
    "PerfProfileSnapshot",
    "NlolibError",
    "PropagationAbortedError",
]
