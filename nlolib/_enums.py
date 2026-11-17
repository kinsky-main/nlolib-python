"""
Private mapping between the public enum strings and the C API integers.

Every enum-style value on the public surface is a lowercase string; the C
integers stay in this module and in `_binding`.
"""

from __future__ import annotations

from typing import Mapping

_VECTOR_BACKEND = {"cpu": 0, "vulkan": 1, "auto": 2}
_FFT_BACKEND = {"auto": 0, "fftw": 1, "vkfft": 2}
_NONLINEAR_MODEL = {"expr": 0, "kerr_raman": 1}
_STORAGE_CAP_POLICY = {"stop_writes": 0, "fail": 1}
_LOG_LEVEL = {"error": 0, "warn": 1, "info": 2, "debug": 3}
_PROGRESS_STREAM = {"stderr": 0, "stdout": 1, "both": 2}
_PROGRESS_EVENT = {"accepted": 0, "rejected": 1, "finish": 2}
_STATUS = {
    "ok": 0,
    "invalid_argument": 1,
    "allocation_failed": 2,
    "not_implemented": 3,
    "aborted": 4,
}


def to_c(name: str, table: Mapping[str, int], value: object) -> int:
    """C integer for a public enum string; ValueError for anything else."""
    if isinstance(value, str) and value in table:
        return table[value]
    choices = ", ".join(repr(key) for key in table)
    raise ValueError(f"{name} must be one of {choices}, got {value!r}")


def from_c(table: Mapping[str, int], value: int) -> str:
    """Public string for a C integer the library handed back."""
    for key, number in table.items():
        if number == int(value):
            return key
    return str(int(value))


def vector_backend(value: object) -> int:
    return to_c("backend_type", _VECTOR_BACKEND, value)


def fft_backend(value: object) -> int:
    return to_c("fft_backend", _FFT_BACKEND, value)


def nonlinear_model(value: object) -> int:
    return to_c("nonlinear_model", _NONLINEAR_MODEL, value)


def cap_policy(value: object) -> int:
    return to_c("cap_policy", _STORAGE_CAP_POLICY, value)


def log_level(value: object) -> int:
    return to_c("level", _LOG_LEVEL, value)


def progress_stream(value: object) -> int:
    return to_c("stream_mode", _PROGRESS_STREAM, value)


def progress_event_name(value: int) -> str:
    return from_c(_PROGRESS_EVENT, value)


def status_name(value: int) -> str:
    return from_c(_STATUS, value)
