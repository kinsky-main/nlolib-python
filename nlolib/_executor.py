"""
Private execution of a normalized propagation request.
"""

from __future__ import annotations

import ctypes

from . import _enums
from ._binding import (
    PROPAGATE_OUTPUT_DENSE,
    PROPAGATE_OUTPUT_FINAL_ONLY,
    NLOLIB_STATUS_ABORTED,
    NLOLIB_STATUS_OK,
    NloComplex,
    NloExecutionOptions,
    NloProgressCallback,
    NloPropagateOptions,
    NloPropagateOutput,
    NloStepEvent,
    NloStorageOptions,
    NloStorageResult,
    complex_array_to_list,
    make_complex_array,
)
from ._config import _progress_info_from_struct, _step_history, _storage_result
from ._models import (
    ExecutionOptions,
    NlolibError,
    PropagationAbortedError,
    PropagationResult,
    StorageOptions,
)
from ._requests import _NormalizedPropagateRequest


def _c_execution_options(options: ExecutionOptions | None) -> NloExecutionOptions | None:
    """The C struct for `options`; the Vulkan handles stay null (library-owned)."""
    if options is None:
        return None
    out = NloExecutionOptions()
    out.backend_type = _enums.vector_backend(options.backend_type)
    out.fft_backend = _enums.fft_backend(options.fft_backend)
    out.device_heap_fraction = float(options.device_heap_fraction)
    out.record_ring_target = int(options.record_ring_target)
    out.forced_device_budget_bytes = int(options.forced_device_budget_bytes)
    return out


def _c_storage_options(storage: StorageOptions, keepalive: list[object]) -> NloStorageOptions:
    out = NloStorageOptions()
    path = str(storage.sqlite_path).encode("utf-8")
    keepalive.append(path)
    out.sqlite_path = ctypes.c_char_p(path)
    if storage.run_id:
        run_id = str(storage.run_id).encode("utf-8")
        keepalive.append(run_id)
        out.run_id = ctypes.c_char_p(run_id)
    else:
        out.run_id = None
    out.sqlite_max_bytes = int(storage.sqlite_max_bytes)
    out.chunk_records = int(storage.chunk_records)
    out.cap_policy = _enums.cap_policy(storage.cap_policy)
    out.log_final_output_field_to_db = int(bool(storage.log_final_output_field_to_db))
    return out


_COMPLEX_BYTES = ctypes.sizeof(NloComplex)


class PropagationExecutor:
    def __init__(self, lib: ctypes.CDLL, storage_is_available) -> None:
        self.lib = lib
        self._storage_is_available = storage_is_available

    def execute(self, request: _NormalizedPropagateRequest) -> PropagationResult:
        if request.storage is not None and not self._storage_is_available():
            raise NlolibError("not_implemented", "SQLite storage is not available in this nlolib build")

        sim_ptr = ctypes.pointer(request.sim_cfg)
        phys_ptr = ctypes.pointer(request.phys_cfg)
        n = len(request.input_seq)
        # The in-memory record limit is the core's: the output buffer below is
        # allocated zero-filled and lazily, and nlolib_propagate refuses a count
        # above max_num_recorded_samples_in_memory before writing to it.
        in_arr = make_complex_array(request.input_seq)
        out_arr = make_complex_array(n * request.num_records) if request.return_records else None
        record_z_arr = None
        progress_callback_ref = None
        progress_callback_context = None
        progress_callback_error: list[BaseException | None] = [None]

        keepalive: list[object] = []
        exec_options = _c_execution_options(request.exec_options)
        storage_opts = (
            _c_storage_options(request.storage, keepalive) if request.storage is not None else None
        )
        if request.record_z is not None:
            record_z_arr = (ctypes.c_double * len(request.record_z))(
                *[float(v) for v in request.record_z]
            )

        propagate_options = (
            self.lib.nlolib_propagate_options_default()
            if bool(getattr(self.lib, "_has_propagate_defaults", False))
            else NloPropagateOptions()
        )
        propagate_options.num_recorded_samples = request.num_records
        propagate_options.output_mode = (
            PROPAGATE_OUTPUT_FINAL_ONLY
            if request.num_records == 1 and request.record_z is None
            else PROPAGATE_OUTPUT_DENSE
        )
        propagate_options.return_records = int(bool(request.return_records))
        propagate_options.exec_options = (
            ctypes.pointer(exec_options) if exec_options is not None else None
        )
        propagate_options.storage_options = (
            ctypes.pointer(storage_opts)
            if storage_opts is not None
            else None
        )
        propagate_options.explicit_record_z = (
            ctypes.cast(record_z_arr, ctypes.POINTER(ctypes.c_double))
            if record_z_arr is not None
            else None
        )
        propagate_options.explicit_record_z_count = (
            len(request.record_z) if request.record_z is not None else 0
        )
        if request.progress_callback is not None:
            progress_callback_context = ctypes.py_object(request.progress_callback)

            def _progress_trampoline(info_ptr, user_data):
                callback = ctypes.cast(user_data, ctypes.POINTER(ctypes.py_object)).contents.value
                assert info_ptr
                progress_info = _progress_info_from_struct(info_ptr.contents)
                try:
                    result = callback(progress_info)
                except BaseException as exc:  # pragma: no cover
                    progress_callback_error[0] = exc
                    return 0
                if result is None:
                    return 1
                return 1 if bool(result) else 0

            progress_callback_ref = NloProgressCallback(_progress_trampoline)
            propagate_options.progress_callback = progress_callback_ref
            propagate_options.progress_user_data = ctypes.cast(
                ctypes.pointer(progress_callback_context),
                ctypes.c_void_p,
            )
        else:
            propagate_options.progress_callback = NloProgressCallback()
            propagate_options.progress_user_data = None

        storage_result = NloStorageResult()
        records_written = ctypes.c_size_t(0)
        step_events_out = (
            (NloStepEvent * int(request.step_history_capacity))()
            if request.capture_step_history and request.step_history_capacity > 0
            else None
        )
        step_events_written = ctypes.c_size_t(0)
        step_events_dropped = ctypes.c_size_t(0)
        propagate_output = (
            self.lib.nlolib_propagate_output_default()
            if bool(getattr(self.lib, "_has_propagate_defaults", False))
            else NloPropagateOutput()
        )
        propagate_output.output_records = (
            ctypes.cast(out_arr, ctypes.POINTER(NloComplex))
            if out_arr is not None
            else None
        )
        propagate_output.output_record_capacity = request.num_records if out_arr is not None else 0
        propagate_output.records_written = ctypes.pointer(records_written)
        propagate_output.storage_result = ctypes.pointer(storage_result)
        propagate_output.output_step_events = (
            ctypes.cast(step_events_out, ctypes.POINTER(NloStepEvent))
            if step_events_out is not None
            else None
        )
        propagate_output.output_step_event_capacity = (
            int(request.step_history_capacity)
            if step_events_out is not None
            else 0
        )
        propagate_output.step_events_written = ctypes.pointer(step_events_written)
        propagate_output.step_events_dropped = ctypes.pointer(step_events_dropped)

        status = int(
            self.lib.nlolib_propagate(
                sim_ptr,
                phys_ptr,
                n,
                ctypes.cast(in_arr, ctypes.POINTER(NloComplex)),
                ctypes.pointer(propagate_options),
                ctypes.pointer(propagate_output),
            )
        )
        if status not in {NLOLIB_STATUS_OK, NLOLIB_STATUS_ABORTED}:
            raise NlolibError(_enums.status_name(status), "nlolib_propagate failed")

        records_written_count = int(records_written.value)
        if out_arr is None:
            out_records: list[list[complex]] = []
        else:
            out_records = self._records_from_complex_array(out_arr, n, records_written_count)

        distance = float(request.sim_cfg.propagation.propagation_distance)
        if request.record_z is not None and len(request.record_z) == 1:
            # Core 9417e92 fills a single record with the final field whatever
            # position was asked for; report the position actually returned.
            z_axis = [distance] if records_written_count > 0 else []
        elif request.record_z is not None:
            z_axis = [float(v) for v in request.record_z[:records_written_count]]
        else:
            z_axis = self._build_z_axis(distance, records_written_count)
        final = list(out_records[-1]) if out_records else []
        meta: dict[str, object] = {
            "preset": request.meta_overrides.get("preset"),
            "records_requested": request.num_records,
            "records_written": records_written_count,
            "backend_requested": (
                request.exec_options.backend_type if request.exec_options is not None else "auto"
            ),
            "coupled": bool(request.meta_overrides.get("coupled", False)),
            "status": _enums.status_name(status),
        }
        step_history = (
            _step_history(
                step_events_out,
                int(step_events_written.value),
                int(step_events_dropped.value),
                int(request.step_history_capacity),
            )
            if step_events_out is not None
            else None
        )
        result = PropagationResult(
            records=out_records,
            z_axis=z_axis,
            final=final,
            step_history=step_history,
            storage_result=_storage_result(storage_result) if request.storage is not None else None,
            meta=meta,
        )
        if progress_callback_error[0] is not None:
            raise progress_callback_error[0]
        if status == NLOLIB_STATUS_ABORTED:
            raise PropagationAbortedError(result)
        return result

    @staticmethod
    def _records_from_complex_array(
        out_arr: ctypes.Array,
        num_time_samples: int,
        num_recorded_samples: int,
    ) -> list[list[complex]]:
        flat = complex_array_to_list(out_arr, num_time_samples * num_recorded_samples)
        records: list[list[complex]] = []
        for i in range(num_recorded_samples):
            start = i * num_time_samples
            records.append(flat[start : start + num_time_samples])
        return records

    @staticmethod
    def _build_z_axis(distance: float, num_records: int) -> list[float]:
        if num_records <= 0:
            return []
        if num_records == 1:
            return [distance]
        return [
            (distance * float(i)) / float(num_records - 1)
            for i in range(num_records)
        ]
