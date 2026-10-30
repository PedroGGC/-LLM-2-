"""Optional read-only GPU energy and host memory measurements."""
import ctypes
import ctypes.util
import os
from pathlib import Path
import sys


def host_memory():
    rss = pss = None
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss = int(value if sys.platform == "darwin" else value*1024)
    except (ImportError, OSError, ValueError): pass
    try:
        for line in Path("/proc/self/smaps_rollup").read_text().splitlines():
            if line.startswith("Pss:"): pss = int(line.split()[1])*1024
    except (OSError, ValueError): pass
    return {"peak_process_rss_bytes": rss, "process_pss_bytes": pss}


class EnergyMeter:
    """NVML board energy; unsupported telemetry stays null, never a FLOP proxy."""
    def __init__(self, device):
        self.library = None; self.handle = ctypes.c_void_p()
        if str(device).split(":")[0] != "cuda": return
        try:
            import torch
            name = ctypes.util.find_library("nvidia-ml") or "libnvidia-ml.so.1"
            library = ctypes.CDLL(name)
            if library.nvmlInit_v2() != 0: return
            self.library = library
            uuid = str(getattr(torch.cuda.get_device_properties(device), "uuid", ""))
            if uuid.startswith("GPU-"):
                status = library.nvmlDeviceGetHandleByUUID(uuid.encode(), ctypes.byref(self.handle))
            else:
                physical = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
                status = library.nvmlDeviceGetHandleByIndex_v2(int(physical), ctypes.byref(self.handle))
            if status != 0: self.close()
        except (OSError, ValueError, AttributeError): self.close()

    def energy_mj(self):
        if self.library is None: return None
        try:
            result = ctypes.c_ulonglong()
            status = self.library.nvmlDeviceGetTotalEnergyConsumption(self.handle, ctypes.byref(result))
            return int(result.value) if status == 0 else None
        except AttributeError: return None

    def power_watts(self):
        if self.library is None: return None
        try:
            result = ctypes.c_uint()
            status = self.library.nvmlDeviceGetPowerUsage(self.handle, ctypes.byref(result))
            return result.value/1000 if status == 0 else None
        except AttributeError: return None

    def close(self):
        if self.library is not None:
            try: self.library.nvmlShutdown()
            except AttributeError: pass
            self.library = None


def energy_result(start_mj, end_mj, power_samples, tokens, seconds):
    method = "unavailable"; joules = None
    if start_mj is not None and end_mj is not None and end_mj > start_mj:
        joules = (end_mj-start_mj)/1000; method = "nvml_energy_counter"
    elif len(power_samples) >= 2:
        joules = sum((b[0]-a[0])*(a[1]+b[1])/2 for a,b in zip(power_samples,power_samples[1:]))
        method = "estimated_from_sampled_board_power"
    return {"method": method, "gpu_joules": joules,
        "gpu_joules_per_training_target": joules/tokens if joules is not None and tokens else None,
        "measured_training_targets": tokens if joules is not None else None,
        "window_seconds": seconds if joules is not None else None,
        "scope": "isolated stabilized GPU board energy; host/startup/compilation excluded"}
