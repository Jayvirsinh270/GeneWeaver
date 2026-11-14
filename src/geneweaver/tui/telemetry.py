"""
Hardware Telemetry Engine.
Queries NVIDIA GPU device metrics (utilization, VRAM allocation, temperature, clock speeds)
using NVIDIA Management Library (NVML / pynvml).
Includes a simulated fallback provider for testing environments without active NVIDIA drivers.
"""

import time
import random
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class GpuDeviceStats:
    """Snapshot of a single GPU's live operational telemetry."""
    device_id: int
    name: str
    utilization_pct: float       # 0.0 to 100.0%
    vram_used_bytes: int         # Memory used in bytes
    vram_total_bytes: int        # Total VRAM in bytes
    temperature_c: int           # Core temperature in Celsius
    power_watts: float           # Power draw in Watts
    is_simulated: bool = False

    @property
    def vram_used_gb(self) -> float:
        return self.vram_used_bytes / (1024 ** 3)

    @property
    def vram_total_gb(self) -> float:
        return self.vram_total_bytes / (1024 ** 3)

    @property
    def vram_percent(self) -> float:
        if self.vram_total_bytes == 0:
            return 0.0
        return (self.vram_used_bytes / self.vram_total_bytes) * 100.0


class HardwareMonitor:
    """
    Polls NVIDIA hardware metrics via NVML at customizable frequencies.
    """
    def __init__(self, device_ids: Optional[List[int]] = None):
        self.device_ids = device_ids or [0]
        self._nvml_initialized = False
        self._device_handles = []

        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            for did in self.device_ids:
                if did < device_count:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(did)
                    self._device_handles.append((did, handle))
            if self._device_handles:
                self._nvml_initialized = True
        except Exception:
            self._nvml_initialized = False

    def query_stats(self) -> List[GpuDeviceStats]:
        """Queries current telemetry across all configured GPUs."""
        if self._nvml_initialized:
            return self._query_nvml()
        else:
            return self._query_simulated()

    def _query_nvml(self) -> List[GpuDeviceStats]:
        import pynvml
        stats_list = []
        for did, handle in self._device_handles:
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            except Exception:
                power = 0.0

            stats_list.append(GpuDeviceStats(
                device_id=did,
                name=name,
                utilization_pct=float(util.gpu),
                vram_used_bytes=int(mem.used),
                vram_total_bytes=int(mem.total),
                temperature_c=int(temp),
                power_watts=power,
                is_simulated=False
            ))
        return stats_list

    def _query_simulated(self) -> List[GpuDeviceStats]:
        """Provides simulated telemetry mimicking an active NVIDIA RTX 4090."""
        stats_list = []
        for did in self.device_ids:
            # Fluctuate utilization realistically around 96-99%
            util = random.uniform(96.0, 99.2)
            vram_total = 24 * (1024 ** 3)  # 24 GB
            vram_used = int(vram_total * random.uniform(0.18, 0.22))  # ~4.8 GB
            temp = random.randint(65, 68)
            power = random.uniform(280.0, 340.0)

            stats_list.append(GpuDeviceStats(
                device_id=did,
                name=f"NVIDIA GeForce RTX 4090 (Simulated Device {did})",
                utilization_pct=util,
                vram_used_bytes=vram_used,
                vram_total_bytes=vram_total,
                temperature_c=temp,
                power_watts=power,
                is_simulated=True
            ))
        return stats_list
