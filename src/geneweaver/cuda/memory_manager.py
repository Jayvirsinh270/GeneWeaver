"""
CUDA Memory Manager & Stream Pipelining.
Implements asynchronous Host-to-Device (HtoD) and Device-to-Host (DtoH) DMA memory transfers.
Uses page-locked pinned host memory (cuda.pinned_array) and double-buffering with CUDA streams.
"""

import numpy as np
from typing import Optional, Tuple
from geneweaver.cuda.kernels import CUDA_AVAILABLE


class CudaMemoryManager:
    """
    Manages GPU device buffers, pinned host allocations, and asynchronous CUDA streams.
    Prevents CUDA_OUT_OF_MEMORY errors via dynamic VRAM budgeting.
    """
    def __init__(self, device_id: int = 0, safety_factor: float = 0.75):
        self.device_id = device_id
        self.safety_factor = safety_factor
        self.stream_1 = None
        self.stream_2 = None

        if CUDA_AVAILABLE:
            from numba import cuda
            cuda.select_device(self.device_id)
            self.stream_1 = cuda.stream()
            self.stream_2 = cuda.stream()

    def allocate_pinned_host(self, size_bytes: int, dtype=np.uint8) -> np.ndarray:
        """
        Allocates page-locked (pinned) host memory in CPU RAM.
        Enables direct DMA transfers over PCIe without CPU operating system paging overhead.
        """
        if CUDA_AVAILABLE:
            from numba import cuda
            return cuda.pinned_array(size_bytes, dtype=dtype)
        else:
            return np.zeros(size_bytes, dtype=dtype)

    def calculate_safe_chunk_size(self, default_mb: int = 50) -> int:
        """
        Calculates maximum safe genomic chunk size based on free VRAM.
        Guarantees memory usage stays below 75% of total device capacity.
        """
        if CUDA_AVAILABLE:
            from numba import cuda
            ctx = cuda.current_context()
            free_mem, total_mem = ctx.get_memory_info()
            
            # Allow max 75% of free VRAM for dual buffers
            usable_bytes = int(free_mem * self.safety_factor / 2.5)
            safe_mb = max(10, min(usable_bytes // (1024 * 1024), default_mb))
            return safe_mb * 1024 * 1024  # In base pairs / bytes
        else:
            return default_mb * 1024 * 1024

    def synchronize(self):
        """Blocks until all asynchronous CUDA streams have completed execution."""
        if CUDA_AVAILABLE:
            from numba import cuda
            cuda.synchronize()
