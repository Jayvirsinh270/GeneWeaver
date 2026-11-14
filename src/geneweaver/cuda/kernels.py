"""
Numba CUDA Kernel Engine.
Implements GPU-accelerated CRISPR sequence alignment kernels running directly on NVIDIA hardware.
Bypasses Python GIL and offloads millions of base comparisons across thousands of CUDA cores.
Includes vectorized fallback when running on non-GPU hosts or CI/CD test runners.
"""

import numpy as np
from typing import Tuple, List, Optional
from geneweaver.biology.hsu_zhang import calculate_hsu_zhang_score, classify_risk_tier
from geneweaver.baseline.cpu_aligner import OffTargetHit

# Check CUDA hardware availability
CUDA_AVAILABLE = False
try:
    from numba import cuda
    if cuda.is_available():
        CUDA_AVAILABLE = True
except Exception:
    CUDA_AVAILABLE = False


if CUDA_AVAILABLE:
    @cuda.jit
    def crispr_align_global_kernel(
        genome_arr,        # In: uint8 array of genomic bases (ASCII)
        genome_len,        # In: Total length of genomic chunk
        target_guide,      # In: 20nt target guide sequence (ASCII)
        max_mismatches,    # In: Maximum allowed mismatches (e.g. 4)
        out_coords,        # Out: Device array storing off-target genome coordinates
        out_mismatches,    # Out: Device array storing mismatch count
        out_count          # Out: Device array [0] acting as atomic counter
    ):
        """
        Week 2 Milestone: Global Memory Alignment Kernel.
        Each GPU thread inspects a single 23-bp window at coordinate tid.
        """
        tid = cuda.grid(1)
        window_size = 23

        # Boundary check
        if tid > (genome_len - window_size):
            return

        # 1. Early PAM Check: Canonical SpCas9 requires 'GG' at positions 21 and 22
        # ASCII 71 == 'G'
        if genome_arr[tid + 21] != 71 or genome_arr[tid + 22] != 71:
            return

        # 2. Compare 20nt guide against locus window
        mismatches = 0
        for j in range(20):
            if genome_arr[tid + j] != target_guide[j]:
                mismatches += 1
                if mismatches > max_mismatches:
                    return

        # 3. Match detected: Atomically reserve output index
        idx = cuda.atomic.add(out_count, 0, 1)
        if idx < out_coords.shape[0]:
            out_coords[idx] = tid
            out_mismatches[idx] = mismatches


def launch_global_alignment_kernel(
    genome_bytes: np.ndarray,
    target_guide_bytes: np.ndarray,
    max_mismatches: int = 4,
    threads_per_block: int = 256,
    max_candidates: int = 500_000,
    chrom_name: str = "chr1",
    global_offset: int = 0
) -> List[OffTargetHit]:
    """
    Host driver for launching the Numba CUDA global memory alignment kernel.
    Handles Host-to-Device (HtoD) and Device-to-Host (DtoH) transfers.
    """
    seq_len = len(genome_bytes)
    if seq_len < 23:
        return []

    if CUDA_AVAILABLE:
        from numba import cuda

        # Calculate grid dimensions
        blocks_per_grid = (seq_len + threads_per_block - 1) // threads_per_block

        # Allocate device memory
        d_genome = cuda.to_device(genome_bytes)
        d_guide = cuda.to_device(target_guide_bytes)

        d_out_coords = cuda.device_array(max_candidates, dtype=np.int64)
        d_out_mismatches = cuda.device_array(max_candidates, dtype=np.int32)
        d_out_count = cuda.to_device(np.zeros(1, dtype=np.int32))

        # Launch kernel on GPU
        crispr_align_global_kernel[blocks_per_grid, threads_per_block](
            d_genome,
            seq_len,
            d_guide,
            max_mismatches,
            d_out_coords,
            d_out_mismatches,
            d_out_count
        )

        # Asynchronously copy results back to host
        h_count = d_out_count.copy_to_host()[0]
        actual_hits = min(h_count, max_candidates)

        if actual_hits == 0:
            return []

        h_coords = d_out_coords[:actual_hits].copy_to_host()
        h_mismatches = d_out_mismatches[:actual_hits].copy_to_host()

        # Build OffTargetHit records
        hits = []
        guide_str = target_guide_bytes.tobytes().decode('ascii')
        for i in range(actual_hits):
            local_pos = int(h_coords[i])
            m = int(h_mismatches[i])
            window_str = genome_bytes[local_pos:local_pos + 23].tobytes().decode('ascii')
            hsu = calculate_hsu_zhang_score(guide_str, window_str[:20])
            risk = classify_risk_tier(hsu)

            hits.append(OffTargetHit(
                chrom_name=chrom_name,
                global_pos=global_offset + local_pos,
                strand="(+)",
                mismatches=m,
                sequence=window_str,
                hsu_score=hsu,
                risk_tier=risk
            ))
        return hits
    else:
        # Vectorized CPU Fallback when CUDA device is offline
        return _fallback_cpu_scan(
            genome_bytes, target_guide_bytes, max_mismatches,
            chrom_name, global_offset
        )


def _fallback_cpu_scan(
    genome_bytes: np.ndarray,
    target_guide_bytes: np.ndarray,
    max_mismatches: int,
    chrom_name: str,
    global_offset: int
) -> List[OffTargetHit]:
    """Vectorized CPU fallback ensuring tests and CI/CD can run on any machine."""
    seq_len = len(genome_bytes)
    hits = []
    guide_str = target_guide_bytes.tobytes().decode('ascii')

    for i in range(seq_len - 22):
        # PAM GG check
        if genome_bytes[i + 21] == 71 and genome_bytes[i + 22] == 71:
            # Count mismatches
            m = np.count_nonzero(genome_bytes[i:i + 20] != target_guide_bytes)
            if m <= max_mismatches:
                window_str = genome_bytes[i:i + 23].tobytes().decode('ascii')
                hsu = calculate_hsu_zhang_score(guide_str, window_str[:20])
                hits.append(OffTargetHit(
                    chrom_name=chrom_name,
                    global_pos=global_offset + i,
                    strand="(+)",
                    mismatches=m,
                    sequence=window_str,
                    hsu_score=hsu,
                    risk_tier=classify_risk_tier(hsu)
                ))
    return hits
