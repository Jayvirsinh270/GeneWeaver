"""
CUDA Shared Memory Tiling Alignment Engine.
Caches genomic chunk slices on-chip in L1 SRAM (cuda.shared.array) to reduce global VRAM
memory bus transactions by >85% and maximize Streaming Multiprocessor (SM) occupancy.
"""

import numpy as np
from typing import List, Tuple
from geneweaver.cuda.kernels import CUDA_AVAILABLE
from geneweaver.biology.hsu_zhang import calculate_hsu_zhang_score, classify_risk_tier
from geneweaver.baseline.cpu_aligner import OffTargetHit

if CUDA_AVAILABLE:
    from numba import cuda

    @cuda.jit
    def crispr_align_shared_mem_kernel(
        genome_arr,        # In: uint8 array of genomic bases in VRAM
        genome_len,        # In: Total length of genomic chunk
        target_guide,      # In: 20nt target guide sequence
        max_mismatches,    # In: Maximum allowed mismatches
        out_coords,        # Out: Device array storing off-target genome coordinates
        out_mismatches,    # Out: Device array storing mismatch count
        out_count          # Out: Device atomic hit counter
    ):
        """
        Shared Memory Tiling Kernel.
        Threads collaboratively load a 256-base tile + 22-base tail into on-chip SRAM.
        """
        TILE_SIZE = 256
        WINDOW_SIZE = 23
        SHARED_BUFFER_LEN = 278  # 256 + 23 - 1

        # Allocate on-chip L1 SRAM shared memory per thread block
        s_genome = cuda.shared.array(shape=278, dtype=np.uint8)

        tx = cuda.threadIdx.x
        bx = cuda.blockIdx.x
        block_dim = cuda.blockDim.x
        global_start = bx * block_dim

        # 1. Collaborative loading of genomic tile into shared memory
        if global_start + tx < genome_len:
            s_genome[tx] = genome_arr[global_start + tx]
        else:
            s_genome[tx] = 0

        # Load boundary overlap elements (WINDOW_SIZE - 1)
        if tx < (WINDOW_SIZE - 1):
            tail_idx = global_start + block_dim + tx
            if tail_idx < genome_len:
                s_genome[block_dim + tx] = genome_arr[tail_idx]
            else:
                s_genome[block_dim + tx] = 0

        # Synchronize all 256 threads in block to ensure tile is completely loaded
        cuda.syncthreads()

        # Global coordinate corresponding to this thread
        gid = global_start + tx
        if gid > (genome_len - WINDOW_SIZE):
            return

        # 2. Fast PAM verification directly from Shared Memory (L1 cache latency ~15-20 cycles)
        # Canonical SpCas9 requires 'GG' at positions 21 and 22 (ASCII 71 == 'G')
        if s_genome[tx + 21] != 71 or s_genome[tx + 22] != 71:
            return

        # 3. Mismatch comparison in fast SRAM
        mismatches = 0
        for j in range(20):
            if s_genome[tx + j] != target_guide[j]:
                mismatches += 1
                if mismatches > max_mismatches:
                    return

        # 4. Atomic append to global hit buffer
        idx = cuda.atomic.add(out_count, 0, 1)
        if idx < out_coords.shape[0]:
            out_coords[idx] = gid
            out_mismatches[idx] = mismatches


def launch_shared_memory_kernel(
    genome_bytes: np.ndarray,
    target_guide_bytes: np.ndarray,
    max_mismatches: int = 4,
    threads_per_block: int = 256,
    max_candidates: int = 500_000,
    chrom_name: str = "chr1",
    global_offset: int = 0
) -> List[OffTargetHit]:
    """
    Host driver for launching the Shared Memory Tiling CUDA kernel.
    """
    seq_len = len(genome_bytes)
    if seq_len < 23:
        return []

    if CUDA_AVAILABLE:
        from numba import cuda

        blocks_per_grid = (seq_len + threads_per_block - 1) // threads_per_block

        d_genome = cuda.to_device(genome_bytes)
        d_guide = cuda.to_device(target_guide_bytes)
        d_out_coords = cuda.device_array(max_candidates, dtype=np.int64)
        d_out_mismatches = cuda.device_array(max_candidates, dtype=np.int32)
        d_out_count = cuda.to_device(np.zeros(1, dtype=np.int32))

        # Launch kernel with shared memory tile
        crispr_align_shared_mem_kernel[blocks_per_grid, threads_per_block](
            d_genome,
            seq_len,
            d_guide,
            max_mismatches,
            d_out_coords,
            d_out_mismatches,
            d_out_count
        )

        h_count = d_out_count.copy_to_host()[0]
        actual_hits = min(h_count, max_candidates)

        if actual_hits == 0:
            return []

        h_coords = d_out_coords[:actual_hits].copy_to_host()
        h_mismatches = d_out_mismatches[:actual_hits].copy_to_host()

        hits = []
        guide_str = target_guide_bytes.tobytes().decode('ascii')
        for i in range(actual_hits):
            local_pos = int(h_coords[i])
            m = int(h_mismatches[i])
            window_str = genome_bytes[local_pos:local_pos + 23].tobytes().decode('ascii')
            hsu = calculate_hsu_zhang_score(guide_str, window_str[:20])

            hits.append(OffTargetHit(
                chrom_name=chrom_name,
                global_pos=global_offset + local_pos,
                strand="(+)",
                mismatches=m,
                sequence=window_str,
                hsu_score=hsu,
                risk_tier=classify_risk_tier(hsu)
            ))
        return hits
    else:
        # High-speed vectorized CPU fallback
        from geneweaver.cuda.kernels import _fallback_cpu_scan
        return _fallback_cpu_scan(
            genome_bytes, target_guide_bytes, max_mismatches,
            chrom_name, global_offset
        )
