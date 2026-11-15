"""
Unit Tests for Phase 4: Shared Memory Tiling & Bit-Parallel POPC Kernels.
"""

import pytest
import numpy as np
from geneweaver.cuda.shared_mem_kernel import launch_shared_memory_kernel
from geneweaver.cuda.bit_parallel import launch_bitparallel_kernel
from geneweaver.parser.mock_genome import generate_mock_chromosome

def test_shared_memory_kernel():
    guide = "GAGTCCGAGCAGAAGAAGAA"
    guide_bytes = np.frombuffer(guide.encode('ascii'), dtype=np.uint8)

    spikes = [
        (100, guide + "AGG"),             # 0 mismatches
        (300, guide[:19] + "T" + "AGG"),  # 1 mismatch
    ]
    seq = generate_mock_chromosome(600, spikes=spikes, seed=42)
    seq_bytes = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)

    hits = launch_shared_memory_kernel(
        seq_bytes, guide_bytes, max_mismatches=2,
        chrom_name="chrShared", global_offset=0
    )

    hit_coords = [h.global_pos for h in hits]
    assert 100 in hit_coords
    assert 300 in hit_coords

def test_bitparallel_kernel():
    guide = "GAGTCCGAGCAGAAGAAGAA"
    spikes = [
        (64, guide + "AGG"),              # Align on word boundary (64 % 16 == 0)
        (160, guide[:19] + "T" + "AGG"),  # Align on word boundary (160 % 16 == 0)
    ]
    seq = generate_mock_chromosome(400, spikes=spikes, seed=42)

    hits = launch_bitparallel_kernel(
        sequence_str=seq,
        target_guide_str=guide,
        max_mismatches=2,
        chrom_name="chrBit",
        global_offset=0
    )

    hit_coords = [h.global_pos for h in hits]
    assert 64 in hit_coords
    assert 160 in hit_coords
