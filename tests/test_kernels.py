"""
Unit Tests for CUDA Kernel Execution & Fallback.
"""

import pytest
import numpy as np
from geneweaver.cuda.kernels import launch_global_alignment_kernel
from geneweaver.parser.mock_genome import generate_mock_chromosome

def test_kernel_alignment_detects_spikes():
    guide = "GAGTCCGAGCAGAAGAAGAA"
    guide_bytes = np.frombuffer(guide.encode('ascii'), dtype=np.uint8)

    spikes = [
        (50, guide + "AGG"),             # 0 mismatches
        (200, guide[:19] + "T" + "AGG"),  # 1 mismatch
        (400, guide + "AAG")             # Non-PAM (should be rejected)
    ]
    seq = generate_mock_chromosome(500, spikes=spikes, seed=123)
    seq_bytes = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)

    hits = launch_global_alignment_kernel(
        seq_bytes, guide_bytes, max_mismatches=2,
        chrom_name="chrTest", global_offset=1000
    )

    hit_positions = [h.global_pos for h in hits]
    # Spikes were placed at offset 50 and 200 (global 1050 and 1200)
    assert 1050 in hit_positions
    assert 1200 in hit_positions

    # Non-PAM spike at 1400 should not be reported
    assert 1400 not in hit_positions

    on_target = [h for h in hits if h.global_pos == 1050][0]
    assert on_target.mismatches == 0
    assert on_target.hsu_score == 1.0
