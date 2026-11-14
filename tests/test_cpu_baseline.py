"""
Unit Tests for Phase 1 CPU Baseline Alignment Engine.
"""

import pytest
from geneweaver.baseline.cpu_aligner import CpuAligner
from geneweaver.parser.mock_genome import generate_mock_chromosome

def test_cpu_aligner_finds_injected_spikes():
    guide = "GAGTCCGAGCAGAAGAAGAA"
    aligner = CpuAligner(target_guide=guide, pam_motif="NGG", max_mismatches=2)

    # Injected spikes:
    # 1. Perfect on-target match at 100
    # 2. 1 mismatch (seed) at 300
    # 3. 2 mismatches at 600
    # 4. Non-PAM motif (AAG) at 800 (should be rejected)
    spikes = [
        (100, guide + "AGG"),
        (300, guide[:19] + "T" + "AGG"),
        (600, "A" + guide[1:9] + "T" + guide[10:] + "AGG"),
        (800, guide + "AAG")
    ]
    seq = generate_mock_chromosome(1000, spikes=spikes, seed=42)
    hits = aligner.align_sequence(seq, chrom_name="chrTest", start_offset=0)

    # Verify that spikes 1, 2, 3 are detected
    hit_positions = [h.global_pos for h in hits]
    assert 100 in hit_positions
    assert 300 in hit_positions
    assert 600 in hit_positions

    # Verify spike 4 (non-PAM) was rejected
    assert 800 not in hit_positions

    # Check on-target properties
    on_target = [h for h in hits if h.global_pos == 100][0]
    assert on_target.mismatches == 0
    assert on_target.hsu_score == 1.0
    assert on_target.risk_tier == "HIGH_RISK"
