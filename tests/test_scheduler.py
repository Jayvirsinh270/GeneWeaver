"""
Unit Tests for Distributed Scheduler and Chunk Partitioner.
"""

import pytest
from geneweaver.scheduler.chunker import create_partition_plan
from geneweaver.scheduler.dask_pipeline import DistributedScheduler
from geneweaver.parser.mock_genome import generate_mock_chromosome

def test_partition_plan_creation():
    total_len = 100_000
    overlap = 22
    plans = create_partition_plan(total_length=total_len, num_partitions=4, overlap_bp=overlap)

    assert len(plans) == 4
    # First partition should have overlap
    assert plans[0].overlap_len == overlap
    assert plans[0].global_start == 0
    assert not plans[0].is_last

    # Last partition should end at total_len with 0 overlap
    assert plans[-1].is_last
    assert plans[-1].global_end == total_len
    assert plans[-1].overlap_len == 0

def test_distributed_scheduler_finds_spikes():
    guide = "GAGTCCGAGCAGAAGAAGAA"
    spikes = [
        (5_000, guide + "AGG"),
        (25_000, guide[:19] + "T" + "AGG"),
        (45_000, "A" + guide[1:] + "AGG")
    ]
    seq = generate_mock_chromosome(50_000, spikes=spikes, seed=42)

    scheduler = DistributedScheduler(num_workers=2, use_dask=False)
    hits, metrics = scheduler.execute_distributed_alignment(
        full_sequence=seq,
        target_guide=guide,
        pam_motif="NGG",
        max_mismatches=2,
        chrom_name="chrTest"
    )

    hit_coords = [h.global_pos for h in hits]
    assert 5_000 in hit_coords
    assert 25_000 in hit_coords
    assert 45_000 in hit_coords
    assert metrics.total_candidates_found >= 3
