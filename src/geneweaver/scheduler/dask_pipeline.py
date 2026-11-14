"""
Distributed Alignment Pipeline & Scheduler.
Coordinates workload distribution across multiple GPUs and CPU worker nodes.
Supports Dask Distributed DAG execution with seamless parallel fallback.
"""

import time
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
import numpy as np

from geneweaver.scheduler.chunker import create_partition_plan, PartitionPlan
from geneweaver.cuda.kernels import launch_global_alignment_kernel
from geneweaver.biology.hsu_zhang import calculate_hsu_zhang_score, classify_risk_tier
from geneweaver.biology.cfd import calculate_cfd_score
from geneweaver.baseline.cpu_aligner import OffTargetHit, AlignmentMetrics

DASK_AVAILABLE = False
try:
    import dask
    from dask import delayed
    from distributed import Client, LocalCluster
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False


def _align_partition_worker(
    partition_seq: str,
    target_guide: str,
    max_mismatches: int,
    chrom_name: str,
    global_offset: int,
    overlap_len: int,
    is_last: bool
) -> List[OffTargetHit]:
    """
    Task worker executed across distributed GPU/CPU worker processes.
    Processes a single partition and filters boundary seam duplicates.
    """
    seq_bytes = np.frombuffer(partition_seq.encode('ascii'), dtype=np.uint8)
    guide_bytes = np.frombuffer(target_guide.encode('ascii'), dtype=np.uint8)

    # Launch kernel
    raw_hits = launch_global_alignment_kernel(
        seq_bytes, guide_bytes, max_mismatches=max_mismatches,
        chrom_name=chrom_name, global_offset=global_offset
    )

    # Filter out hits that fall within the tail overlap margin (prevent duplicate counting)
    if not is_last and overlap_len > 0:
        core_end = global_offset + len(partition_seq) - overlap_len
        filtered_hits = [h for h in raw_hits if h.global_pos < core_end]
    else:
        filtered_hits = raw_hits

    return filtered_hits


class DistributedScheduler:
    """
    Distributes sequence arrays across multiple GPUs or CPU worker processes.
    Ensures load balancing and zero boundary seam omissions.
    """
    def __init__(self, num_workers: int = 2, use_dask: bool = True):
        self.num_workers = max(1, num_workers)
        self.use_dask = use_dask and DASK_AVAILABLE
        self._client = None

    def execute_distributed_alignment(
        self,
        full_sequence: str,
        target_guide: str,
        pam_motif: str = "NGG",
        max_mismatches: int = 4,
        chrom_name: str = "chr1"
    ) -> Tuple[List[OffTargetHit], AlignmentMetrics]:
        """
        Partitions the genomic sequence and dispatches jobs across distributed workers.
        """
        total_len = len(full_sequence)
        start_time = time.perf_counter()

        # Step 1: Create Overlapped Partition Plan
        plans = create_partition_plan(
            total_length=total_len,
            num_partitions=self.num_workers,
            overlap_bp=22
        )

        all_hits: List[OffTargetHit] = []

        if self.use_dask:
            # Dask Distributed Task Graph Execution
            delayed_tasks = []
            for plan in plans:
                part_seq = full_sequence[plan.global_start:plan.global_end]
                task = delayed(_align_partition_worker)(
                    partition_seq=part_seq,
                    target_guide=target_guide,
                    max_mismatches=max_mismatches,
                    chrom_name=chrom_name,
                    global_offset=plan.global_start,
                    overlap_len=plan.overlap_len,
                    is_last=plan.is_last
                )
                delayed_tasks.append(task)

            # Compute distributed DAG in parallel
            computed_results = dask.compute(*delayed_tasks)
            for res in computed_results:
                all_hits.extend(res)
        else:
            # Parallel Multi-Worker Fallback (concurrent.futures)
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = []
                for plan in plans:
                    part_seq = full_sequence[plan.global_start:plan.global_end]
                    f = executor.submit(
                        _align_partition_worker,
                        part_seq,
                        target_guide,
                        max_mismatches,
                        chrom_name,
                        plan.global_start,
                        plan.overlap_len,
                        plan.is_last
                    )
                    futures.append(f)

                for f in futures:
                    all_hits.extend(f.result())

        elapsed = time.perf_counter() - start_time
        throughput_bps = total_len / elapsed if elapsed > 0 else 0.0
        throughput_mbps = (total_len / 1_000_000.0) / elapsed if elapsed > 0 else 0.0

        # Sort all hits by chromosomal coordinate
        all_hits.sort(key=lambda h: h.global_pos)
        high_risk = sum(1 for h in all_hits if h.risk_tier == "HIGH_RISK")

        metrics = AlignmentMetrics(
            total_base_pairs=total_len,
            elapsed_seconds=elapsed,
            throughput_bp_per_sec=throughput_bps,
            throughput_mbp_per_sec=throughput_mbps,
            total_candidates_found=len(all_hits),
            high_risk_count=high_risk
        )

        return all_hits, metrics
