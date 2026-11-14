"""
Genomic Workload Partitioner for Distributed Multi-Worker / Multi-GPU Scheduling.
Handles boundary seam overlap calculation (22bp) and coordinate re-indexing.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class PartitionPlan:
    """Represents a discrete slice assigned to an independent worker or GPU."""
    partition_id: int
    worker_id: int
    global_start: int
    global_end: int
    overlap_len: int
    is_last: bool

    @property
    def core_length(self) -> int:
        return self.global_end - self.global_start - (self.overlap_len if not self.is_last else 0)


def create_partition_plan(
    total_length: int,
    num_partitions: int,
    overlap_bp: int = 22
) -> List[PartitionPlan]:
    """
    Divides a sequence of total_length into balanced partitions.
    Appends overlap_bp to all non-terminal partitions to prevent boundary seam misses.
    """
    if num_partitions <= 0:
        num_partitions = 1

    base_partition_size = math.ceil(total_length / num_partitions)
    plans: List[PartitionPlan] = []

    for i in range(num_partitions):
        start = i * base_partition_size
        if start >= total_length:
            break

        is_last = (i == num_partitions - 1) or ((start + base_partition_size) >= total_length)
        if is_last:
            end = total_length
            overlap = 0
        else:
            end = min(start + base_partition_size + overlap_bp, total_length)
            overlap = overlap_bp

        plans.append(PartitionPlan(
            partition_id=i,
            worker_id=i % num_partitions,
            global_start=start,
            global_end=end,
            overlap_len=overlap,
            is_last=is_last
        ))

    return plans
