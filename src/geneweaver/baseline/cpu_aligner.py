"""
CPU Reference Alignment Engine (Pure Python & BioPython Baseline).
Used to establish baseline performance metrics (time, throughput in Mbp/sec)
against which Week 2's CUDA GPU kernels will be benchmarked to prove >=100x speedup.
"""

import time
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional
from geneweaver.biology.hsu_zhang import calculate_hsu_zhang_score, classify_risk_tier
from geneweaver.parser.fasta_streamer import FastaStreamer, GenomicChunk


@dataclass
class OffTargetHit:
    """Represents a validated off-target alignment."""
    chrom_name: str
    global_pos: int          # Global 0-indexed chromosome coordinate
    strand: str              # '(+)' or '(-)'
    mismatches: int          # Number of mismatching bases (0 to max_mismatches)
    sequence: str            # 23-bp target locus sequence (20nt + 3nt PAM)
    hsu_score: float         # Hsu-Zhang cutting probability (0.0 to 1.0)
    risk_tier: str           # 'HIGH_RISK', 'MODERATE_RISK', 'LOW_RISK', 'BENIGN'


@dataclass
class AlignmentMetrics:
    """Captures empirical execution telemetry for benchmarking."""
    total_base_pairs: int
    elapsed_seconds: float
    throughput_bp_per_sec: float
    throughput_mbp_per_sec: float
    total_candidates_found: int
    high_risk_count: int


class CpuAligner:
    """
    Pure Python reference alignment engine implementing a sliding window
    over genomic sequences with PAM recognition and mismatch counting.
    """
    def __init__(self, target_guide: str, pam_motif: str = "NGG", max_mismatches: int = 4):
        if len(target_guide) != 20:
            raise ValueError(f"Target guide must be 20nt, got {len(target_guide)}")
        self.target_guide = target_guide.upper()
        self.pam_motif = pam_motif.upper()
        self.max_mismatches = max_mismatches
        self.window_size = 23  # 20nt guide + 3nt PAM

    def align_sequence(self, sequence: str, chrom_name: str = "chr1", start_offset: int = 0) -> List[OffTargetHit]:
        """
        Scans a sequence string using a sliding-window in pure Python.
        Checks both canonical SpCas9 PAM (NGG) and counts mismatches.
        """
        hits: List[OffTargetHit] = []
        n = len(sequence)
        guide = self.target_guide
        max_m = self.max_mismatches
        w_len = self.window_size

        # Slide window base by base
        for i in range(n - w_len + 1):
            window = sequence[i:i + w_len]
            
            # Step 1: Rapid PAM check (positions 21 and 22 must be 'GG')
            if window[21] != 'G' or window[22] != 'G':
                continue
                
            # Step 2: Compare 20nt guide sequence
            mismatches = 0
            for j in range(20):
                if window[j] != guide[j]:
                    mismatches += 1
                    if mismatches > max_m:
                        break
                        
            if mismatches <= max_m:
                hsu_score = calculate_hsu_zhang_score(guide, window[:20])
                risk = classify_risk_tier(hsu_score)
                hits.append(OffTargetHit(
                    chrom_name=chrom_name,
                    global_pos=start_offset + i,
                    strand="(+)",
                    mismatches=mismatches,
                    sequence=window,
                    hsu_score=hsu_score,
                    risk_tier=risk
                ))
                
        return hits

    def benchmark_chunk_stream(self, streamer: FastaStreamer, fasta_path: str) -> Tuple[List[OffTargetHit], AlignmentMetrics]:
        """
        Executes a complete scan across a streaming FASTA file, recording precise execution time.
        """
        all_hits: List[OffTargetHit] = []
        total_bp = 0
        start_time = time.perf_counter()

        for chunk in streamer.stream_chunks(fasta_path):
            chunk_hits = self.align_sequence(
                sequence=chunk.sequence,
                chrom_name=chunk.chrom_name,
                start_offset=chunk.global_start
            )
            
            # If not the last chunk, filter out hits that land in the overlap margin (avoid duplicates)
            if not chunk.is_last and chunk.overlap_len > 0:
                boundary = chunk.global_start + chunk.core_length
                chunk_hits = [h for h in chunk_hits if h.global_pos < boundary]
                
            all_hits.extend(chunk_hits)
            total_bp += chunk.core_length

        elapsed = time.perf_counter() - start_time
        throughput_bps = total_bp / elapsed if elapsed > 0 else 0.0
        throughput_mbps = (total_bp / 1_000_000.0) / elapsed if elapsed > 0 else 0.0

        high_risk = sum(1 for h in all_hits if h.risk_tier == "HIGH_RISK")

        metrics = AlignmentMetrics(
            total_base_pairs=total_bp,
            elapsed_seconds=elapsed,
            throughput_bp_per_sec=throughput_bps,
            throughput_mbp_per_sec=throughput_mbps,
            total_candidates_found=len(all_hits),
            high_risk_count=high_risk
        )

        return all_hits, metrics
