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


IUPAC_MAP = {
    'A': {'A'}, 'C': {'C'}, 'G': {'G'}, 'T': {'T'},
    'R': {'A', 'G'}, 'Y': {'C', 'T'}, 'S': {'G', 'C'}, 'W': {'A', 'T'},
    'K': {'G', 'T'}, 'M': {'A', 'C'}, 'B': {'C', 'G', 'T'}, 'D': {'A', 'G', 'T'},
    'H': {'A', 'C', 'T'}, 'V': {'A', 'C', 'G'}, 'N': {'A', 'C', 'G', 'T'}
}


def matches_pam(seq_pam: str, motif: str) -> bool:
    """Checks whether a sequence PAM matches an IUPAC degenerate motif."""
    if len(seq_pam) != len(motif):
        return False
    for s_char, m_char in zip(seq_pam.upper(), motif.upper()):
        allowed = IUPAC_MAP.get(m_char, {'A', 'C', 'G', 'T'})
        if s_char not in allowed:
            return False
    return True


class CpuAligner:
    """
    Pure Python reference alignment engine implementing a sliding window
    over genomic sequences with PAM recognition and mismatch counting.
    Supports SpCas9, SaCas9, Cas12a/Cpf1, and CasMINI with 5' and 3' PAM orientations.
    """
    def __init__(
        self,
        target_guide: str,
        pam_motif: str = "NGG",
        max_mismatches: int = 4,
        pam_orientation: str = "3_PRIME"
    ):
        self.target_guide = target_guide.upper()
        self.pam_motif = pam_motif.upper()
        self.max_mismatches = max_mismatches
        self.pam_orientation = pam_orientation.upper()
        self.guide_len = len(self.target_guide)
        self.pam_len = len(self.pam_motif)
        self.window_size = self.guide_len + self.pam_len

    def align_sequence(self, sequence: str, chrom_name: str = "chr1", start_offset: int = 0) -> List[OffTargetHit]:
        """
        Scans a sequence string using a sliding-window in pure Python.
        Checks PAM recognition (5' or 3' orientation) and counts guide mismatches.
        """
        hits: List[OffTargetHit] = []
        n = len(sequence)
        guide = self.target_guide
        max_m = self.max_mismatches
        w_len = self.window_size
        g_len = self.guide_len
        p_len = self.pam_len
        is_5prime = (self.pam_orientation == "5_PRIME")

        # Slide window base by base
        for i in range(n - w_len + 1):
            window = sequence[i:i + w_len]
            
            # Step 1: Rapid PAM check based on orientation
            if is_5prime:
                win_pam = window[:p_len]
                win_guide = window[p_len:]
            else:
                win_guide = window[:g_len]
                win_pam = window[g_len:]

            # Fast path check for canonical NGG
            if self.pam_motif == "NGG":
                if win_pam[1] != 'G' or win_pam[2] != 'G':
                    continue
            else:
                if not matches_pam(win_pam, self.pam_motif):
                    continue
                
            # Step 2: Compare guide sequence
            mismatches = 0
            for j in range(g_len):
                if win_guide[j] != guide[j]:
                    mismatches += 1
                    if mismatches > max_m:
                        break
                        
            if mismatches <= max_m:
                hsu_score = calculate_hsu_zhang_score(guide, win_guide)
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
