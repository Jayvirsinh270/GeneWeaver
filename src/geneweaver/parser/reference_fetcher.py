"""
Reference Genome Manager & Preset Provider.
Manages both synthetic benchmark datasets with known ground-truth spikes and
real reference chromosome access for high-performance testing.
"""

import os
import urllib.request
import gzip
from typing import Optional, Tuple
from geneweaver.parser.mock_genome import generate_mock_chromosome, write_fasta

REFERENCE_DIR = os.path.abspath("data/references")

# Preset catalog
PRESETS = {
    "synthetic_1mb": {
        "name": "Synthetic Benchmark (1 Mbp - Rapid Demo)",
        "filename": "benchmark_1mb.fa",
        "length_bp": 1_000_000,
        "is_synthetic": True
    },
    "synthetic_10mb": {
        "name": "Synthetic Stress Test (10 Mbp - Heavy Load)",
        "filename": "benchmark_10mb.fa",
        "length_bp": 10_000_000,
        "is_synthetic": True
    },
    "synthetic_50mb": {
        "name": "Chromosome-Scale Benchmark (50 Mbp)",
        "filename": "benchmark_50mb.fa",
        "length_bp": 50_000_000,
        "is_synthetic": True
    }
}


class ReferenceManager:
    """Manages local availability of reference genomes and synthetic benchmark sequences."""
    def __init__(self, data_dir: str = REFERENCE_DIR):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def get_preset_path(self, preset_key: str, default_guide: str = "GAGTCCGAGCAGAAGAAGAA") -> str:
        """
        Retrieves path to a preset FASTA file.
        If the file does not exist locally, generates it deterministically with known spike mutations.
        """
        if preset_key not in PRESETS:
            preset_key = "synthetic_1mb"

        info = PRESETS[preset_key]
        target_path = os.path.join(self.data_dir, info["filename"])

        if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
            return target_path

        # Generate synthetic benchmark with deterministic spikes
        length = info["length_bp"]
        spikes = [
            (min(50_000, length // 10), default_guide + "AGG"),                    # On-target (0 mismatches)
            (min(150_000, length // 5), default_guide[:19] + "T" + "AGG"),         # Seed mismatch (pos 20)
            (min(300_000, length // 3), "A" + default_guide[1:] + "AGG"),          # Distal mismatch (pos 1)
            (min(600_000, length // 2), "A" + default_guide[1:9] + "T" + default_guide[10:] + "AGG") # 2 Mismatches
        ]
        valid_spikes = [(pos, seq) for pos, seq in spikes if pos + len(seq) <= length]

        seq = generate_mock_chromosome(length_bp=length, spikes=valid_spikes, seed=42)
        write_fasta(seq, target_path, header=f"chr_{preset_key}")
        return target_path

    def list_available_presets(self):
        """Returns catalog of presets for UI select boxes."""
        return {k: v["name"] for k, v in PRESETS.items()}
