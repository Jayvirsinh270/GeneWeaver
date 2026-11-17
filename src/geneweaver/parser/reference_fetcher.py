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


def _mutate_base(c: str) -> str:
    """Returns a guaranteed nucleotide mismatch for synthetic spike injection."""
    mapping = {"A": "G", "C": "T", "G": "A", "T": "C"}
    return mapping.get(c.upper(), "C")


class ReferenceManager:
    """Manages local availability of reference genomes and synthetic benchmark sequences."""
    def __init__(self, data_dir: str = REFERENCE_DIR):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def get_preset_path(self, preset_key: str, default_guide: str = "GAGTCCGAGCAGAAGAAGAA") -> str:
        """
        Retrieves path to a preset FASTA file.
        If the file does not exist locally, generates it deterministically with known spike mutations
        tailored to the active guide sequence.
        """
        if preset_key not in PRESETS:
            preset_key = "synthetic_1mb"

        info = PRESETS[preset_key]
        
        # Determine cached benchmark filename keyed to the target guide
        safe_guide = "".join(c for c in default_guide.upper() if c in "ACGT")[:12]
        if not safe_guide:
            safe_guide = "GAGTCCGAGCAG"

        base, ext = os.path.splitext(info["filename"])
        if safe_guide == "GAGTCCGAGCAG":
            target_path = os.path.join(self.data_dir, info["filename"])
        else:
            target_path = os.path.join(self.data_dir, f"{base}_{safe_guide}{ext}")

        if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
            return target_path

        # Generate synthetic benchmark with deterministic spikes
        length = info["length_bp"]
        g = (default_guide.upper() + "A" * 20)[:20]
        spikes = [
            # 1. On-target perfect match (0 mismatches) -> High Risk
            (min(50_000, length // 10), g + "AGG"),
            # 2. Distal mismatch (position 1) -> High Risk
            (min(150_000, length // 5), _mutate_base(g[0]) + g[1:] + "AGG"),
            # 3. Seed region mismatch (position 19) -> Moderate Risk
            (min(300_000, length // 3), g[:18] + _mutate_base(g[18]) + g[19:] + "AGG"),
            # 4. 2 Mismatches (positions 2 and 12) -> Low/Moderate Risk
            (min(550_000, length // 2), g[0] + _mutate_base(g[1]) + g[2:11] + _mutate_base(g[11]) + g[12:] + "AGG"),
            # 5. 3 Mismatches (positions 1, 9, 18) -> Benign / Low Risk
            (min(800_000, int(length * 0.8)), _mutate_base(g[0]) + g[1:8] + _mutate_base(g[8]) + g[9:17] + _mutate_base(g[17]) + g[18:] + "AGG"),
        ]
        valid_spikes = [(pos, seq) for pos, seq in spikes if pos + len(seq) <= length]

        seq = generate_mock_chromosome(length_bp=length, spikes=valid_spikes, seed=42)
        write_fasta(seq, target_path, header=f"chr_{preset_key}")
        return target_path

    def list_available_presets(self):
        """Returns catalog of presets for UI select boxes."""
        return {k: v["name"] for k, v in PRESETS.items()}
