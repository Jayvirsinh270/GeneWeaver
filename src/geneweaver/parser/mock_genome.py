"""
Synthetic Genome Generator with Known Off-Target Spikes.
Used for ground-truth validation and CPU vs GPU benchmark calibrations.
"""

import os
import random
import argparse
from typing import List, Tuple

BASES = ['A', 'C', 'G', 'T']

def generate_mock_chromosome(
    length_bp: int,
    spikes: List[Tuple[int, str]] = None,
    seed: int = 42
) -> str:
    """
    Generates a pseudo-random DNA sequence and injects known spike mutations.
    """
    random.seed(seed)
    # Generate random sequence
    dna_list = [random.choice(BASES) for _ in range(length_bp)]
    
    # Inject spike sequences at specified coordinates
    if spikes:
        for coord, motif in spikes:
            motif_len = len(motif)
            if coord + motif_len <= length_bp:
                dna_list[coord:coord + motif_len] = list(motif)
                
    return "".join(dna_list)

def write_fasta(sequence: str, output_path: str, header: str = "chr_synthetic_benchmark"):
    """Writes a sequence to disk in standard FASTA format (80 chars per line)."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f">{header}\n")
        line_len = 80
        for i in range(0, len(sequence), line_len):
            f.write(sequence[i:i + line_len] + "\n")
    print(f"[Mock Genome] Wrote {len(sequence)} bp to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic FASTA genome with known spikes.")
    parser.add_argument("--length", type=int, default=10_000_000, help="Chromosome length in base pairs.")
    parser.add_argument("--output", type=str, default="data/mock_chr.fa", help="Output FASTA path.")
    parser.add_argument("--guide", type=str, default="GAGTCCGAGCAGAAGAAGAA", help="Target guide 20nt.")
    args = parser.parse_args()

    # Spikes: exact match + mutated variants
    sample_spikes = [
        (150_000, args.guide + "AGG"),               # 0 mismatches (On-target)
        (450_000, args.guide[:19] + "T" + "AGG"),     # 1 mismatch (Seed pos 20)
        (1_200_000, "A" + args.guide[1:] + "AGG"),    # 1 mismatch (Distal pos 1)
        (2_800_000, "A" + args.guide[1:9] + "T" + args.guide[10:] + "AGG"), # 2 mismatches
    ]

    seq = generate_mock_chromosome(args.length, spikes=sample_spikes)
    write_fasta(seq, args.output)

if __name__ == "__main__":
    main()
