"""
Hsu-Zhang CRISPR Off-Target Cleavage Probability Scoring Engine.
Reference: Hsu et al., Nature Biotechnology (2013)
"""

import numpy as np

# Empirical position penalties W[i] for positions 1 to 20 (1-indexed)
# Index 0 is unused padding so index matches base position directly
HSU_WEIGHTS = np.array([
    0.0,
    0.000, 0.000, 0.014, 0.000, 0.000,  # Pos 1 - 5
    0.395, 0.317, 0.000, 0.389, 0.079,  # Pos 6 - 10
    0.445, 0.508, 0.613, 0.851, 0.732,  # Pos 11 - 15 (Seed transition)
    0.828, 0.615, 0.804, 0.685, 0.583   # Pos 16 - 20 (Seed proximal)
], dtype=np.float64)

def calculate_hsu_zhang_score(target_guide: str, off_target_seq: str) -> float:
    """
    Computes the Hsu-Zhang score between a 20-nt guide and a 20-nt off-target sequence.
    Score ranges from 0.0 (negligible cleavage) to 1.0 (maximal/on-target cleavage).
    """
    if len(target_guide) != 20 or len(off_target_seq) != 20:
        raise ValueError(f"Sequences must be exactly 20nt. Got {len(target_guide)} and {len(off_target_seq)}")

    # Identify 1-indexed mismatch positions
    mismatches = [i + 1 for i in range(20) if target_guide[i].upper() != off_target_seq[i].upper()]
    n = len(mismatches)

    if n == 0:
        return 1.0

    # 1. Product of (1 - W[i])
    term1 = 1.0
    for pos in mismatches:
        term1 *= (1.0 - HSU_WEIGHTS[pos])

    # 2. Mean pairwise distance penalty
    if n > 1:
        total_dist = sum(mismatches[k + 1] - mismatches[k] for k in range(n - 1))
        d_bar = total_dist / (n - 1)
    else:
        d_bar = 19.0

    term2 = 1.0 / (((19.0 - d_bar) / 19.0) * 4.0 + 1.0)

    # 3. Mismatch count penalty
    term3 = 1.0 / (n * n)

    score = term1 * term2 * term3
    return float(score)

def classify_risk_tier(score: float) -> str:
    """Classifies candidate off-target into clinical risk tiers."""
    if score >= 0.50:
        return "HIGH_RISK"
    elif score >= 0.10:
        return "MODERATE_RISK"
    elif score >= 0.01:
        return "LOW_RISK"
    else:
        return "BENIGN"
