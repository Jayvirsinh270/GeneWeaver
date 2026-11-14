"""
Cutting Frequency Determination (CFD) Scoring Engine.
Reference: Doench et al., Nature Biotechnology (2016).
Computes off-target cleavage efficiency based on position-specific RNA:DNA base mismatch
and non-canonical PAM interaction affinities.
"""

from typing import Dict, Tuple

# Empirically measured PAM activity scores for SpCas9
PAM_WEIGHTS: Dict[str, float] = {
    "NGG": 1.000, "AGG": 1.000, "CGG": 1.000, "GGG": 1.000, "TGG": 1.000,
    "NAG": 0.259, "AAG": 0.259, "CAG": 0.259, "GAG": 0.259, "TAG": 0.259,
    "NGA": 0.069, "AGA": 0.069, "CGA": 0.069, "GGA": 0.069, "TGA": 0.069,
    "NGC": 0.022, "AGC": 0.022, "CGC": 0.022, "GGC": 0.022, "TGC": 0.022,
    "NGT": 0.017, "AGT": 0.017, "CGT": 0.017, "GGT": 0.017, "TGT": 0.017,
    "NNG": 0.005
}

# Position-dependent average penalties for mismatches (positions 1 to 20, 1-indexed)
# Seed region (pos 11-20) has substantially lower mismatch tolerance
DEFAULT_POSITION_PENALTY = [
    0.0,
    0.95, 0.94, 0.91, 0.92, 0.90,  # Pos 1-5 (Distal: High tolerance ~0.9)
    0.85, 0.82, 0.84, 0.78, 0.75,  # Pos 6-10
    0.68, 0.62, 0.55, 0.42, 0.38,  # Pos 11-15 (Seed transition)
    0.30, 0.25, 0.20, 0.15, 0.10   # Pos 16-20 (Seed proximal: Very low tolerance ~0.10)
]

# Base-specific mismatch multipliers (rGuide:dTarget)
BASE_PAIR_WEIGHTS: Dict[Tuple[str, str], float] = {
    # Transitions (Purine-Purine or Pyrimidine-Pyrimidine) are tolerated better
    ('A', 'G'): 0.85, ('G', 'A'): 0.85,
    ('C', 'T'): 0.80, ('T', 'C'): 0.80,
    # Transversions (Purine-Pyrimidine) cause greater structural distortion
    ('A', 'C'): 0.55, ('C', 'A'): 0.55,
    ('G', 'T'): 0.60, ('T', 'G'): 0.60,
    ('A', 'T'): 0.50, ('T', 'A'): 0.50,
    ('C', 'G'): 0.45, ('G', 'C'): 0.45,
}


def calculate_cfd_score(target_guide: str, off_target_seq: str, pam: str = "NGG") -> float:
    """
    Computes the CFD score between 20-nt guide and 20-nt off-target sequence.
    Multiplies base-mismatch penalties with the PAM motif affinity score.
    Returns:
        Score between 0.0 (zero cleavage) and 1.0 (maximal on-target cleavage).
    """
    if len(target_guide) != 20 or len(off_target_seq) != 20:
        raise ValueError(f"Sequences must be 20nt, got {len(target_guide)} and {len(off_target_seq)}")

    # 1. PAM Weight
    pam_upper = pam.upper()
    pam_score = PAM_WEIGHTS.get(pam_upper, 0.0)
    # Check general 2-base motif if 3-base exact key not found
    if pam_score == 0.0 and len(pam_upper) == 3:
        pam_tail = "N" + pam_upper[1:]
        pam_score = PAM_WEIGHTS.get(pam_tail, 0.0)

    if pam_score == 0.0:
        return 0.0

    # 2. Position and Base-Pair Specific Mismatch Multiplication
    cfd_score = pam_score
    guide_u = target_guide.upper()
    off_u = off_target_seq.upper()

    for i in range(20):
        pos = i + 1  # 1-indexed
        r_base = guide_u[i]
        d_base = off_u[i]

        if r_base != d_base:
            # Base mismatch weight
            bp_multiplier = BASE_PAIR_WEIGHTS.get((r_base, d_base), 0.60)
            pos_weight = DEFAULT_POSITION_PENALTY[pos]
            mismatch_factor = pos_weight * bp_multiplier
            cfd_score *= mismatch_factor

    return float(max(0.0, min(1.0, cfd_score)))


def classify_cfd_risk(score: float) -> str:
    """Classifies CFD cutting score into risk categories."""
    if score >= 0.50:
        return "HIGH_RISK"
    elif score >= 0.10:
        return "MODERATE_RISK"
    elif score >= 0.01:
        return "LOW_RISK"
    else:
        return "BENIGN"
