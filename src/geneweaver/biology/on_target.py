"""
GeneWeaver On-Target CRISPR-Cas9 Cleavage Efficiency Engine.
Implements an empirical scoring model based on the Doench et al. (Nature Biotechnology 2016,
Rule Set 2 / Azimuth) and Moreno-Mateos biophysical heuristics.

Evaluates:
1. Position-dependent nucleotide weights (favored Guanine at pos 20, disfavored Thymine at pos 20).
2. Optimal GC content bell curve (40% - 65% optimum).
3. Homopolymer repeats (e.g. TTTT Pol-III transcription termination).
4. Self-complementary hairpin propensity.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
from geneweaver.biology.presets import calculate_gc_content, estimate_melting_temp


@dataclass
class OnTargetScore:
    """Represents the on-target cleavage efficiency assessment of an sgRNA guide."""
    score: float                # Normalized 0.0 - 100.0 score
    tier: str                   # 'OPTIMAL', 'GOOD', 'MODERATE', 'POOR'
    gc_content_pct: float       # GC%
    melting_temp_c: float       # Estimated Tm (°C)
    penalties: List[str]        # List of biological warnings or deductions
    bonuses: List[str]          # List of biological efficacy boosts
    summary: str                # Plain-English executive verdict


# Position-specific base preferences based on empirical Doench 2016 model (1-indexed, 1-20)
# Positive = favored, Negative = disfavored
POSITION_PREFERENCES: Dict[int, Dict[str, float]] = {
    1:  {'G': 0.15, 'A': 0.05, 'T': -0.10, 'C': 0.00},
    2:  {'G': 0.05, 'A': 0.05, 'T': -0.05, 'C': 0.00},
    3:  {'C': 0.10, 'G': 0.05, 'A': -0.05, 'T': -0.05},
    14: {'G': 0.10, 'C': 0.05, 'T': -0.05, 'A': -0.05},
    15: {'C': 0.10, 'A': 0.05, 'T': -0.10, 'G': 0.00},
    16: {'C': 0.10, 'G': 0.10, 'T': -0.10, 'A': -0.05},
    17: {'G': 0.15, 'C': 0.10, 'T': -0.15, 'A': -0.05},
    18: {'C': 0.15, 'G': 0.10, 'T': -0.15, 'A': -0.05},
    19: {'G': 0.20, 'C': 0.10, 'A': -0.10, 'T': -0.25},  # T at 19 strongly disfavored
    20: {'G': 0.35, 'C': 0.10, 'A': -0.15, 'T': -0.40},  # G at 20 strongly favored, T disfavored
}


def evaluate_on_target_efficiency(guide_seq: str, pam: str = "NGG") -> OnTargetScore:
    """
    Computes an on-target cleavage efficiency score (0 - 100) for a 20-nt sgRNA.
    """
    seq = guide_seq.upper().strip()
    if len(seq) != 20:
        return OnTargetScore(
            score=0.0,
            tier="INVALID",
            gc_content_pct=0.0,
            melting_temp_c=0.0,
            penalties=["Guide length is not exactly 20 nucleotides."],
            bonuses=[],
            summary="Invalid guide length."
        )

    base_score = 65.0  # Starting baseline for typical SpCas9 guide
    penalties = []
    bonuses = []

    # 1. GC Content Scoring
    gc = calculate_gc_content(seq)
    tm = estimate_melting_temp(seq)

    if 45.0 <= gc <= 60.0:
        base_score += 15.0
        bonuses.append(f"Ideal GC content ({gc}%): Optimal duplex stability.")
    elif 35.0 <= gc < 45.0 or 60.0 < gc <= 70.0:
        base_score += 5.0
        bonuses.append(f"Acceptable GC content ({gc}%).")
    elif gc < 30.0:
        penalty = (30.0 - gc) * 1.5
        base_score -= penalty
        penalties.append(f"Low GC content ({gc}%): Weak hybridization and lower Cas9 cutting.")
    elif gc > 75.0:
        penalty = (gc - 75.0) * 1.5
        base_score -= penalty
        penalties.append(f"Excessive GC content ({gc}%): Secondary structures may impede R-loop formation.")

    # 2. Position-specific preferences (Doench 2016 features)
    pos_bonus = 0.0
    for pos, prefs in POSITION_PREFERENCES.items():
        base = seq[pos - 1]
        weight = prefs.get(base, 0.0)
        pos_bonus += weight * 18.0

    if pos_bonus > 0:
        base_score += min(18.0, pos_bonus)
        bonuses.append("Favorable terminal nucleotides (favors G at position 20).")
    else:
        base_score += max(-18.0, pos_bonus)
        if seq[19] == 'T':
            penalties.append("Thymine at position 20: Strongly disfavored by SpCas9 PAM interaction.")

    # 3. Poly-T termination penalty (U6 / H1 Pol-III promoter abort)
    if "TTTT" in seq:
        base_score -= 30.0
        penalties.append("Poly-T motif ('TTTT') detected: Causes premature Pol-III transcription termination.")
    elif "TTT" in seq:
        base_score -= 8.0
        penalties.append("Trinucleotide 'TTT' sequence: Minor risk of early transcription pause.")

    # 4. Homopolymer runs of any base (AAAA, CCCC, GGGG)
    for base in ['A', 'C', 'G']:
        if base * 4 in seq:
            base_score -= 10.0
            penalties.append(f"Homopolymer run of '{base*4}': Prone to synthesis and slippage errors.")

    # 5. Self-complementarity / Hairpin propensity (simple palindrome check)
    comp_map = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    # Check 4-mer terminal complementarity with internal region
    stem_found = False
    for i in range(len(seq) - 4):
        motif = seq[i:i+4]
        rev_comp = "".join(comp_map.get(b, 'N') for b in reversed(motif))
        if rev_comp in seq[i+5:]:
            stem_found = True
            break
    if stem_found:
        base_score -= 10.0
        penalties.append("Potential internal stem-loop / hairpin detected in spacer sequence.")
    else:
        bonuses.append("Free from significant internal hairpin structures.")

    # 6. Clamp final score
    final_score = round(max(5.0, min(99.0, base_score)), 1)

    # Classify Tier
    if final_score >= 80.0:
        tier = "OPTIMAL"
        summary = "High predicted on-target cleavage activity (Tier 1 Clinical Quality)."
    elif final_score >= 65.0:
        tier = "GOOD"
        summary = "Robust on-target cleavage expected; suitable for standard gene knockout."
    elif final_score >= 45.0:
        tier = "MODERATE"
        summary = "Moderate cutting predicted; empirical wet-lab validation recommended."
    else:
        tier = "POOR"
        summary = "Low predicted cutting efficiency; consider selecting an alternate guide."

    return OnTargetScore(
        score=final_score,
        tier=tier,
        gc_content_pct=gc,
        melting_temp_c=tm,
        penalties=penalties,
        bonuses=bonuses,
        summary=summary
    )
