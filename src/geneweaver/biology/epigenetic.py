"""
Epigenetic CRISPRa & CRISPRi Transcriptional Modulation Engine.

Models non-cleaving catalytically dead Cas9 (dCas9: D10A + H840A mutations) fused to
potent transcriptional activators (VPR, SunTag, SAM) or repressors (KRAB, KRAB-MeCP2).
Enables precision tuning of endogenous gene expression (activation or silencing)
without creating double-stranded DNA breaks.

Mathematical Modeling:
1. TSS Distance Attenuation Curve:
   - CRISPRa Window: -400 bp to -50 bp relative to Transcription Start Site (TSS).
     Optimal peak: ~ -150 to -100 bp (nucleosome-depleted promoter core).
   - CRISPRi Window: -50 bp to +300 bp relative to TSS.
     Optimal peak: ~ +50 to +150 bp (steric occlusion of elongating Pol II).
2. Fold-Change Prediction:
   Computes predicted log2 fold change, fold induction, and percentage knockdown
   accounting for effector potency, chromatin accessibility, and guide efficiency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class ModulationModality(str, Enum):
    """Transcriptional modulation modality."""
    ACTIVATION = "CRISPRa"        # Up-regulation / induction of target gene
    INTERFERENCE = "CRISPRi"      # Down-regulation / silencing of target gene


class EffectorDomain(str, Enum):
    """Fused epigenetic activator or repressor catalytic domain."""
    # Activators (CRISPRa)
    VPR = "dCas9-VPR"                     # VP64-p65-Rta tripartite activator (High potency)
    SUNTAG = "dCas9-SunTag"               # Multi-peptide epitope array recruiting 10-24x VP64
    SAM = "dCas9-SAM"                     # Synergistic Activation Mediator (VP64 + MS2-p65-HSF1)
    # Repressors (CRISPRi)
    KRAB = "dCas9-KRAB"                   # Krüppel-associated box domain (H3K9me3 heterochromatin)
    KRAB_MECP2 = "dCas9-KRAB-MeCP2"       # Dual bipartite repressor for durable epigenetic silencing


@dataclass
class EffectorProfile:
    """Biophysical characteristics of an epigenetic fusion construct."""
    name: str
    modality: ModulationModality
    domain: EffectorDomain
    max_fold_modulation: float            # Maximum theoretical fold-change (e.g. 35x for VPR, 0.04x for KRAB-MeCP2)
    optimal_window_start_bp: int          # Relative to TSS (negative = upstream promoter, positive = downstream exon 1)
    optimal_window_end_bp: int
    peak_distance_bp: int
    epigenetic_mark: str                  # e.g., H3K27ac deposition or H3K9me3 heterochromatinization
    delivery_size_kb: float               # Coding sequence cargo size in kilobases (AAV packaging constraint)
    description: str


EFFECTOR_PROFILES: Dict[str, EffectorProfile] = {
    "dcas9_vpr": EffectorProfile(
        name="dCas9-VPR",
        modality=ModulationModality.ACTIVATION,
        domain=EffectorDomain.VPR,
        max_fold_modulation=35.0,
        optimal_window_start_bp=-400,
        optimal_window_end_bp=-50,
        peak_distance_bp=-125,
        epigenetic_mark="H3K27ac & Mediator recruitment",
        delivery_size_kb=5.2,
        description="Robust tripartite activator composed of VP64, p65, and Rta domains. Standard clinical benchmark."
    ),
    "dcas9_suntag": EffectorProfile(
        name="dCas9-SunTag",
        modality=ModulationModality.ACTIVATION,
        domain=EffectorDomain.SUNTAG,
        max_fold_modulation=60.0,
        optimal_window_start_bp=-350,
        optimal_window_end_bp=-50,
        peak_distance_bp=-150,
        epigenetic_mark="Multi-valent GCN4-VP64 transactivation",
        delivery_size_kb=6.8,
        description="Repeating peptide scaffold recruiting up to 24 VP64 activators for massive transactivation."
    ),
    "dcas9_sam": EffectorProfile(
        name="dCas9-SAM",
        modality=ModulationModality.ACTIVATION,
        domain=EffectorDomain.SAM,
        max_fold_modulation=85.0,
        optimal_window_start_bp=-450,
        optimal_window_end_bp=-50,
        peak_distance_bp=-175,
        epigenetic_mark="H3K4me3, H3K27ac & SWI/SNF chromatin remodeling",
        delivery_size_kb=5.8,
        description="Synergistic Activation Mediator incorporating engineered sgRNA MS2 loops recruiting p65 and HSF1."
    ),
    "dcas9_krab": EffectorProfile(
        name="dCas9-KRAB",
        modality=ModulationModality.INTERFERENCE,
        domain=EffectorDomain.KRAB,
        max_fold_modulation=0.08,             # ~92% knockdown (retained expression 8%)
        optimal_window_start_bp=-50,
        optimal_window_end_bp=300,
        peak_distance_bp=100,
        epigenetic_mark="H3K9me3 chromatin condensation via KAP1/HP1",
        delivery_size_kb=4.5,
        description="Classic repressive domain inducing localized heterochromatinization and RNA Pol II pausing."
    ),
    "dcas9_krab_mecp2": EffectorProfile(
        name="dCas9-KRAB-MeCP2",
        modality=ModulationModality.INTERFERENCE,
        domain=EffectorDomain.KRAB_MECP2,
        max_fold_modulation=0.02,             # ~98% knockdown (retained expression 2%)
        optimal_window_start_bp=-50,
        optimal_window_end_bp=350,
        peak_distance_bp=120,
        epigenetic_mark="Dual H3K9me3 + CpG histone deacetylation",
        delivery_size_kb=5.4,
        description="Advanced dual-repressor fusing KRAB with MeCP2 transcriptional repression domain for durable silencing."
    ),
}


@dataclass
class TranscriptionalModulationResult:
    """Comprehensive modeling output of epigenetic expression tuning."""
    gene_symbol: str
    modality: ModulationModality
    effector_name: str
    guide_seq: str
    distance_to_tss_bp: int               # Distance in bp from Transcription Start Site (TSS)
    is_in_optimal_window: bool
    window_attenuation_factor: float      # Distance factor between 0.0 and 1.0 (Gaussian bell curve)
    predicted_fold_change: float          # Relative expression ratio (>1 for activation, <1 for repression)
    log2_fold_change: float               # log2(predicted_fold_change)
    percent_modulation: float             # e.g., +3200% for 33x activation, or -95% for silencing
    retained_expression_pct: float        # Remaining baseline expression (100% = WT level)
    off_target_epigenetic_risk: str       # MINIMAL, MODERATE, ELEVATED
    delivery_feasibility: str             # AAV Single, AAV Dual, LNP / Electroporation
    clinical_recommendation: str
    biophysical_notes: List[str] = field(default_factory=list)


def calculate_tss_window_factor(distance_bp: int, profile: EffectorProfile) -> float:
    """
    Calculates the spatial attenuation factor (0.0 to 1.0) using an asymmetric
    Gaussian decay curve centered at the effector's optimal distance from the TSS.
    """
    peak = profile.peak_distance_bp
    w_start = profile.optimal_window_start_bp
    w_end = profile.optimal_window_end_bp

    # Check boundaries
    if distance_bp < w_start or distance_bp > w_end:
        # Distance outside canonical window experiences sharp exponential decay
        dist_from_nearest = min(abs(distance_bp - w_start), abs(distance_bp - w_end))
        return max(0.02, math.exp(-0.5 * ((dist_from_nearest / 80.0) ** 2)) * 0.15)

    # Inside optimal window: Gaussian peak around peak_distance_bp
    sigma = (w_end - w_start) / 3.2
    factor = math.exp(-0.5 * (((distance_bp - peak) / sigma) ** 2))
    return round(min(1.0, max(0.1, factor)), 3)


def simulate_epigenetic_modulation(
    gene_symbol: str,
    guide_seq: str,
    distance_to_tss_bp: int,
    effector_key: str = "dcas9_vpr",
    guide_efficiency_score: float = 85.0
) -> TranscriptionalModulationResult:
    """
    Simulates transcriptional modulation for a given guide targeting a promoter/TSS locus.
    """
    profile = EFFECTOR_PROFILES.get(effector_key, EFFECTOR_PROFILES["dcas9_vpr"])
    window_factor = calculate_tss_window_factor(distance_to_tss_bp, profile)
    guide_factor = min(1.0, max(0.2, guide_efficiency_score / 100.0))

    combined_potency = window_factor * guide_factor
    is_optimal = (profile.optimal_window_start_bp <= distance_to_tss_bp <= profile.optimal_window_end_bp)

    biophysical_notes = []

    if profile.modality == ModulationModality.ACTIVATION:
        # CRISPRa: baseline is 1.0, scales up to max_fold_modulation
        fold_change = 1.0 + (profile.max_fold_modulation - 1.0) * combined_potency
        fold_change = round(fold_change, 2)
        log2_fc = round(math.log2(max(0.01, fold_change)), 2)
        pct_mod = round((fold_change - 1.0) * 100.0, 1)
        retained_exp = round(fold_change * 100.0, 1)

        if distance_to_tss_bp > -50:
            biophysical_notes.append("CAUTION: Guide is positioned too close to or downstream of TSS. May cause steric clash with RNA Pol II initiation complex.")
        elif distance_to_tss_bp < -450:
            biophysical_notes.append("NOTICE: Guide is distant from core promoter; transactivation potency is attenuated by nucleosome barriers.")
        else:
            biophysical_notes.append(f"OPTIMAL: Position {distance_to_tss_bp} bp falls within the nucleosome-depleted promoter core for maximal transactivation.")

        rec = f"Predicted {fold_change}x transcriptional induction of {gene_symbol} via {profile.name} recruitment of {profile.epigenetic_mark}."

    else:
        # CRISPRi: baseline is 1.0, drops towards max_fold_modulation (e.g. 0.02)
        # Fraction of repression achieved
        repression_fraction = (1.0 - profile.max_fold_modulation) * combined_potency
        fold_change = round(max(0.01, 1.0 - repression_fraction), 3)
        log2_fc = round(math.log2(fold_change), 2)
        pct_mod = round(-(1.0 - fold_change) * 100.0, 1)
        retained_exp = round(fold_change * 100.0, 1)

        if distance_to_tss_bp < -100:
            biophysical_notes.append("NOTICE: Guide is upstream of TSS; silencing relies predominantly on long-range heterochromatin spreading.")
        elif distance_to_tss_bp > 300:
            biophysical_notes.append("CAUTION: Guide is far downstream in gene body; Pol II processivity may overcome the dCas9 roadblock.")
        else:
            biophysical_notes.append(f"OPTIMAL: Position +{distance_to_tss_bp} bp creates an impenetrable steric block against RNA Polymerase II elongation.")

        rec = f"Predicted {abs(pct_mod)}% silencing ({retained_exp}% residual expression) of {gene_symbol} via {profile.epigenetic_mark}."

    # Delivery feasibility assessment
    if profile.delivery_size_kb <= 4.7:
        delivery_feasibility = "Single-AAV Compatible (< 4.7 kb)"
    elif profile.delivery_size_kb <= 9.0:
        delivery_feasibility = "Dual-AAV Reconstitution or LNP mRNA"
    else:
        delivery_feasibility = "Lipid Nanoparticle (LNP) / Electroporation Only"

    # Off-target epigenetic risk
    if guide_efficiency_score >= 80:
        off_target_risk = "MINIMAL (High on-target binding specificity)"
    elif guide_efficiency_score >= 50:
        off_target_risk = "MODERATE (Monitor non-specific promoter binding)"
    else:
        off_target_risk = "ELEVATED (Low guide affinity may promote spurious chromatin marks)"

    return TranscriptionalModulationResult(
        gene_symbol=gene_symbol,
        modality=profile.modality,
        effector_name=profile.name,
        guide_seq=guide_seq,
        distance_to_tss_bp=distance_to_tss_bp,
        is_in_optimal_window=is_optimal,
        window_attenuation_factor=window_factor,
        predicted_fold_change=fold_change,
        log2_fold_change=log2_fc,
        percent_modulation=pct_mod,
        retained_expression_pct=retained_exp,
        off_target_epigenetic_risk=off_target_risk,
        delivery_feasibility=delivery_feasibility,
        clinical_recommendation=rec,
        biophysical_notes=biophysical_notes
    )


def generate_promoter_window_profile(
    gene_symbol: str,
    effector_key: str = "dcas9_vpr",
    step_bp: int = 25
) -> List[Dict]:
    """
    Generates a continuous predicted modulation curve across the promoter spectrum (-600 bp to +500 bp)
    for interactive Plotly visualization.
    """
    profile = EFFECTOR_PROFILES.get(effector_key, EFFECTOR_PROFILES["dcas9_vpr"])
    curve_points = []

    for d in range(-600, 501, step_bp):
        w_factor = calculate_tss_window_factor(d, profile)
        if profile.modality == ModulationModality.ACTIVATION:
            fc = 1.0 + (profile.max_fold_modulation - 1.0) * w_factor
        else:
            rep = (1.0 - profile.max_fold_modulation) * w_factor
            fc = max(0.01, 1.0 - rep)

        curve_points.append({
            "distance_from_tss_bp": d,
            "window_factor": round(w_factor, 3),
            "predicted_fold_change": round(fc, 2),
            "log2_fold_change": round(math.log2(max(0.01, fc)), 2),
            "in_optimal_window": (profile.optimal_window_start_bp <= d <= profile.optimal_window_end_bp)
        })

    return curve_points


def get_clinical_epigenetic_presets() -> Dict[str, Dict]:
    """
    Returns curated clinical benchmark presets for epigenetic activation and interference therapies.
    """
    return {
        "HBG1_Activation_SickleCell": {
            "name": "Fetal Hemoglobin (HBG1/2) Reactivation (Sickle Cell & β-Thalassemia)",
            "gene": "HBG1",
            "modality": "CRISPRa",
            "effector": "dcas9_vpr",
            "guide_seq": "GGTCAACTTTCCTCTCACAC",
            "pam_motif": "AGG",
            "distance_to_tss_bp": -115,
            "target_locus": "chr11:5249870",
            "clinical_mechanism": "Induces γ-globin expression to assemble functional HbF tetramers, overcoming adult HbS sickle polymerisation."
        },
        "MYC_Silencing_Oncology": {
            "name": "MYC Proto-Oncogene Epigenetic Silencing (Burkitt & Solid Tumors)",
            "gene": "MYC",
            "modality": "CRISPRi",
            "effector": "dcas9_krab_mecp2",
            "guide_seq": "CGGCGCTTTTCCCACCAAGG",
            "pam_motif": "GGG",
            "distance_to_tss_bp": 110,
            "target_locus": "chr8:127735500",
            "clinical_mechanism": "Deposits H3K9me3 repressive chromatin at MYC promoter, arresting oncogenic cell cycle amplification without DSB chromosomal translocations."
        },
        "FOXP3_Activation_Autoimmunity": {
            "name": "FOXP3 Transcription Factor Activation (Regulatory T-Cell Induction)",
            "gene": "FOXP3",
            "modality": "CRISPRa",
            "effector": "dcas9_sam",
            "guide_seq": "TCAGCGTGAGCAAGGAAGAG",
            "pam_motif": "CGG",
            "distance_to_tss_bp": -160,
            "target_locus": "chrX:49258200",
            "clinical_mechanism": "Drives stable CD4+ Treg polarization to suppress graft-versus-host disease and autoimmune flares."
        },
        "KRAS_Repression_Pancreatic": {
            "name": "KRAS Pancreatic & Colorectal Driver Repression",
            "gene": "KRAS",
            "modality": "CRISPRi",
            "effector": "dcas9_krab",
            "guide_seq": "GACTGAGTACAAACTGGTGG",
            "pam_motif": "TGG",
            "distance_to_tss_bp": 85,
            "target_locus": "chr12:25205200",
            "clinical_mechanism": "Blocks RNA Pol II transcription of KRAS transcripts, selectively starving mutant tumors of MAPK pathway signaling."
        }
    }
