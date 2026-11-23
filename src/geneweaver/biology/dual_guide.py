"""
Dual-Guide Excision & Cas9 Nickase (D10A) Simulation Engine.
Models large genomic deletions, exon skipping (e.g. DMD Exon 51), viral provirus excision,
dual-guide recombination kinetics (deletion vs inversion), paired nickase offsets,
and virtual agarose gel electrophoresis amplicon migration.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from geneweaver.biology.on_target import evaluate_on_target_efficiency
from geneweaver.biology.presets import calculate_gc_content, estimate_melting_temp


@dataclass
class DualGuidePair:
    """Specification of two coordinating CRISPR guides."""
    guide_a_seq: str
    guide_b_seq: str
    cut_pos_a: int              # Global or relative chromosome coordinate of cut A
    cut_pos_b: int              # Global or relative chromosome coordinate of cut B
    strand_a: str = "(+)"
    strand_b: str = "(+)"
    pam_a: str = "NGG"
    pam_b: str = "NGG"
    label: str = "Targeted Deletion"


@dataclass
class VirtualGelBand:
    """Represents a fluorescing DNA band on an agarose gel."""
    name: str
    size_bp: int
    relative_migration_rf: float  # 0.0 (well at top) to 1.0 (bottom of gel)
    intensity_pct: float          # Fluorescing brightness (proportional to molarity * mass)
    band_color: str               # Hex code for visualization (e.g. #00F0FF, #00E676)


@dataclass
class VirtualGelLane:
    """Represents an electrophoretic lane on an agarose gel."""
    lane_number: int
    sample_label: str
    bands: List[VirtualGelBand]


@dataclass
class ExcisionSimulationResult:
    """Comprehensive biophysical outcome of dual-guide cleavage."""
    deletion_span_bp: int
    is_nickase_mode: bool
    nickase_offset_bp: int
    nickase_orientation: str       # 'PAM_OUT', 'PAM_IN', 'SAME_STRAND', 'STANDARD_DSB'
    predicted_deletion_efficiency_pct: float
    predicted_inversion_rate_pct: float
    predicted_unedited_or_indel_rate_pct: float
    guide_a_on_target_score: float
    guide_b_on_target_score: float
    composite_cutting_potency: float
    off_target_safety_advantage: str
    wildtype_amplicon_bp: int
    deleted_amplicon_bp: int
    virtual_gel_lanes: List[VirtualGelLane]
    deletion_junction_sequence: str
    clinical_recommendation: str
    biophysical_warnings: List[str]


# Standard Molecular DNA Ladders for Gel Electrophoresis
DNA_LADDER_1KB = [10000, 8000, 6000, 5000, 4000, 3000, 2000, 1500, 1000, 750, 500, 250]
DNA_LADDER_100BP = [1500, 1200, 1000, 900, 800, 700, 600, 500, 400, 300, 200, 100]

# Clinical Benchmarks for Exon Skipping & Excision
CLINICAL_EXCISION_PRESETS = {
    "dmd_exon_51_skipping": {
        "title": "Duchenne Muscular Dystrophy (DMD Exon 51 Skipping)",
        "gene": "DMD",
        "disease": "Duchenne Muscular Dystrophy",
        "guide_a": "GGTTAATCCTTCAGCACATC",  # Intron 50 splice acceptor target
        "guide_b": "AGTTTTAATTTCATTCATTT",  # Intron 51 splice donor target
        "cut_a": 1000,
        "cut_b": 1580,
        "deletion_bp": 580,
        "strand_a": "(+)",
        "strand_b": "(-)",
        "rationale": "Restores the reading frame of dystrophin to convert severe Duchenne phenotype into mild Becker phenotype."
    },
    "egfr_exon_19_deletion": {
        "title": "Non-Small Cell Lung Cancer (EGFR Exon 19 Deletion Model)",
        "gene": "EGFR",
        "disease": "Non-Small Cell Lung Carcinoma",
        "guide_a": "GTCCATGCCCTTCGGCTGCC",
        "guide_b": "CAAGGAATTAAGAGAAGCAA",
        "cut_a": 500,
        "cut_b": 725,
        "deletion_bp": 225,
        "strand_a": "(+)",
        "strand_b": "(+)",
        "rationale": "Excises oncogenic activating in-frame microdeletions in tyrosine kinase domain."
    },
    "hiv1_proviral_ltr_excision": {
        "title": "HIV-1 Integrated Provirus Dual-LTR Excision",
        "gene": "HIV-1 LTR",
        "disease": "Latent HIV-1 Infection",
        "guide_a": "GCCTTGAGTGCTTCAAGTAG",  # 5' LTR U3 region
        "guide_b": "CCTTGAGTGCTTCAAGTAGTG",  # 3' LTR U3 region
        "cut_a": 200,
        "cut_b": 9890,
        "deletion_bp": 9690,
        "strand_a": "(+)",
        "strand_b": "(+)",
        "rationale": "Completely excises full-length 9.7 kb integrated proviral genome from host CD4+ T cell reservoir."
    }
}


def calculate_migration_rf(size_bp: int, max_size: int = 10000, min_size: int = 100) -> float:
    """
    Computes relative electrophoretic mobility (Rf) in an agarose gel matrix.
    Mobility is inversely proportional to log10(molecular weight).
    Returns Rf from 0.05 (near well) to 0.95 (front of gel).
    """
    size_clamped = max(min_size, min(max_size, size_bp))
    log_max = math.log10(max_size)
    log_min = math.log10(min_size)
    log_val = math.log10(size_clamped)

    # Invert so smaller fragments migrate further (higher Rf)
    rf = (log_max - log_val) / (log_max - log_min)
    return round(0.08 + rf * 0.84, 3)


def simulate_dual_guide_excision(
    guide_a_seq: str,
    guide_b_seq: str,
    cut_pos_a: int,
    cut_pos_b: int,
    is_nickase_mode: bool = False,
    strand_a: str = "(+)",
    strand_b: str = "(-)",
    pam_a: str = "NGG",
    pam_b: str = "NGG",
    external_amplicon_padding_bp: int = 300
) -> ExcisionSimulationResult:
    """
    Simulates dual-guide excision efficiency, inversion probability,
    nickase orientation constraints, and virtual agarose gel electrophoresis bands.
    """
    pos_1 = min(cut_pos_a, cut_pos_b)
    pos_2 = max(cut_pos_a, cut_pos_b)
    span = pos_2 - pos_1

    # On-target efficiency of individual guides
    ot_a = evaluate_on_target_efficiency(guide_a_seq[:20], pam_a)
    ot_b = evaluate_on_target_efficiency(guide_b_seq[:20], pam_b)
    composite_potency = round((ot_a.score * ot_b.score) / 100.0, 1)

    warnings: List[str] = []

    # 1. Nickase Architecture Analysis (Cas9 D10A)
    nickase_offset = span
    if is_nickase_mode:
        # Determine paired nickase orientation
        if strand_a != strand_b:
            if strand_a == "(+)" and strand_b == "(-)":
                orientation = "PAM_OUT"  # Optimal for 5' overhangs
                safety_adv = "PAM-Out orientation generates 5' overhangs that stimulate high-fidelity non-homologous re-ligation."
            else:
                orientation = "PAM_IN"   # Generates 3' overhangs (lower efficiency)
                safety_adv = "PAM-In orientation produces 3' overhangs with moderate editing kinetics."
                warnings.append("PAM-In orientation observed; PAM-Out typically yields higher excision rates.")
        else:
            orientation = "SAME_STRAND"
            safety_adv = "Nicking on the same strand creates single-strand gaps rather than double-strand break."
            warnings.append("Both guides target the same DNA strand. Dual nickase requires opposite strands to induce a DSB!")
    else:
        orientation = "STANDARD_DSB"
        safety_adv = "Dual wild-type Cas9 induces two simultaneous double-strand breaks for robust physical excision."

    # 2. Recombination Kinetics (Deletion vs Inversion)
    # Deletion efficiency decays with distance; inversions occur at lower frequency
    if span <= 100:
        base_del_eff = 82.0
        inv_rate = 3.0
    elif span <= 1000:
        # Decay gently for sub-kilobase
        decay = math.log10(span / 100.0) * 12.0
        base_del_eff = max(45.0, 82.0 - decay)
        inv_rate = min(15.0, 3.0 + (span / 200.0))
    elif span <= 10000:
        # Multi-kilobase drop
        decay = 25.0 + math.log10(span / 1000.0) * 20.0
        base_del_eff = max(20.0, 75.0 - decay)
        inv_rate = min(22.0, 10.0 + (span / 1000.0) * 1.2)
    else:
        # Ultra-large >10 kb excision
        base_del_eff = 18.0
        inv_rate = 14.0
        warnings.append(f"Very large excision span ({span:,} bp). Cellular re-ligation frequency is attenuated.")

    # Modulate by composite guide on-target potency
    potency_factor = max(0.55, 0.4 + (composite_potency / 150.0))
    del_eff = round(min(95.0, base_del_eff * potency_factor), 1)
    inv_eff = round(min(30.0, inv_rate * (potency_factor * 0.8)), 1)
    unedited_eff = round(max(5.0, 100.0 - (del_eff + inv_eff)), 1)

    # 3. Virtual Gel Electrophoresis Simulation
    wt_amplicon = span + (external_amplicon_padding_bp * 2)
    del_amplicon = external_amplicon_padding_bp * 2

    ladder_bands = [
        VirtualGelBand(
            name=f"{size} bp",
            size_bp=size,
            relative_migration_rf=calculate_migration_rf(size),
            intensity_pct=100.0 if size in [1000, 3000] else 65.0,  # Reference bands brighter
            band_color="#FFFFFF"
        )
        for size in DNA_LADDER_1KB
    ]

    ctrl_bands = [
        VirtualGelBand(
            name=f"Wildtype ({wt_amplicon} bp)",
            size_bp=wt_amplicon,
            relative_migration_rf=calculate_migration_rf(wt_amplicon),
            intensity_pct=95.0,
            band_color="#00F0FF"
        )
    ]

    edited_bands = [
        VirtualGelBand(
            name=f"Excised Allele ({del_amplicon} bp)",
            size_bp=del_amplicon,
            relative_migration_rf=calculate_migration_rf(del_amplicon),
            intensity_pct=del_eff,
            band_color="#00E676"
        ),
        VirtualGelBand(
            name=f"Unexcised / WT ({wt_amplicon} bp)",
            size_bp=wt_amplicon,
            relative_migration_rf=calculate_migration_rf(wt_amplicon),
            intensity_pct=unedited_eff,
            band_color="#8B949E"
        )
    ]

    virtual_lanes = [
        VirtualGelLane(lane_number=1, sample_label="Ladder (1 kb)", bands=ladder_bands),
        VirtualGelLane(lane_number=2, sample_label="Untreated Control", bands=ctrl_bands),
        VirtualGelLane(lane_number=3, sample_label="Dual-Guide Edited Sample", bands=edited_bands)
    ]

    # 4. Deletion Junction Model
    junction_seq = f"{guide_a_seq[:10]}[--- {span} bp DELETED ---]{guide_b_seq[-10:]}"

    # 5. Recommendation
    if is_nickase_mode:
        rec = f"Paired Cas9 D10A nickases with {orientation} geometry: Eliminates off-target double-strand break risk while excising {span} bp."
    else:
        rec = f"Dual Cas9 nuclease cutting: Highly robust for physical fragment excision ({del_eff}% predicted deletion rate)."

    return ExcisionSimulationResult(
        deletion_span_bp=span,
        is_nickase_mode=is_nickase_mode,
        nickase_offset_bp=nickase_offset,
        nickase_orientation=orientation,
        predicted_deletion_efficiency_pct=del_eff,
        predicted_inversion_rate_pct=inv_eff,
        predicted_unedited_or_indel_rate_pct=unedited_eff,
        guide_a_on_target_score=ot_a.score,
        guide_b_on_target_score=ot_b.score,
        composite_cutting_potency=composite_potency,
        off_target_safety_advantage=safety_adv,
        wildtype_amplicon_bp=wt_amplicon,
        deleted_amplicon_bp=del_amplicon,
        virtual_gel_lanes=virtual_lanes,
        deletion_junction_sequence=junction_seq,
        clinical_recommendation=rec,
        biophysical_warnings=warnings
    )


def design_excision_junction_primers(
    chrom: str,
    cut_pos_a: int,
    cut_pos_b: int,
    upstream_flank_dna: str = "GACGTACTGACGTACGTACTGACGTACTGACGTACGTACT",
    downstream_flank_dna: str = "TGCATGCACTGCATGCACTGTGCATGCACTGCATGCACTG"
) -> Dict[str, str]:
    """
    Designs PCR primers to amplify specifically across the deletion scar.
    """
    fwd_primer = upstream_flank_dna[:20].upper()
    # Reverse primer is reverse complement of downstream flank
    tr = str.maketrans("ACGT", "TGCA")
    rev_primer = downstream_flank_dna[-20:].translate(tr)[::-1].upper()

    fwd_tm = estimate_melting_temp(fwd_primer)
    rev_tm = estimate_melting_temp(rev_primer)
    fwd_gc = calculate_gc_content(fwd_primer)
    rev_gc = calculate_gc_content(rev_primer)

    return {
        "chromosome": chrom,
        "forward_primer_5to3": fwd_primer,
        "forward_tm": fwd_tm,
        "forward_gc": fwd_gc,
        "reverse_primer_5to3": rev_primer,
        "reverse_tm": rev_tm,
        "reverse_gc": rev_gc,
        "cut_a_coordinate": min(cut_pos_a, cut_pos_b),
        "cut_b_coordinate": max(cut_pos_a, cut_pos_b),
        "target_excision_span_bp": abs(cut_pos_b - cut_pos_a)
    }
