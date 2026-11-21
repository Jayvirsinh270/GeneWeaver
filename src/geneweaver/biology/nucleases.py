"""
Universal Multi-Nuclease Architecture & Biophysical Registry for CRISPR Gene Editing.
Supports SpCas9, Cas12a (Cpf1), SaCas9, and CasMINI (Cas12f).
Models PAM orientations (5' vs 3'), guide lengths (20nt, 21nt, 23nt),
cut geometries (blunt vs staggered overhangs), and AAV vector packaging limits.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from geneweaver.biology.presets import calculate_gc_content, estimate_melting_temp


@dataclass
class NucleaseProfile:
    """Biophysical and clinical profile of a CRISPR endonuclease."""
    id: str
    name: str
    organism: str
    canonical_pam: str
    pam_orientation: str       # '3_PRIME' or '5_PRIME'
    guide_length_nt: int       # 20, 21, or 23 nt
    cut_geometry: str          # 'BLUNT' or 'STAGGERED_5NT_OVERHANG'
    cleavage_description: str  # e.g., 'Blunt double-strand break 3bp upstream of PAM'
    protein_size_aa: int       # Amino acid length
    aav_packageable: bool      # True if single AAV vector can package nuclease + sgRNA (<4.7kb)
    seed_region_range: Tuple[int, int]  # (start, end) 1-indexed
    ideal_delivery_vector: str
    description: str
    clinical_use_case: str


@dataclass
class NucleaseEvaluation:
    """Evaluation result of a guide RNA against a selected nuclease system."""
    nuclease_id: str
    is_valid_length: bool
    expected_length: int
    actual_length: int
    gc_pct: float
    melting_temp: float
    seed_stability_score: float
    aav_single_vector_fit: bool
    cut_type: str
    cleavage_mechanism: str
    warnings: List[str]
    advantages: List[str]


# Universal Multi-Nuclease Clinical Registry
NUCLEASE_REGISTRY: Dict[str, NucleaseProfile] = {
    "sp_cas9": NucleaseProfile(
        id="sp_cas9",
        name="SpCas9 (Standard)",
        organism="Streptococcus pyogenes",
        canonical_pam="NGG",
        pam_orientation="3_PRIME",
        guide_length_nt=20,
        cut_geometry="BLUNT",
        cleavage_description="Blunt double-stranded break located 3 bp upstream of PAM (between nt 17 and 18).",
        protein_size_aa=1368,
        aav_packageable=False,  # Too large for single AAV (4.1 kb cDNA leaves no room for promoters/sgRNA)
        seed_region_range=(11, 20),
        ideal_delivery_vector="Lipid Nanoparticles (LNP) or Ex-Vivo Electroporation",
        description="The gold-standard CRISPR endonuclease with maximum targeting frequency and widespread clinical validation (e.g. Casgevy).",
        clinical_use_case="Gene knockout via Non-Homologous End Joining (NHEJ) and enhancer disruption in ex-vivo hematopoietic cells."
    ),
    "cas12a_cpf1": NucleaseProfile(
        id="cas12a_cpf1",
        name="Cas12a / Cpf1 (Sticky Overhangs)",
        organism="Acidaminococcus sp. / Lachnospiraceae",
        canonical_pam="TTTV",
        pam_orientation="5_PRIME",
        guide_length_nt=23,
        cut_geometry="STAGGERED_5NT_OVERHANG",
        cleavage_description="Staggered 4-5 nt 5'-cohesive overhang cleavage distal to 5' PAM (cuts at nt 18 on non-target and nt 23 on target).",
        protein_size_aa=1307,
        aav_packageable=False,
        seed_region_range=(1, 6),  # 5'-proximal seed
        ideal_delivery_vector="Electroporation RNP or Engineered LNP",
        description="Single-RNA guided endonuclease requiring only a short crRNA (no tracrRNA needed). Targets AT-rich promoter and non-coding genomic regions.",
        clinical_use_case="Targeted therapeutic gene insertion (Knock-in) via Homology-Directed Repair (HDR) and multiplexed multi-gene excision."
    ),
    "sa_cas9": NucleaseProfile(
        id="sa_cas9",
        name="SaCas9 (AAV-Packageable)",
        organism="Staphylococcus aureus",
        canonical_pam="NNGRRT",
        pam_orientation="3_PRIME",
        guide_length_nt=21,
        cut_geometry="BLUNT",
        cleavage_description="Blunt double-stranded break located 3 bp upstream of PAM (between nt 18 and 19).",
        protein_size_aa=1053,  # Over 300 aa smaller than SpCas9!
        aav_packageable=True,  # 3.16 kb cDNA easily fits in single AAV capsid (<4.7 kb limit)
        seed_region_range=(12, 21),
        ideal_delivery_vector="Single Adeno-Associated Virus (AAV8 / AAV9)",
        description="Compact miniature Cas9 endonuclease specifically engineered to overcome the 4.7 kb packaging limit of single AAV gene therapy vectors.",
        clinical_use_case="In-vivo systemic gene therapies targeting CNS, cardiac myocardium, and skeletal muscle via single-capsid AAV delivery."
    ),
    "cas_mini": NucleaseProfile(
        id="cas_mini",
        name="CasMINI / Cas12f (Ultra-Compact)",
        organism="Synthetically Engineered Archaea",
        canonical_pam="TTTR",
        pam_orientation="5_PRIME",
        guide_length_nt=20,
        cut_geometry="STAGGERED_5NT_OVERHANG",
        cleavage_description="Staggered double-stranded break leaving 4-nt overhangs with an ultra-miniature ribonucleoprotein architecture.",
        protein_size_aa=529,  # Less than half the size of SpCas9!
        aav_packageable=True,  # Ultra-compact cDNA allows multiplex guides and heavy promoters in AAV
        seed_region_range=(1, 8),
        ideal_delivery_vector="All-In-One AAV or Extracellular Nanovesicle",
        description="Revolutionary ultra-compact RNA-guided nuclease (<half the size of Cas9), unlocking all-in-one AAV viral vectors with complex dual guides.",
        clinical_use_case="All-in-one viral AAV delivery and cellular therapies requiring multiplexed dual-promoter expression cassettes."
    )
}


def get_supported_nucleases() -> Dict[str, NucleaseProfile]:
    """Returns all supported clinical and bioinformatic nuclease systems."""
    return NUCLEASE_REGISTRY


def evaluate_nuclease_compatibility(
    nuclease_id: str,
    guide_seq: str,
    pam_motif: str = ""
) -> NucleaseEvaluation:
    """
    Evaluates an sgRNA against the specific biophysical rules of the selected nuclease system.
    """
    seq = guide_seq.upper().strip()
    profile = NUCLEASE_REGISTRY.get(nuclease_id, NUCLEASE_REGISTRY["sp_cas9"])

    is_valid_len = (len(seq) == profile.guide_length_nt)
    gc = calculate_gc_content(seq)
    tm = estimate_melting_temp(seq)

    warnings: List[str] = []
    advantages: List[str] = []

    # 1. Guide Length Validation
    if not is_valid_len:
        warnings.append(
            f"Guide length ({len(seq)} nt) does not match {profile.name} requirement ({profile.guide_length_nt} nt)."
        )

    # 2. GC and Stability Profile
    if profile.id == "cas12a_cpf1":
        # Cas12a thrives in AT-rich DNA (30-55% GC)
        if 25.0 <= gc <= 55.0:
            advantages.append(f"Ideal AT-rich GC content ({gc}%) matching Cas12a target preferences.")
        elif gc > 65.0:
            warnings.append(f"High GC content ({gc}%) may hinder Cas12a crRNA R-loop propagation.")
    else:
        # Standard Cas9 optimal 40-65%
        if 40.0 <= gc <= 65.0:
            advantages.append(f"Balanced GC content ({gc}%) for optimal duplex hybridisation.")
        elif gc < 30.0 or gc > 70.0:
            warnings.append(f"GC content ({gc}%) is outside optimal 40-65% window.")

    # 3. Nuclease-Specific Advantages & Warnings
    if profile.aav_packageable:
        advantages.append(
            f"Compact protein ({profile.protein_size_aa} aa): 100% compatible with single-vector AAV packaging (<4.7 kb)."
        )
    else:
        warnings.append(
            f"Large protein ({profile.protein_size_aa} aa): Requires dual-AAV vectors or non-viral LNP/electroporation."
        )

    if profile.cut_geometry == "STAGGERED_5NT_OVERHANG":
        advantages.append(
            "Staggered 5' overhangs provide cohesive sticky ends that stimulate Homology-Directed Repair (HDR) knock-in."
        )
    else:
        advantages.append(
            "Blunt cleavage facilitates high-efficiency indels via Non-Homologous End Joining (NHEJ) gene knockout."
        )

    # 4. Seed region calculation
    s_start, s_end = profile.seed_region_range
    if len(seq) >= s_end:
        seed_subseq = seq[s_start - 1:s_end]
        seed_gc = calculate_gc_content(seed_subseq)
        seed_score = round(min(100.0, seed_gc * 1.5 + 20.0), 1)
    else:
        seed_score = 50.0

    return NucleaseEvaluation(
        nuclease_id=profile.id,
        is_valid_length=is_valid_len,
        expected_length=profile.guide_length_nt,
        actual_length=len(seq),
        gc_pct=gc,
        melting_temp=tm,
        seed_stability_score=seed_score,
        aav_single_vector_fit=profile.aav_packageable,
        cut_type=profile.cut_geometry,
        cleavage_mechanism=profile.cleavage_description,
        warnings=warnings,
        advantages=advantages
    )
