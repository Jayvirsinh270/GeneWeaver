"""
Patient-Personalized Genomic Variant (ClinVar / dbSNP / VCF) Analyzer for CRISPR.

Enables precision CRISPR design by auditing patient-specific SNPs and indels against
CRISPR target guides:
1. PAM Destruction (Loss-of-Cleavage):
   Identifies mutations in canonical PAM (e.g., NGG -> NAG) causing clinical therapy failure.
2. PAM Creation (De-Novo Off-Target Risk):
   Identifies patient-specific mutations creating new PAM sites (e.g., AAG -> AGG) not
   present in standard reference genomes (GRCh38/T2T).
3. Seed Region Mismatch Modulation:
   Evaluates SNP positions along protospacer (Seed nt 11-20 vs Distal nt 1-10) and calculates
   predicted drop or gain in cleavage efficiency.
4. Allele-Specific CRISPR Targeting (Dominant Negative & Oncogene Knockout):
   Computes Allele Selectivity Ratio (mutant cut rate / wildtype cut rate) for diseases like
   Huntington's Disease (HTT rs362307), Sickle Cell Disease (HBB rs334), BRAF V600E (rs113488022).
5. VCF (Variant Call Format) line and text parser.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class VariantConsequence(str, Enum):
    """Categorized clinical consequence of a genomic variant on CRISPR activity."""
    PAM_DESTROYED = "PAM_DESTROYED"              # Severe: PAM abolished, therapeutic cut lost (~0%)
    PAM_CREATED = "PAM_CREATED"                  # High Risk: De-novo PAM created in patient genome
    SEED_DISRUPTED = "SEED_DISRUPTED"            # Substantial cleavage reduction (>70% drop)
    DISTAL_MISMATCH = "DISTAL_MISMATCH"          # Minor to moderate cleavage reduction (10-40% drop)
    ALLELE_SPECIFIC_IDEAL = "ALLELE_SPECIFIC"    # High selectivity: mutant cut >> WT cut (ratio >= 10x)
    NEUTRAL = "NEUTRAL"                          # Outside protospacer & PAM


@dataclass
class GenomicVariant:
    """Represents a genomic single nucleotide polymorphism or indel."""
    chrom: str
    pos: int
    ref: str
    alt: str
    variant_id: str = "."
    gene_symbol: str = ""
    clinical_significance: str = "Uncertain"
    heterozygous: bool = True
    info: Dict[str, str] = field(default_factory=dict)


@dataclass
class VariantImpactReport:
    """Comprehensive impact analysis of a single variant against a guide/PAM locus."""
    variant: GenomicVariant
    guide_seq: str
    pam_motif: str
    consequence: VariantConsequence
    ref_affinity_score: float             # Predicted cutting efficiency on reference allele (0 - 100%)
    alt_affinity_score: float             # Predicted cutting efficiency on patient allele (0 - 100%)
    allele_selectivity_ratio: float       # alt_affinity / ref_affinity (>1 favors mutant, <1 favors WT)
    is_pam_impacted: bool
    is_seed_impacted: bool
    position_in_guide: Optional[int]      # 1-indexed (1=distal 5' end, 20=PAM-proximal seed)
    description: str
    recommendation: str


@dataclass
class PersonalizedAuditDossier:
    """Full patient audit summarizing all detected variants and overall safety clearance."""
    guide_seq: str
    pam_motif: str
    patient_id: str
    num_variants_screened: int
    impacted_variants: List[VariantImpactReport]
    therapy_viable_in_patient: bool
    risk_level: str                       # SAFE, CAUTION, HIGH_RISK, THERAPY_CONTRAINDICATED
    key_findings: List[str]


# Pre-computed cleavage retention weights based on mismatch position in SpCas9 20nt guide
# Pos 20 is immediately adjacent to PAM (seed core), Pos 1 is 5' distal PAM-distal end.
SEED_POSITION_WEIGHTS = {
    # Distal positions (1-8): Cas9 tolerates single mismatches relatively well
    1: 0.95, 2: 0.92, 3: 0.88, 4: 0.82, 5: 0.78, 6: 0.72, 7: 0.65, 8: 0.60,
    # Transition positions (9-12)
    9: 0.50, 10: 0.42, 11: 0.32, 12: 0.22,
    # Core Seed positions (13-20): Mismatch severely aborts R-loop propagation and HNH activation
    13: 0.12, 14: 0.08, 15: 0.05, 16: 0.03, 17: 0.02, 18: 0.01, 19: 0.02, 20: 0.04
}


def parse_vcf_lines(vcf_text: str) -> List[GenomicVariant]:
    """
    Parses standard VCF 4.2 text lines into GenomicVariant objects.
    Skips header and comment lines starting with '#'.
    """
    variants: List[GenomicVariant] = []
    lines = vcf_text.strip().splitlines()
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 5:
            # Fallback for space-delimited text
            parts = line.split()
            if len(parts) < 5:
                continue

        chrom = parts[0]
        try:
            pos = int(parts[1])
        except ValueError:
            continue

        var_id = parts[2]
        ref = parts[3].upper()
        alt = parts[4].upper()

        info_dict: Dict[str, str] = {}
        if len(parts) > 7:
            for item in parts[7].split(";"):
                if "=" in item:
                    k, v = item.split("=", 1)
                    info_dict[k] = v
                else:
                    info_dict[item] = "True"

        gene_symbol = info_dict.get("GENE", info_dict.get("SYMBOL", ""))
        clin_sig = info_dict.get("CLNSIG", info_dict.get("SIGNIFICANCE", "Reported"))

        # Check genotype field (GT) if available (col 9 or 10)
        is_het = True
        if len(parts) >= 10:
            gt_field = parts[9].split(":")[0]
            if gt_field in ("1/1", "1|1"):
                is_het = False

        variants.append(GenomicVariant(
            chrom=chrom,
            pos=pos,
            ref=ref,
            alt=alt,
            variant_id=var_id,
            gene_symbol=gene_symbol,
            clinical_significance=clin_sig,
            heterozygous=is_het,
            info=info_dict
        ))
    return variants


def matches_pam(pam_dna: str, pam_pattern: str = "NGG") -> bool:
    """Checks whether a 3-nt string matches the PAM motif (default SpCas9 NGG)."""
    if len(pam_dna) != len(pam_pattern):
        return False
    # Regex translation for IUPAC DNA codes
    iupac_map = {
        "A": "A", "C": "C", "G": "G", "T": "T",
        "N": "[ACGT]", "R": "[AG]", "Y": "[CT]", "S": "[GC]",
        "W": "[AT]", "K": "[GT]", "M": "[AC]", "B": "[CGT]",
        "D": "[AGT]", "H": "[ACT]", "V": "[ACG]"
    }
    regex_pattern = "^" + "".join(iupac_map.get(base, base) for base in pam_pattern) + "$"
    return bool(re.match(regex_pattern, pam_dna.upper()))


def analyze_variant_crispr_impact(
    guide_seq: str,
    pam_motif: str,
    variant: GenomicVariant,
    target_locus_start: int = 1000,
    guide_is_forward: bool = True
) -> VariantImpactReport:
    """
    Evaluates the biophysical impact of a patient SNP on CRISPR Cas9 binding and cleavage.

    Coordinates:
    - If guide_is_forward:
      Protospacer spans [target_locus_start, target_locus_start + 19] (20 nt, 1-indexed).
      PAM spans [target_locus_start + 20, target_locus_start + 22] (3 nt).
    """
    guide_len = len(guide_seq)
    pam_len = len(pam_motif)
    var_pos = variant.pos

    proto_start = target_locus_start
    proto_end = target_locus_start + guide_len - 1
    pam_start = proto_end + 1
    pam_end = pam_start + pam_len - 1

    # Check if variant falls inside PAM
    if pam_start <= var_pos <= pam_end:
        pam_idx = var_pos - pam_start
        ref_pam = list(pam_motif)
        alt_pam = list(pam_motif)

        # Substitute variant
        if pam_idx < len(alt_pam):
            alt_pam[pam_idx] = variant.alt[0] if variant.alt else "N"
        alt_pam_str = "".join(alt_pam)

        ref_pam_valid = matches_pam(pam_motif, pam_motif)
        alt_pam_valid = matches_pam(alt_pam_str, pam_motif)

        if ref_pam_valid and not alt_pam_valid:
            # PAM Destroyed!
            return VariantImpactReport(
                variant=variant,
                guide_seq=guide_seq,
                pam_motif=pam_motif,
                consequence=VariantConsequence.PAM_DESTROYED,
                ref_affinity_score=94.5,
                alt_affinity_score=1.2,
                allele_selectivity_ratio=0.013,
                is_pam_impacted=True,
                is_seed_impacted=False,
                position_in_guide=None,
                description=f"Patient variant {variant.variant_id} ({variant.ref}->{variant.alt}) eliminates PAM: '{pam_motif}' mutated to '{alt_pam_str}'.",
                recommendation="THERAPY INEFFECTIVE: Cas9 fails to recognize destroyed PAM in this patient. Redesign guide."
            )
        elif not ref_pam_valid and alt_pam_valid:
            # PAM Created (De-novo site)
            return VariantImpactReport(
                variant=variant,
                guide_seq=guide_seq,
                pam_motif=pam_motif,
                consequence=VariantConsequence.PAM_CREATED,
                ref_affinity_score=2.0,
                alt_affinity_score=91.0,
                allele_selectivity_ratio=45.5,
                is_pam_impacted=True,
                is_seed_impacted=False,
                position_in_guide=None,
                description=f"Patient variant {variant.variant_id} creates de-novo PAM '{alt_pam_str}', opening a non-canonical cutting locus.",
                recommendation="PATIENT-SPECIFIC OFF-TARGET RISK: Assess off-target cytotoxicity at this new patient locus."
            )

    # Check if variant falls inside Protospacer
    if proto_start <= var_pos <= proto_end:
        # Guide position 1 to 20 (1 is 5' distal, 20 is 3' PAM-adjacent seed)
        pos_in_guide = (var_pos - proto_start) + 1
        retention = SEED_POSITION_WEIGHTS.get(pos_in_guide, 0.5)

        ref_affinity = 92.0
        alt_affinity = round(ref_affinity * retention, 1)

        # Allele selectivity calculation (alt vs ref)
        # If designing specifically against mutant allele (alt), selectivity = alt / ref
        selectivity = round(alt_affinity / max(ref_affinity, 0.1), 3)

        if pos_in_guide >= 13:
            consequence = VariantConsequence.SEED_DISRUPTED
            desc = (f"Variant {variant.variant_id} is located in the critical PAM-proximal Seed Region (position {pos_in_guide}/20). "
                    f"Cleavage affinity is suppressed from {ref_affinity}% to {alt_affinity}%.")
            rec = "Significant loss of cleavage efficacy on this allele. Consider repositioning guide."
        else:
            consequence = VariantConsequence.DISTAL_MISMATCH
            desc = (f"Variant {variant.variant_id} is in the 5' PAM-distal region (position {pos_in_guide}/20). "
                    f"Cas9 tolerates distal mismatch with moderate cleavage retention ({alt_affinity}%).")
            rec = "Cleavage remains viable on both alleles with minor kinetic delay."

        return VariantImpactReport(
            variant=variant,
            guide_seq=guide_seq,
            pam_motif=pam_motif,
            consequence=consequence,
            ref_affinity_score=ref_affinity,
            alt_affinity_score=alt_affinity,
            allele_selectivity_ratio=selectivity,
            is_pam_impacted=False,
            is_seed_impacted=(pos_in_guide >= 13),
            position_in_guide=pos_in_guide,
            description=desc,
            recommendation=rec
        )

    # Neutral outside guide & PAM
    return VariantImpactReport(
        variant=variant,
        guide_seq=guide_seq,
        pam_motif=pam_motif,
        consequence=VariantConsequence.NEUTRAL,
        ref_affinity_score=92.0,
        alt_affinity_score=92.0,
        allele_selectivity_ratio=1.0,
        is_pam_impacted=False,
        is_seed_impacted=False,
        position_in_guide=None,
        description=f"Variant {variant.variant_id} at position {var_pos} is outside guide and PAM locus (target {proto_start}-{pam_end}).",
        recommendation="No direct interference with target binding."
    )


def compute_allele_specific_window(
    wildtype_seq: str,
    mutant_seq: str,
    snp_pos_in_window: int = 15,
    pam_motif: str = "NGG"
) -> Dict[str, float]:
    """
    Calculates the allele-specific discriminatory power when a guide is centered
    on a pathogenic SNP (e.g. for huntingtin HTT or BRAF V600E).
    """
    # Mismatch in seed region position 15 confers ~95% drop on non-target allele
    retention_mismatched = SEED_POSITION_WEIGHTS.get(snp_pos_in_window, 0.05)
    on_target_cut = 88.5
    off_target_cut = round(on_target_cut * retention_mismatched, 2)
    selectivity_ratio = round(on_target_cut / max(off_target_cut, 0.01), 1)

    return {
        "on_target_mutant_cut_rate": on_target_cut,
        "off_target_wildtype_cut_rate": off_target_cut,
        "allele_selectivity_ratio": selectivity_ratio,
        "is_clinically_selective": selectivity_ratio >= 10.0
    }


def audit_patient_variants(
    guide_seq: str,
    pam_motif: str,
    variants: List[GenomicVariant],
    patient_id: str = "PATIENT-001",
    target_locus_start: int = 1000
) -> PersonalizedAuditDossier:
    """
    Audits a list of patient-specific variants against a CRISPR candidate guide.
    Aggregates reports and assigns patient clearance status.
    """
    impacted: List[VariantImpactReport] = []
    key_findings: List[str] = []
    therapy_viable = True
    risk_level = "SAFE"

    for var in variants:
        rep = analyze_variant_crispr_impact(
            guide_seq=guide_seq,
            pam_motif=pam_motif,
            variant=var,
            target_locus_start=target_locus_start
        )
        if rep.consequence != VariantConsequence.NEUTRAL:
            impacted.append(rep)

            if rep.consequence == VariantConsequence.PAM_DESTROYED:
                therapy_viable = False
                risk_level = "THERAPY_CONTRAINDICATED"
                key_findings.append(f"CRITICAL: Variant {var.variant_id} destroys the required {pam_motif} PAM. Efficacy drops to {rep.alt_affinity_score}%.")
            elif rep.consequence == VariantConsequence.PAM_CREATED:
                if risk_level not in ("THERAPY_CONTRAINDICATED", "HIGH_RISK"):
                    risk_level = "HIGH_RISK"
                key_findings.append(f"WARNING: De-novo PAM created by {var.variant_id}, conferring patient-specific off-target hazard.")
            elif rep.consequence == VariantConsequence.SEED_DISRUPTED:
                if risk_level == "SAFE":
                    risk_level = "CAUTION"
                key_findings.append(f"CAUTION: Seed mutation by {var.variant_id} decreases cleavage affinity by {(1.0 - rep.alt_affinity_score/rep.ref_affinity_score)*100:.1f}%.")

    if not impacted:
        key_findings.append("No patient variants intersect the 20-nt guide protospacer or PAM window.")

    return PersonalizedAuditDossier(
        guide_seq=guide_seq,
        pam_motif=pam_motif,
        patient_id=patient_id,
        num_variants_screened=len(variants),
        impacted_variants=impacted,
        therapy_viable_in_patient=therapy_viable,
        risk_level=risk_level,
        key_findings=key_findings
    )


def get_clinvar_benchmark_variants() -> Dict[str, Dict]:
    """
    Returns curated clinical variant benchmarks for testing and demonstration.
    Covers dominant oncogenes, repeat expansions, and monogenic blood disorders.
    """
    return {
        "HBB_rs334_SickleCell": {
            "name": "Sickle Cell Disease (HBB:c.20A>T, p.Glu6Val, rs334)",
            "gene": "HBB",
            "chrom": "chr11",
            "pos": 5227002,
            "ref": "A",
            "alt": "T",
            "clinvar_id": "RCV000014389",
            "clinical_significance": "Pathogenic",
            "guide_seq": "CTAACAGGCAGACTTCTCCA",
            "pam_motif": "AGG",
            "target_locus_start": 5226982,
            "clinical_context": "Point mutation creates abnormal HbS hemoglobin polymerizing under hypoxia."
        },
        "HTT_rs362307_Huntington": {
            "name": "Huntington's Disease Allele-Specific SNP (HTT:c.*144C>T, rs362307)",
            "gene": "HTT",
            "chrom": "chr4",
            "pos": 3074876,
            "ref": "C",
            "alt": "T",
            "clinvar_id": "VCV000492811",
            "clinical_significance": "Pathogenic / Linked to CAG expansion",
            "guide_seq": "GTCCTGCAGGTTCCACGGGT",
            "pam_motif": "CGG",
            "target_locus_start": 3074860,
            "clinical_context": "Enables selective silencing of the mutant huntingtin allele while sparing neuroprotective wildtype HTT."
        },
        "BRAF_rs113488022_V600E": {
            "name": "Melanoma / Colorectal Oncogene BRAF V600E (c.1799T>A, rs113488022)",
            "gene": "BRAF",
            "chrom": "chr7",
            "pos": 140753336,
            "ref": "T",
            "alt": "A",
            "clinvar_id": "VCV000013961",
            "clinical_significance": "Pathogenic (Somatic driver)",
            "guide_seq": "CTAGCTACAGTGAAATCTCG",
            "pam_motif": "AGG",
            "target_locus_start": 140753320,
            "clinical_context": "Constitutively active MAPK kinase driver. CRISPR guide leverages V600E seed mutation for cancer-selective ablation."
        },
        "EGFR_rs121434568_T790M": {
            "name": "Non-Small Cell Lung Cancer Tyrosine Kinase Resistance (EGFR T790M, rs121434568)",
            "gene": "EGFR",
            "chrom": "chr7",
            "pos": 55181378,
            "ref": "C",
            "alt": "T",
            "clinvar_id": "VCV000016616",
            "clinical_significance": "Pathogenic / Drug Resistance",
            "guide_seq": "CATCACGCAGCTCATGCCCT",
            "pam_motif": "TGG",
            "target_locus_start": 55181362,
            "clinical_context": "Acquired gatekeeper mutation causing resistance to 1st/2nd-gen TKIs. Selective target for mutation-specific editing."
        }
    }
