"""
Unit tests for Patient-Personalized Genomic Variant Analyzer (ClinVar / dbSNP / VCF).
"""

import pytest
from geneweaver.biology.variant_analyzer import (
    GenomicVariant,
    VariantConsequence,
    VariantImpactReport,
    parse_vcf_lines,
    matches_pam,
    analyze_variant_crispr_impact,
    compute_allele_specific_window,
    audit_patient_variants,
    get_clinvar_benchmark_variants,
)


def test_vcf_parser():
    sample_vcf = """##fileformat=VCFv4.2
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tPATIENT1
chr11\t5227002\trs334\tA\tT\t.\tPASS\tGENE=HBB;CLNSIG=Pathogenic\tGT\t0/1
chr4\t3074876\trs362307\tC\tT\t.\tPASS\tGENE=HTT;CLNSIG=Pathogenic\tGT\t1/1
"""
    variants = parse_vcf_lines(sample_vcf)
    assert len(variants) == 2

    v1 = variants[0]
    assert v1.chrom == "chr11"
    assert v1.pos == 5227002
    assert v1.variant_id == "rs334"
    assert v1.ref == "A"
    assert v1.alt == "T"
    assert v1.gene_symbol == "HBB"
    assert v1.clinical_significance == "Pathogenic"
    assert v1.heterozygous is True

    v2 = variants[1]
    assert v2.chrom == "chr4"
    assert v2.pos == 3074876
    assert v2.variant_id == "rs362307"
    assert v2.heterozygous is False  # 1/1 homozygote


def test_pam_destruction_detection():
    # Target locus: protospacer nt 1000..1019, PAM nt 1020..1022 (e.g. AGG)
    guide = "ATCGATCGATCGATCGATCG"
    pam = "AGG"
    # Variant at 1021: G -> A turns AGG into AAG
    destroyed_pam_var = GenomicVariant(
        chrom="chr1",
        pos=1021,
        ref="G",
        alt="A",
        variant_id="rs_pam_destroy",
        gene_symbol="TEST"
    )

    report = analyze_variant_crispr_impact(
        guide_seq=guide,
        pam_motif=pam,
        variant=destroyed_pam_var,
        target_locus_start=1000
    )

    assert report.consequence == VariantConsequence.PAM_DESTROYED
    assert report.is_pam_impacted is True
    assert report.alt_affinity_score < 5.0
    assert "THERAPY INEFFECTIVE" in report.recommendation


def test_seed_region_mismatch_penalty():
    guide = "ATCGATCGATCGATCGATCG"  # 20 nt
    pam = "CGG"
    # Position 1016 corresponds to position 17 of 20 (Seed Region)
    seed_var = GenomicVariant(
        chrom="chr1",
        pos=1016,
        ref="C",
        alt="T",
        variant_id="rs_seed_snp"
    )

    report = analyze_variant_crispr_impact(
        guide_seq=guide,
        pam_motif=pam,
        variant=seed_var,
        target_locus_start=1000
    )

    assert report.consequence == VariantConsequence.SEED_DISRUPTED
    assert report.is_seed_impacted is True
    assert report.position_in_guide == 17
    assert report.alt_affinity_score < 10.0  # Seed position 17 retention is ~2%


def test_distal_mismatch_tolerance():
    guide = "ATCGATCGATCGATCGATCG"
    pam = "CGG"
    # Position 1002 corresponds to position 3 of 20 (PAM-distal 5' end)
    distal_var = GenomicVariant(
        chrom="chr1",
        pos=1002,
        ref="T",
        alt="G",
        variant_id="rs_distal_snp"
    )

    report = analyze_variant_crispr_impact(
        guide_seq=guide,
        pam_motif=pam,
        variant=distal_var,
        target_locus_start=1000
    )

    assert report.consequence == VariantConsequence.DISTAL_MISMATCH
    assert report.is_seed_impacted is False
    assert report.position_in_guide == 3
    assert report.alt_affinity_score > 60.0  # Distal tolerance is high


def test_allele_specific_selectivity():
    res = compute_allele_specific_window(
        wildtype_seq="GTCCTGCAGGCTCCACGGGT",
        mutant_seq="GTCCTGCAGGTTCCACGGGT",
        snp_pos_in_window=15
    )
    assert res["on_target_mutant_cut_rate"] > 80.0
    assert res["off_target_wildtype_cut_rate"] < 10.0
    assert res["allele_selectivity_ratio"] >= 10.0
    assert res["is_clinically_selective"] is True


def test_patient_audit_dossier():
    guide = "CTAACAGGCAGACTTCTCCA"
    pam = "AGG"
    # Two variants: one pam destroyer, one neutral outside
    vars_list = [
        GenomicVariant(chrom="chr11", pos=1021, ref="G", alt="T", variant_id="var_pam"),
        GenomicVariant(chrom="chr11", pos=900, ref="A", alt="G", variant_id="var_outside"),
    ]

    dossier = audit_patient_variants(
        guide_seq=guide,
        pam_motif=pam,
        variants=vars_list,
        patient_id="PT-SICKLE-01",
        target_locus_start=1000
    )

    assert dossier.patient_id == "PT-SICKLE-01"
    assert dossier.num_variants_screened == 2
    assert len(dossier.impacted_variants) == 1
    assert dossier.therapy_viable_in_patient is False
    assert dossier.risk_level == "THERAPY_CONTRAINDICATED"


def test_clinvar_benchmarks():
    benchmarks = get_clinvar_benchmark_variants()
    assert "HBB_rs334_SickleCell" in benchmarks
    assert "HTT_rs362307_Huntington" in benchmarks
    assert "BRAF_rs113488022_V600E" in benchmarks
    assert "EGFR_rs121434568_T790M" in benchmarks
