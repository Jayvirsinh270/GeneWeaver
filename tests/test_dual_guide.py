"""
Unit test suite for Dual-Guide Excision & Cas9 Nickase (D10A) module.
"""

import pytest
from geneweaver.biology.dual_guide import (
    simulate_dual_guide_excision,
    calculate_migration_rf,
    design_excision_junction_primers,
    CLINICAL_EXCISION_PRESETS,
    DNA_LADDER_1KB
)


def test_dmd_dual_guide_excision():
    preset = CLINICAL_EXCISION_PRESETS["dmd_exon_51_skipping"]
    res = simulate_dual_guide_excision(
        guide_a_seq=preset["guide_a"],
        guide_b_seq=preset["guide_b"],
        cut_pos_a=preset["cut_a"],
        cut_pos_b=preset["cut_b"],
        is_nickase_mode=False
    )

    assert res.deletion_span_bp == 580
    assert res.predicted_deletion_efficiency_pct > 30.0
    assert res.predicted_inversion_rate_pct > 0.0
    assert res.wildtype_amplicon_bp > res.deleted_amplicon_bp
    assert len(res.virtual_gel_lanes) == 3
    assert "DMD" in preset["gene"]


def test_paired_nickase_pam_out():
    res = simulate_dual_guide_excision(
        guide_a_seq="GGTTAATCCTTCAGCACATC",
        guide_b_seq="AGTTTTAATTTCATTCATTT",
        cut_pos_a=100,
        cut_pos_b=140,
        is_nickase_mode=True,
        strand_a="(+)",
        strand_b="(-)"
    )

    assert res.is_nickase_mode is True
    assert res.nickase_orientation == "PAM_OUT"
    assert "PAM-Out" in res.off_target_safety_advantage


def test_paired_nickase_same_strand_warning():
    res = simulate_dual_guide_excision(
        guide_a_seq="GGTTAATCCTTCAGCACATC",
        guide_b_seq="AGTTTTAATTTCATTCATTT",
        cut_pos_a=100,
        cut_pos_b=140,
        is_nickase_mode=True,
        strand_a="(+)",
        strand_b="(+)"
    )

    assert res.nickase_orientation == "SAME_STRAND"
    assert len(res.biophysical_warnings) > 0


def test_migration_rf_ordering():
    rf_100bp = calculate_migration_rf(100)
    rf_500bp = calculate_migration_rf(500)
    rf_1000bp = calculate_migration_rf(1000)
    rf_5000bp = calculate_migration_rf(5000)

    # Smaller fragments migrate further (higher Rf)
    assert rf_100bp > rf_500bp > rf_1000bp > rf_5000bp


def test_junction_primer_design():
    primers = design_excision_junction_primers(
        chrom="chrX",
        cut_pos_a=1000,
        cut_pos_b=1580
    )

    assert primers["chromosome"] == "chrX"
    assert len(primers["forward_primer_5to3"]) == 20
    assert len(primers["reverse_primer_5to3"]) == 20
    assert primers["target_excision_span_bp"] == 580
    assert primers["forward_tm"] > 40.0
