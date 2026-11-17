"""
Unit Tests for Clinical Presets & Guide Diagnostics.
"""

import pytest
from geneweaver.biology.presets import (
    CLINICAL_PRESETS,
    calculate_gc_content,
    estimate_melting_temp
)

def test_clinical_presets_catalog():
    assert "sickle_cell_bcl11a" in CLINICAL_PRESETS
    casgevy = CLINICAL_PRESETS["sickle_cell_bcl11a"]
    assert len(casgevy.guide_seq_20nt) == 20
    assert casgevy.pam_motif == "AGG"
    assert "FDA Approved" in casgevy.clinical_phase

def test_gc_content_calculation():
    # 10 G/C out of 20 bases = 50.0%
    seq = "GCGCGCGCGCATATATATAT"
    assert calculate_gc_content(seq) == 50.0

    # All A/T = 0.0%
    assert calculate_gc_content("ATATATATATATATATATAT") == 0.0

def test_melting_temperature():
    seq = "GAGTCCGAGCAGAAGAAGAA"
    tm = estimate_melting_temp(seq)
    assert 50.0 < tm < 75.0
