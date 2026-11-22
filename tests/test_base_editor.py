"""
Unit test suite for Base Editing (CBE / ABE) & Prime Editing (pegRNA) module.
"""

import pytest
from geneweaver.biology.base_editor import (
    get_supported_base_editors,
    simulate_base_editing,
    design_pegrna,
    BASE_EDITOR_REGISTRY
)


def test_supported_base_editors():
    editors = get_supported_base_editors()
    assert "be4max" in editors
    assert "ancbe4max" in editors
    assert "abe8e" in editors
    assert "abe7_10" in editors

    be4 = editors["be4max"]
    assert be4.editor_type == "CBE"
    assert be4.target_base == "C"
    assert be4.product_base == "T"
    assert be4.deamination_window == (4, 8)


def test_simulate_cbe_clean_edit():
    # Protospacer with only 1 'C' in the editing window (pos 4-8):
    # pos 1-3: GAG, pos 4-8: T C T T T (pos 5 is 'C', pos 4,6,7,8 are not C)
    proto = "GAGTCTTTGAGAAGAAGAAG"
    res = simulate_base_editing(proto, target_pos=5, editor_id="be4max")

    assert res.target_position == 5
    assert res.target_base == "C"
    assert res.converted_base == "T"
    assert res.is_target_in_window is True
    assert res.bystander_count == 0
    assert res.bystander_risk_tier == "CLEAN_EDIT_NO_BYSTANDERS"
    assert res.product_purity_score_pct >= 90.0
    assert res.edited_protospacer[4] == "T"


def test_simulate_cbe_bystander_edit():
    # Protospacer with multiple C's in window: pos 4 is C, pos 6 is C
    # pos 1-3: GAG, pos 4: C, pos 5: T, pos 6: C, pos 7: A, pos 8: G
    proto = "GAGCTCGGTAGAAGAAGAAG"
    res = simulate_base_editing(proto, target_pos=4, editor_id="be4max")

    assert res.target_position == 4
    assert res.is_target_in_window is True
    assert res.bystander_count >= 1
    assert res.bystander_risk_tier in ["MODERATE_BYSTANDER_RISK", "HIGH_BYSTANDER_RISK"]
    assert len(res.all_possible_alleles) >= 2


def test_simulate_abe_editing():
    # ABE8e converts A -> G
    # Let target be pos 6
    proto = "GAGTCAGCAGAAGAAGAAGA"
    res = simulate_base_editing(proto, target_pos=6, editor_id="abe8e")

    assert res.editor_type == "ABE"
    assert res.converted_base == "G"
    assert res.predicted_on_target_efficiency_pct > 70.0


def test_pegrna_design():
    spacer = "GAGTCCGAGCAGAAGAAGAA"
    design = design_pegrna(
        target_guide=spacer,
        edit_type="SUBSTITUTION",
        edit_offset_from_nick=4,
        desired_sequence="T",
        pbs_length=13
    )

    assert design.spacer_seq_20nt == spacer
    assert len(design.pbs_seq) == 13
    assert len(design.scaffold_seq) > 50
    assert design.composite_pegrna_seq.startswith(spacer)
    assert design.feasibility_score > 0
    assert design.feasibility_tier in ["OPTIMAL", "MODERATE", "CHALLENGING"]
    assert len(design.recommendations) > 0
