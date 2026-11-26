"""
Unit tests for Epigenetic CRISPRa & CRISPRi Transcriptional Modulation Engine.
"""

import pytest
from geneweaver.biology.epigenetic import (
    ModulationModality,
    EffectorDomain,
    EffectorProfile,
    EFFECTOR_PROFILES,
    calculate_tss_window_factor,
    simulate_epigenetic_modulation,
    generate_promoter_window_profile,
    get_clinical_epigenetic_presets
)


def test_effector_profiles_registry():
    assert "dcas9_vpr" in EFFECTOR_PROFILES
    assert "dcas9_sam" in EFFECTOR_PROFILES
    assert "dcas9_krab" in EFFECTOR_PROFILES
    assert "dcas9_krab_mecp2" in EFFECTOR_PROFILES

    vpr = EFFECTOR_PROFILES["dcas9_vpr"]
    assert vpr.modality == ModulationModality.ACTIVATION
    assert vpr.max_fold_modulation > 10.0

    krab = EFFECTOR_PROFILES["dcas9_krab_mecp2"]
    assert krab.modality == ModulationModality.INTERFERENCE
    assert krab.max_fold_modulation < 0.1


def test_crispra_activation_simulation():
    # Guide placed at -125 bp (optimal peak for VPR)
    res = simulate_epigenetic_modulation(
        gene_symbol="HBG1",
        guide_seq="GGTCAACTTTCCTCTCACAC",
        distance_to_tss_bp=-125,
        effector_key="dcas9_vpr",
        guide_efficiency_score=90.0
    )
    assert res.modality == ModulationModality.ACTIVATION
    assert res.is_in_optimal_window is True
    assert res.predicted_fold_change > 15.0
    assert res.log2_fold_change > 3.0
    assert res.percent_modulation > 1000.0  # >1000% induction
    assert "Single-AAV" in res.delivery_feasibility or "Dual-AAV" in res.delivery_feasibility


def test_crispri_interference_simulation():
    # Guide placed at +100 bp (optimal peak for KRAB-MeCP2)
    res = simulate_epigenetic_modulation(
        gene_symbol="MYC",
        guide_seq="CGGCGCTTTTCCCACCAAGG",
        distance_to_tss_bp=100,
        effector_key="dcas9_krab_mecp2",
        guide_efficiency_score=88.0
    )
    assert res.modality == ModulationModality.INTERFERENCE
    assert res.is_in_optimal_window is True
    assert res.predicted_fold_change < 0.2  # Heavy knockdown
    assert res.percent_modulation < -80.0   # >80% reduction
    assert res.retained_expression_pct < 20.0


def test_out_of_window_attenuation():
    # Guide far upstream at -800 bp should attenuate sharply
    res_far = simulate_epigenetic_modulation(
        gene_symbol="TEST",
        guide_seq="ATCGATCGATCGATCGATCG",
        distance_to_tss_bp=-800,
        effector_key="dcas9_vpr"
    )
    assert res_far.is_in_optimal_window is False
    assert res_far.window_attenuation_factor < 0.15
    assert res_far.predicted_fold_change < 5.0  # Severely reduced compared to peak


def test_promoter_window_profile_curve():
    curve = generate_promoter_window_profile(
        gene_symbol="HBG1",
        effector_key="dcas9_vpr",
        step_bp=50
    )
    assert len(curve) > 10
    distances = [pt["distance_from_tss_bp"] for pt in curve]
    assert -600 in distances
    assert 500 in distances


def test_clinical_epigenetic_presets():
    presets = get_clinical_epigenetic_presets()
    assert "HBG1_Activation_SickleCell" in presets
    assert "MYC_Silencing_Oncology" in presets
    assert "FOXP3_Activation_Autoimmunity" in presets
    assert "KRAS_Repression_Pancreatic" in presets
