"""
Unit Tests for Cutting Frequency Determination (CFD) Scoring.
"""

import pytest
from geneweaver.biology.cfd import calculate_cfd_score, classify_cfd_risk

def test_cfd_perfect_match():
    guide = "GAGTCCGAGCAGAAGAAGAA"
    score = calculate_cfd_score(guide, guide, pam="AGG")
    assert score == 1.0
    assert classify_cfd_risk(score) == "HIGH_RISK"

def test_cfd_pam_weighting():
    guide = "GAGTCCGAGCAGAAGAAGAA"
    # Canonical NGG PAM (score = 1.0)
    score_ngg = calculate_cfd_score(guide, guide, pam="TGG")
    # Alternate NAG PAM (score = 0.259)
    score_nag = calculate_cfd_score(guide, guide, pam="AAG")
    # Non-functional PAM (score = 0.0)
    score_invalid = calculate_cfd_score(guide, guide, pam="ATC")

    assert score_ngg == 1.0
    assert abs(score_nag - 0.259) < 0.001
    assert score_invalid == 0.0

def test_cfd_seed_vs_distal():
    guide = "GAGTCCGAGCAGAAGAAGAA"
    # Distal mismatch at pos 1
    distal_hit = "A" + guide[1:]
    score_distal = calculate_cfd_score(guide, distal_hit, pam="AGG")

    # Seed mismatch at pos 20
    seed_hit = guide[:19] + "T"
    score_seed = calculate_cfd_score(guide, seed_hit, pam="AGG")

    # Seed mismatch should be penalized much more heavily than distal mismatch
    assert score_distal > score_seed
    print(f"CFD Distal: {score_distal:.4f} vs CFD Seed: {score_seed:.4f}")
