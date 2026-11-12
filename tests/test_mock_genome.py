"""
Unit Tests for GeneWeaver Core Modules.
"""

import pytest
from geneweaver.biology.hsu_zhang import calculate_hsu_zhang_score, classify_risk_tier
from geneweaver.parser.mock_genome import generate_mock_chromosome

def test_hsu_score_perfect_match():
    guide = "GAGTCCGAGCAGAAGAAGAA"
    score = calculate_hsu_zhang_score(guide, guide)
    assert score == 1.0
    assert classify_risk_tier(score) == "HIGH_RISK"

def test_hsu_score_seed_vs_distal():
    guide = "GAGTCCGAGCAGAAGAAGAA"
    # Seed mismatch at pos 20 (penalized heavily)
    seed_variant = guide[:19] + "T"
    seed_score = calculate_hsu_zhang_score(guide, seed_variant)

    # Distal mismatch at pos 1 (tolerated more)
    distal_variant = "T" + guide[1:]
    distal_score = calculate_hsu_zhang_score(guide, distal_variant)

    # Distal mismatch should retain higher cleavage probability than seed mismatch
    assert distal_score > seed_score
    print(f"Distal Score: {distal_score:.4f} vs Seed Score: {seed_score:.4f}")

def test_mock_chromosome_generation():
    spikes = [(100, "ATGCGATCGATCGAACGTACAGG")]
    seq = generate_mock_chromosome(1000, spikes=spikes, seed=123)
    assert len(seq) == 1000
    assert seq[100:100 + 23] == "ATGCGATCGATCGAACGTACAGG"
