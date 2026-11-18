"""
Unit tests for competitive bioinformatics features:
- On-target efficiency scoring (Doench 2016 heuristics)
- Genomic feature annotation & oncogene flags
- PCR validation primer design
"""

import pytest
from geneweaver.biology.on_target import evaluate_on_target_efficiency
from geneweaver.biology.annotator import annotate_genomic_locus
from geneweaver.biology.primer import design_validation_primers


def test_on_target_efficiency():
    # Test valid guide with good GC and terminal G
    res = evaluate_on_target_efficiency("GAGTCCGAGCAGAAGAAGAG")
    assert 0.0 <= res.score <= 100.0
    assert res.tier in ["OPTIMAL", "GOOD", "MODERATE", "POOR"]
    assert res.gc_content_pct > 0
    assert res.melting_temp_c > 0

    # Test guide with Poly-T penalty
    res_poly_t = evaluate_on_target_efficiency("GAGTCCGATTTTAAGAAGAG")
    assert any("Poly-T" in p for p in res_poly_t.penalties)
    assert res_poly_t.score < res.score

    # Test invalid length
    res_invalid = evaluate_on_target_efficiency("ATGC")
    assert res_invalid.score == 0.0
    assert res_invalid.tier == "INVALID"


def test_genomic_annotation():
    # Test known clinical locus TP53
    ann_tp53 = annotate_genomic_locus("chr17", 7670000)
    assert ann_tp53.gene_symbol == "TP53"
    assert ann_tp53.is_cancer_hallmark is True
    assert ann_tp53.feature_type == "Exon (Coding)"

    # Test arbitrary position returns structured annotation
    ann_rand = annotate_genomic_locus("chr1", 1234567)
    assert ann_rand.feature_type in ["Exon (Coding)", "Promoter / 5' UTR", "Intron", "Intergenic Desert"]
    assert ann_rand.gene_symbol is not None


def test_pcr_validation_primers():
    primers = design_validation_primers("chr7", 142509211, "GAGTCCGAGCAGAAGAAGAATGG")
    assert len(primers.forward_primer) == 20
    assert len(primers.reverse_primer) == 20
    assert primers.forward_tm > 45.0
    assert primers.reverse_tm > 45.0
    assert primers.amplicon_length_bp == 350
    assert "Assay Method" in primers.pcr_protocol
