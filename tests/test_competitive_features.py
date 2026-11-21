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


def test_karyotype_ideogram_and_pdf_generation():
    import pandas as pd
    from geneweaver.web.ideogram import build_karyotype_ideogram_plot
    from geneweaver.web.pdf_report import generate_clinical_pdf_dossier

    df = pd.DataFrame([{
        'Chromosome': 'chr7',
        'Position': 142509211,
        'Strand': '+',
        'Mismatches': 1,
        'Hsu_Score': 0.842,
        'CFD_Score': 0.912,
        'Risk_Tier': 'HIGH_RISK',
        'Gene_Symbol': 'LOC231',
        'Genomic_Feature': 'Exon (Coding)',
        'Clinical_Consequence': 'CRITICAL',
        'Is_Cancer_Hallmark': False,
        'Sequence': 'GAGTCCGAGCAGAAGAAGAATGG'
    }])

    # 1. Ideogram test
    fig = build_karyotype_ideogram_plot(df)
    assert fig is not None
    assert len(fig.data) > 0

    # 2. PDF report test
    metrics = {
        'disease_name': 'Sickle Cell BCL11A',
        'guide_used': 'CTAACAGTTGCTTTTATCAC',
        'pam_used': 'AGG',
        'on_target_score': 85.0,
        'on_target_tier': 'OPTIMAL',
        'total_bp': 1000000,
        'throughput_mbp': 25.0,
        'elapsed': 0.04,
        'total_candidates': 1,
        'high_risk_count': 1,
        'exonic_hits_count': 1,
        'cancer_hallmark_hits_count': 0
    }
    pdf_bytes = generate_clinical_pdf_dossier(metrics, df)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF")


def test_auto_repair_optimizer():
    from geneweaver.biology.optimizer import repair_and_optimize_guide

    dossier = repair_and_optimize_guide("GAGTCCGAGCAGAAGAAGAT")
    assert dossier.original_guide == "GAGTCCGAGCAGAAGAAGAT"
    assert len(dossier.repaired_candidates) == 3
    assert dossier.best_candidate is not None
    assert dossier.best_candidate.therapeutic_fitness_index > 0
    assert len(dossier.best_candidate.guide_sequence) == 20
    assert dossier.best_candidate.on_target_score > 0


def test_gene_search_and_designer():
    from geneweaver.biology.gene_search import search_genes_and_diseases, extract_and_design_sgrnas_for_gene, GENE_DATABASE

    # 1. Search by disease name
    results_huntington = search_genes_and_diseases("Huntington")
    assert len(results_huntington) >= 1
    assert results_huntington[0].gene_symbol == "HTT"

    # 2. Search by gene symbol
    results_egfr = search_genes_and_diseases("EGFR")
    assert len(results_egfr) >= 1
    assert results_egfr[0].chromosome == "chr7"

    # 3. Design sgRNAs from gene exon
    entry = GENE_DATABASE["HBB"]
    candidates = extract_and_design_sgrnas_for_gene(entry, target_pam="NGG", max_candidates=5)
    assert len(candidates) > 0
    assert len(candidates[0].guide_seq) == 20
    assert candidates[0].on_target_score > 0


def test_delivery_simulator():
    from geneweaver.biology.delivery_sim import simulate_in_vivo_delivery, CELL_TYPE_PROFILES, DELIVERY_MODALITIES

    cell_type = list(CELL_TYPE_PROFILES.keys())[0]
    delivery = list(DELIVERY_MODALITIES.keys())[0]

    sim = simulate_in_vivo_delivery(
        cell_type_choice=cell_type,
        delivery_choice=delivery,
        on_target_score=85.0,
        high_risk_off_target_count=0
    )

    assert sim.fda_ind_feasibility_score > 0
    assert sim.effective_in_vivo_cutting_pct > 0
    assert sim.feasibility_tier in ["HIGH_FEASIBILITY", "MODERATE_FEASIBILITY", "CHALLENGING"]
    assert len(sim.preclinical_advantages) > 0


def test_3d_structure_viewer():
    from geneweaver.web.structure_viewer import generate_3dmol_html

    html = generate_3dmol_html(
        target_guide="GAGTCCGAGCAGAAGAAGAA",
        locus_seq="GAGTCCGAGCAGAAGAAGAT",
        mismatch_positions=[20],
        height=400
    )
    assert isinstance(html, str)
    assert "3Dmol" in html
    assert "pdb:4UN3" in html
    assert "STERIC CLASH" in html


def test_multi_nuclease_architecture():
    from geneweaver.biology.nucleases import (
        get_supported_nucleases,
        evaluate_nuclease_compatibility,
        NUCLEASE_REGISTRY
    )

    nucleases = get_supported_nucleases()
    assert "sp_cas9" in nucleases
    assert "cas12a_cpf1" in nucleases
    assert "sa_cas9" in nucleases
    assert "cas_mini" in nucleases

    # 1. SpCas9 evaluation (20nt)
    ev_sp = evaluate_nuclease_compatibility("sp_cas9", "GAGTCCGAGCAGAAGAAGAA")
    assert ev_sp.is_valid_length is True
    assert ev_sp.expected_length == 20
    assert ev_sp.cut_type == "BLUNT"

    # 2. Cas12a evaluation (23nt)
    ev_c12 = evaluate_nuclease_compatibility("cas12a_cpf1", "GAGTCCGAGCAGAAGAAGAATTT")
    assert ev_c12.is_valid_length is True
    assert ev_c12.expected_length == 23
    assert ev_c12.cut_type == "STAGGERED_5NT_OVERHANG"

    # 3. SaCas9 evaluation (21nt)
    ev_sa = evaluate_nuclease_compatibility("sa_cas9", "GAGTCCGAGCAGAAGAAGAAT")
    assert ev_sa.is_valid_length is True
    assert ev_sa.expected_length == 21
    assert ev_sa.aav_single_vector_fit is True

    # 4. Multi-Nuclease Aligner Tests
    from geneweaver.baseline.cpu_aligner import CpuAligner
    from geneweaver.parser.validator import validate_and_clean_guide

    # Test Cas12a alignment with 5' PAM (TTTV)
    cas12_guide = "GAGTCCGAGCAGAAGAAGAATTT"
    clean_c12 = validate_and_clean_guide(cas12_guide, expected_length=23)
    assert len(clean_c12) == 23

    aligner_c12 = CpuAligner(
        target_guide=clean_c12,
        pam_motif="TTTA",
        max_mismatches=2,
        pam_orientation="5_PRIME"
    )
    # Synthetic target with 5' TTTA followed by exact 23nt guide
    synth_target = "ACGT" + "TTTA" + clean_c12 + "ACGT"
    hits = aligner_c12.align_sequence(synth_target)
    assert len(hits) == 1
    assert hits[0].mismatches == 0
    assert hits[0].hsu_score == 1.0

    # Test SaCas9 alignment with 3' PAM (NNGRRT)
    sa_guide = "GAGTCCGAGCAGAAGAAGAAT"
    clean_sa = validate_and_clean_guide(sa_guide, expected_length=21)
    assert len(clean_sa) == 21

    aligner_sa = CpuAligner(
        target_guide=clean_sa,
        pam_motif="NNGGAT",
        max_mismatches=2,
        pam_orientation="3_PRIME"
    )
    synth_sa = "ACGT" + clean_sa + "AAGGAT" + "ACGT"
    hits_sa = aligner_sa.align_sequence(synth_sa)
    assert len(hits_sa) == 1
    assert hits_sa[0].mismatches == 0






