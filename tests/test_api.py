"""
Unit test suite for GeneWeaver REST API Microservice.
Tests endpoints directly and validates Pydantic request/response schemas.
"""

from geneweaver.api.app import (
    root,
    align_target,
    get_on_target_score,
    simulate_base_edit_endpoint,
    design_pegrna_endpoint,
    simulate_delivery_endpoint,
    generate_primers,
    search_genes,
    list_nucleases,
    list_base_editors,
    simulate_dual_guide_endpoint,
    design_junction_primers_endpoint,
    list_excision_presets,
    AlignRequest,
    OnTargetRequest,
    BaseEditRequest,
    PrimeEditRequest,
    DeliverySimRequest,
    PrimerRequest,
    DualGuideExcisionRequest,
    ExcisionPrimerRequest,
    VariantAuditRequest,
    VariantAuditItem,
    audit_variants_endpoint,
    VcfParseRequest,
    parse_vcf_endpoint,
    AlleleSelectivityRequest,
    allele_selectivity_endpoint,
    get_benchmarks_endpoint,
    EpigeneticModulationRequest,
    modulate_epigenetic_endpoint,
    list_epigenetic_effectors,
    list_epigenetic_presets,
    get_promoter_curve
)


def test_api_root():
    data = root()
    assert data["status"] == "ONLINE"
    assert "capabilities" in data
    assert len(data["capabilities"]) >= 5


def test_api_on_target():
    req = OnTargetRequest(
        guide_sequence="GAGTCCGAGCAGAAGAAGAA",
        pam_motif="NGG"
    )
    data = get_on_target_score(req)
    assert "on_target_score" in data
    assert data["on_target_score"] > 0
    assert data["tier"] in ["OPTIMAL", "GOOD", "MODERATE", "POOR"]


def test_api_base_edit():
    req = BaseEditRequest(
        protospacer_sequence="GAGTCCGAGCAGAAGAAGAA",
        target_position=5,
        editor_id="be4max"
    )
    data = simulate_base_edit_endpoint(req)
    assert data["editor_type"] == "CBE"
    assert data["converted_base"] == "T"
    assert "product_purity_score_pct" in data
    assert "bystander_count" in data


def test_api_prime_edit():
    req = PrimeEditRequest(
        target_guide="GAGTCCGAGCAGAAGAAGAA",
        edit_type="SUBSTITUTION",
        edit_offset_from_nick=4,
        desired_sequence="A",
        pbs_length=13
    )
    data = design_pegrna_endpoint(req)
    assert "composite_pegrna_seq" in data
    assert data["pbs_length_nt"] == 13
    assert "feasibility_score" in data


def test_api_align():
    req = AlignRequest(
        guide_sequence="GAGTCCGAGCAGAAGAAGAA",
        pam_motif="NGG",
        target_dna_sequence="GAGTCCGAGCAGAAGAAGAATGGACGT",
        chromosome_name="chr1",
        max_mismatches=2,
        nuclease_id="sp_cas9"
    )
    data = align_target(req)
    assert data["total_hits_found"] >= 1
    assert data["hits"][0]["mismatches"] == 0
    assert "on_target_score" in data


def test_api_delivery_sim():
    req = DeliverySimRequest(
        cell_type="HSC (Hematopoietic Stem Cells)",
        delivery_modality="LNP (Lipid Nanoparticles)",
        on_target_score=85.0,
        high_risk_off_targets=0
    )
    data = simulate_delivery_endpoint(req)
    assert data["fda_ind_feasibility_score"] > 0
    assert "transduction_efficiency_pct" in data


def test_api_primers():
    req = PrimerRequest(
        chromosome="chr1",
        cut_position=100000,
        locus_sequence="GAGTCCGAGCAGAAGAAGAATGG"
    )
    data = generate_primers(req)
    assert len(data["forward_primer"]) >= 15
    assert len(data["reverse_primer"]) >= 15


def test_api_genes():
    data = search_genes(query="Huntington")
    assert data["count"] >= 1
    assert data["results"][0]["gene_symbol"] == "HTT"


def test_api_nucleases():
    data = list_nucleases()
    assert "sp_cas9" in data
    assert "cas12a_cpf1" in data
    assert "sa_cas9" in data
    assert "cas_mini" in data


def test_api_base_editors():
    data = list_base_editors()
    assert "be4max" in data
    assert "abe8e" in data


def test_api_dual_guide_excision():
    req = DualGuideExcisionRequest(
        guide_a_sequence="GGTTAATCCTTCAGCACATC",
        guide_b_sequence="AGTTTTAATTTCATTCATTT",
        cut_pos_a=1000,
        cut_pos_b=1580,
        is_nickase_mode=False
    )
    data = simulate_dual_guide_endpoint(req)
    assert data["deletion_span_bp"] == 580
    assert data["predicted_deletion_efficiency_pct"] > 0.0
    assert data["wildtype_amplicon_bp"] > data["deleted_amplicon_bp"]


def test_api_dual_guide_primers():
    req = ExcisionPrimerRequest(
        chromosome="chrX",
        cut_pos_a=1000,
        cut_pos_b=1580
    )
    data = design_junction_primers_endpoint(req)
    assert data["chromosome"] == "chrX"
    assert "forward_primer_5to3" in data
    assert "reverse_primer_5to3" in data


def test_api_dual_guide_presets():
    presets = list_excision_presets()
    assert "dmd_exon_51_skipping" in presets
    assert "egfr_exon_19_deletion" in presets


def test_api_variant_audit():
    req = VariantAuditRequest(
        guide_sequence="CTAACAGGCAGACTTCTCCA",
        pam_motif="AGG",
        variants=[
            VariantAuditItem(
                chrom="chr11",
                pos=5227003,  # Within PAM
                ref="G",
                alt="A",
                variant_id="rs_pam_destroy",
                gene_symbol="HBB"
            )
        ],
        patient_id="PT-001",
        target_locus_start=5226982
    )
    res = audit_variants_endpoint(req)
    assert res["patient_id"] == "PT-001"
    assert res["num_variants_screened"] == 1
    assert res["therapy_viable_in_patient"] is False
    assert res["risk_level"] == "THERAPY_CONTRAINDICATED"
    assert len(res["impacted_variants"]) == 1
    assert res["impacted_variants"][0]["consequence"] == "PAM_DESTROYED"


def test_api_vcf_parse():
    vcf_text = "chr11\t5227002\trs334\tA\tT\t.\tPASS\tGENE=HBB\tGT\t0/1\n"
    res = parse_vcf_endpoint(VcfParseRequest(vcf_text=vcf_text))
    assert res["total_parsed"] == 1
    assert res["variants"][0]["variant_id"] == "rs334"


def test_api_allele_selectivity():
    req = AlleleSelectivityRequest(
        wildtype_seq="GTCCTGCAGGCTCCACGGGT",
        mutant_seq="GTCCTGCAGGTTCCACGGGT",
        snp_pos_in_window=15
    )
    res = allele_selectivity_endpoint(req)
    assert res["allele_selectivity_ratio"] >= 10.0
    assert res["is_clinically_selective"] is True


def test_api_variant_benchmarks():
    benchmarks = get_benchmarks_endpoint()
    assert "HBB_rs334_SickleCell" in benchmarks
    assert "BRAF_rs113488022_V600E" in benchmarks


def test_api_epigenetic_modulate():
    req = EpigeneticModulationRequest(
        gene_symbol="HBG1",
        guide_sequence="GGTCAACTTTCCTCTCACAC",
        distance_to_tss_bp=-115,
        effector_key="dcas9_vpr",
        guide_efficiency_score=85.0
    )
    res = modulate_epigenetic_endpoint(req)
    assert res["gene_symbol"] == "HBG1"
    assert res["modality"] == "CRISPRa"
    assert res["predicted_fold_change"] > 10.0
    assert res["is_in_optimal_window"] is True


def test_api_epigenetic_effectors():
    effs = list_epigenetic_effectors()
    assert "dcas9_vpr" in effs
    assert "dcas9_krab_mecp2" in effs
    assert effs["dcas9_vpr"]["modality"] == "CRISPRa"


def test_api_epigenetic_presets():
    presets = list_epigenetic_presets()
    assert "HBG1_Activation_SickleCell" in presets
    assert "MYC_Silencing_Oncology" in presets


def test_api_promoter_curve():
    pts = get_promoter_curve(gene_symbol="HBG1", effector_key="dcas9_vpr", step_bp=50)
    assert len(pts) > 10
    assert pts[0]["distance_from_tss_bp"] == -600



