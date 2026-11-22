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
    AlignRequest,
    OnTargetRequest,
    BaseEditRequest,
    PrimeEditRequest,
    DeliverySimRequest,
    PrimerRequest
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
