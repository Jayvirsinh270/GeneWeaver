"""
GeneWeaver Production REST API Microservice.
High-throughput bioinformatic endpoints for CRISPR alignment, Doench '16 on-target efficiency,
base editing (CBE/ABE) deamination windows, pegRNA prime editing, and in-vivo delivery kinetics.
Includes auto-generated Swagger UI (/docs) and OpenAPI specification.
"""

import time
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from geneweaver.parser.validator import validate_and_clean_guide, validate_and_clean_pam
from geneweaver.exceptions import InvalidGuideError, InvalidPamError
from geneweaver.baseline.cpu_aligner import CpuAligner
from geneweaver.biology.hsu_zhang import calculate_hsu_zhang_score, classify_risk_tier
from geneweaver.biology.cfd import calculate_cfd_score
from geneweaver.biology.on_target import evaluate_on_target_efficiency
from geneweaver.biology.annotator import annotate_genomic_locus
from geneweaver.biology.primer import design_validation_primers
from geneweaver.biology.delivery_sim import (
    simulate_in_vivo_delivery,
    CELL_TYPE_PROFILES,
    DELIVERY_MODALITIES
)
from geneweaver.biology.nucleases import (
    get_supported_nucleases,
    evaluate_nuclease_compatibility,
    NUCLEASE_REGISTRY
)
from geneweaver.biology.base_editor import (
    get_supported_base_editors,
    simulate_base_editing,
    design_pegrna,
    BASE_EDITOR_REGISTRY
)
from geneweaver.biology.dual_guide import (
    simulate_dual_guide_excision,
    design_excision_junction_primers,
    CLINICAL_EXCISION_PRESETS
)
from geneweaver.biology.variant_analyzer import (
    GenomicVariant,
    VariantImpactReport,
    PersonalizedAuditDossier,
    parse_vcf_lines,
    analyze_variant_crispr_impact,
    compute_allele_specific_window,
    audit_patient_variants,
    get_clinvar_benchmark_variants,
)
from geneweaver.biology.epigenetic import (
    ModulationModality,
    EffectorDomain,
    EffectorProfile,
    EFFECTOR_PROFILES,
    simulate_epigenetic_modulation,
    generate_promoter_window_profile,
    get_clinical_epigenetic_presets
)
from geneweaver.biology.gene_search import search_genes_and_diseases, extract_and_design_sgrnas_for_gene, GENE_DATABASE

app = FastAPI(
    title="GeneWeaver REST API",
    description=(
        "Production-grade RESTful API for GPU/CPU-accelerated CRISPR target alignment, "
        "MIT Hsu-Zhang & Broad CFD off-target scoring, Doench '16 on-target efficiency, "
        "Base Editing (CBE/ABE) bystander quantification, Prime Editing pegRNA architecture, "
        "and Preclinical In-Vivo Delivery kinetics."
    ),
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for frontend and notebook access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------- PYDANTIC SCHEMAS -----------------

class AlignRequest(BaseModel):
    guide_sequence: str = Field(..., description="CRISPR guide spacer (20-23 nt)", examples=["GAGTCCGAGCAGAAGAAGAA"])
    pam_motif: str = Field("NGG", description="PAM pattern", examples=["NGG"])
    target_dna_sequence: str = Field(..., description="Target genomic DNA sequence string to scan", examples=["GAGTCCGAGCAGAAGAAGAATGG"])
    chromosome_name: str = Field("chr1", description="Chromosome identifier", examples=["chr1"])
    max_mismatches: int = Field(4, ge=0, le=6, description="Max tolerated Hamming distance")
    nuclease_id: str = Field("sp_cas9", description="Endonuclease system: sp_cas9, cas12a_cpf1, sa_cas9, cas_mini")
    min_score_cutoff: float = Field(0.01, ge=0.0, le=1.0, description="Minimum cutting probability threshold")


class OnTargetRequest(BaseModel):
    guide_sequence: str = Field(..., description="Guide spacer sequence", examples=["GAGTCCGAGCAGAAGAAGAA"])
    pam_motif: str = Field("NGG", description="PAM motif", examples=["NGG"])


class BaseEditRequest(BaseModel):
    protospacer_sequence: str = Field(..., description="20-nt guide spacer sequence", examples=["GAGTCCGAGCAGAAGAAGAA"])
    target_position: int = Field(..., ge=1, le=20, description="1-indexed target base position to deaminate", examples=[5])
    editor_id: str = Field("be4max", description="Base editor system: be4max, ancbe4max, abe8e, abe7_10")


class PrimeEditRequest(BaseModel):
    target_guide: str = Field(..., description="20-nt guide spacer", examples=["GAGTCCGAGCAGAAGAAGAA"])
    edit_type: str = Field("SUBSTITUTION", description="Mutation type: SUBSTITUTION, INSERTION, DELETION")
    edit_offset_from_nick: int = Field(4, description="Distance (bp) from Cas9 nick position (nt 17-18)")
    desired_sequence: str = Field(..., description="Sequence to synthesize into target locus", examples=["T"])
    flanking_downstream_dna: Optional[str] = Field("GACGTACTGACGTACGTACT", description="Downstream reference DNA for homology arm")
    pbs_length: int = Field(13, ge=9, le=17, description="Primer Binding Site length (nt)")


class PrimerRequest(BaseModel):
    chromosome: str = Field("chr1", examples=["chr1"])
    cut_position: int = Field(..., examples=[1054200])
    locus_sequence: str = Field(..., description="23-bp target sequence", examples=["GAGTCCGAGCAGAAGAAGAATGG"])


class DeliverySimRequest(BaseModel):
    cell_type: str = Field("HSC (Hematopoietic Stem Cells)", description="Target patient cell/organ system")
    delivery_modality: str = Field("LNP (Lipid Nanoparticles)", description="Delivery vector: LNP, AAV, Electroporation, VLP")
    on_target_score: float = Field(80.0, ge=0.0, le=100.0)
    high_risk_off_targets: int = Field(0, ge=0)


class DualGuideExcisionRequest(BaseModel):
    guide_a_sequence: str = Field(..., description="5' upstream guide spacer (20 nt)", examples=["GGTTAATCCTTCAGCACATC"])
    guide_b_sequence: str = Field(..., description="3' downstream guide spacer (20 nt)", examples=["AGTTTTAATTTCATTCATTT"])
    cut_pos_a: int = Field(..., description="Chromosomal cut position A", examples=[1000])
    cut_pos_b: int = Field(..., description="Chromosomal cut position B", examples=[1580])
    is_nickase_mode: bool = Field(False, description="Whether to use paired Cas9 D10A nickases")
    strand_a: str = Field("(+)", description="Strand of guide A: (+) or (-)")
    strand_b: str = Field("(-)", description="Strand of guide B: (+) or (-)")
    pam_a: str = Field("NGG", description="PAM motif for guide A")
    pam_b: str = Field("NGG", description="PAM motif for guide B")


class ExcisionPrimerRequest(BaseModel):
    chromosome: str = Field("chrX", examples=["chrX"])
    cut_pos_a: int = Field(..., examples=[1000])
    cut_pos_b: int = Field(..., examples=[1580])
    upstream_flank_dna: Optional[str] = Field("GACGTACTGACGTACGTACTGACGTACTGACGTACGTACT")
    downstream_flank_dna: Optional[str] = Field("TGCATGCACTGCATGCACTGTGCATGCACTGCATGCACTG")


class VariantAuditItem(BaseModel):
    chrom: str = Field("chr11", examples=["chr11"])
    pos: int = Field(..., examples=[5227002])
    ref: str = Field(..., examples=["A"])
    alt: str = Field(..., examples=["T"])
    variant_id: str = Field(".", examples=["rs334"])
    gene_symbol: str = Field("", examples=["HBB"])
    clinical_significance: str = Field("Pathogenic", examples=["Pathogenic"])
    heterozygous: bool = Field(True)


class VariantAuditRequest(BaseModel):
    guide_sequence: str = Field(..., description="Target guide spacer (20 nt)", examples=["CTAACAGGCAGACTTCTCCA"])
    pam_motif: str = Field("AGG", description="PAM motif", examples=["AGG"])
    variants: List[VariantAuditItem] = Field(..., description="List of patient genomic variants")
    patient_id: str = Field("PATIENT-001", examples=["PATIENT-001"])
    target_locus_start: int = Field(5226982, description="Target locus 1-indexed start position")


class VcfParseRequest(BaseModel):
    vcf_text: str = Field(..., description="VCF v4.2 formatted text with tab or space-delimited columns")


class AlleleSelectivityRequest(BaseModel):
    wildtype_seq: str = Field(..., examples=["GTCCTGCAGGCTCCACGGGT"])
    mutant_seq: str = Field(..., examples=["GTCCTGCAGGTTCCACGGGT"])
    snp_pos_in_window: int = Field(15, ge=1, le=20, examples=[15])
    pam_motif: str = Field("NGG", examples=["NGG"])


class EpigeneticModulationRequest(BaseModel):
    gene_symbol: str = Field("HBG1", description="Target gene symbol", examples=["HBG1"])
    guide_sequence: str = Field(..., description="Target guide spacer (20 nt)", examples=["GGTCAACTTTCCTCTCACAC"])
    distance_to_tss_bp: int = Field(..., description="Distance in bp from TSS (negative = upstream promoter, positive = downstream)", examples=[-115])
    effector_key: str = Field("dcas9_vpr", description="Effector fusion key: dcas9_vpr, dcas9_suntag, dcas9_sam, dcas9_krab, dcas9_krab_mecp2")
    guide_efficiency_score: float = Field(85.0, ge=0.0, le=100.0, description="Predicted on-target binding efficiency score")


# ----------------- API ENDPOINTS -----------------

@app.get("/", summary="Root Health & Metadata")
def root():
    return {
        "engine": "GeneWeaver CRISPR Discovery & Precision Editing Engine",
        "version": "2.1.0",
        "status": "ONLINE",
        "documentation_url": "/docs",
        "capabilities": [
            "dual_mode_alignment",
            "doench16_on_target_efficiency",
            "hsu_zhang_and_cfd_scoring",
            "universal_multi_nuclease (SpCas9, Cas12a, SaCas9, CasMINI)",
            "base_editing_cbe_abe_bystander_simulation",
            "prime_editing_pegrna_architecture",
            "in_vivo_delivery_and_chromatin_simulation",
            "pcr_validation_primer_synthesis",
            "clinical_gene_disease_search"
        ]
    }


@app.post("/api/v1/align", summary="Execute Alignment & Off-Target Cleavage Scoring")
def align_target(req: AlignRequest):
    nuc_dict = get_supported_nucleases()
    nuc = nuc_dict.get(req.nuclease_id, nuc_dict["sp_cas9"])

    try:
        clean_g = validate_and_clean_guide(req.guide_sequence, expected_length=nuc.guide_length_nt)
        clean_pam = validate_and_clean_pam(req.pam_motif)
    except (InvalidGuideError, InvalidPamError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    t_start = time.perf_counter()
    aligner = CpuAligner(
        target_guide=clean_g,
        pam_motif=clean_pam,
        max_mismatches=req.max_mismatches,
        pam_orientation=nuc.pam_orientation
    )

    raw_hits = aligner.align_sequence(
        sequence=req.target_dna_sequence.upper(),
        chrom_name=req.chromosome_name,
        start_offset=0
    )
    elapsed = time.perf_counter() - t_start

    scored_hits = []
    p_len = len(clean_pam)
    g_len = nuc.guide_length_nt

    for h in raw_hits:
        if nuc.pam_orientation == "5_PRIME":
            cand_pam = h.sequence[:p_len]
            cand_guide = h.sequence[p_len:p_len + g_len]
        else:
            cand_guide = h.sequence[:g_len]
            cand_pam = h.sequence[g_len:g_len + p_len] if len(h.sequence) >= g_len + p_len else clean_pam

        cfd = calculate_cfd_score(clean_g, cand_guide, pam=cand_pam)
        if h.hsu_score >= req.min_score_cutoff or cfd >= req.min_score_cutoff:
            ann = annotate_genomic_locus(h.chrom_name, h.global_pos)
            scored_hits.append({
                "chromosome": h.chrom_name,
                "position": h.global_pos,
                "strand": h.strand,
                "mismatches": h.mismatches,
                "hsu_score": round(h.hsu_score, 4),
                "cfd_score": round(cfd, 4),
                "risk_tier": h.risk_tier,
                "gene_symbol": ann.gene_symbol,
                "genomic_feature": ann.feature_type,
                "clinical_consequence": ann.clinical_risk,
                "is_cancer_hallmark": ann.is_cancer_hallmark,
                "sequence": h.sequence
            })

    on_target = evaluate_on_target_efficiency(clean_g, clean_pam)

    return {
        "guide_used": clean_g,
        "pam_used": clean_pam,
        "nuclease": nuc.name,
        "nuclease_cut_geometry": nuc.cut_geometry,
        "pam_orientation": nuc.pam_orientation,
        "elapsed_seconds": round(elapsed, 4),
        "total_bp_scanned": len(req.target_dna_sequence),
        "total_hits_found": len(scored_hits),
        "high_risk_count": sum(1 for h in scored_hits if h["risk_tier"] == "HIGH_RISK"),
        "on_target_score": on_target.score,
        "on_target_tier": on_target.tier,
        "hits": scored_hits
    }


@app.post("/api/v1/on-target", summary="Doench 2016 Rule Set 2 Cutting Efficiency")
def get_on_target_score(req: OnTargetRequest):
    try:
        clean_g = validate_and_clean_guide(req.guide_sequence)
        clean_pam = validate_and_clean_pam(req.pam_motif)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    res = evaluate_on_target_efficiency(clean_g, clean_pam)
    return {
        "guide_sequence": clean_g,
        "pam_motif": clean_pam,
        "on_target_score": res.score,
        "tier": res.tier,
        "summary": res.summary,
        "gc_content_pct": res.gc_content_pct,
        "melting_temp_celsius": res.melting_temp_c,
        "penalties": res.penalties,
        "bonuses": res.bonuses
    }


@app.post("/api/v1/base-edit", summary="Base Editing & Bystander Mutation Simulator")
def simulate_base_edit_endpoint(req: BaseEditRequest):
    editors = get_supported_base_editors()
    if req.editor_id not in editors:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown base editor '{req.editor_id}'. Supported: {list(editors.keys())}"
        )

    try:
        clean_g = validate_and_clean_guide(req.protospacer_sequence, expected_length=20)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    try:
        res = simulate_base_editing(clean_g, req.target_position, req.editor_id)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))

    return {
        "editor_id": res.editor_id,
        "editor_name": res.editor_name,
        "editor_type": res.editor_type,
        "original_protospacer": res.original_protospacer,
        "target_position": res.target_position,
        "target_base": res.target_base,
        "converted_base": res.converted_base,
        "edited_protospacer": res.edited_protospacer,
        "is_target_in_window": res.is_target_in_window,
        "window_range": list(res.window_range),
        "window_sequence": res.window_sequence,
        "bystander_count": res.bystander_count,
        "bystander_risk_tier": res.bystander_risk_tier,
        "predicted_on_target_efficiency_pct": res.predicted_on_target_efficiency_pct,
        "product_purity_score_pct": res.product_purity_score_pct,
        "bystanders": [
            {
                "position": b.position,
                "original_base": b.original_base,
                "mutated_base": b.mutated_base,
                "predicted_edit_rate_pct": b.predicted_edit_rate_pct,
                "risk_level": b.risk_level
            }
            for b in res.bystanders_found
        ],
        "all_possible_alleles": res.all_possible_alleles,
        "biophysical_verdict": res.biophysical_verdict,
        "remediation_guidance": res.remediation_guidance
    }


@app.post("/api/v1/prime-edit", summary="Prime Editing pegRNA Architect")
def design_pegrna_endpoint(req: PrimeEditRequest):
    try:
        clean_g = validate_and_clean_guide(req.target_guide, expected_length=20)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    peg = design_pegrna(
        target_guide=clean_g,
        edit_type=req.edit_type,
        edit_offset_from_nick=req.edit_offset_from_nick,
        desired_sequence=req.desired_sequence,
        flanking_downstream_dna=req.flanking_downstream_dna or "GACGTACTGACGTACGTACT",
        pbs_length=req.pbs_length
    )

    return {
        "spacer_20nt": peg.spacer_seq_20nt,
        "scaffold_seq": peg.scaffold_seq,
        "rtt_seq": peg.rtt_seq,
        "pbs_seq": peg.pbs_seq,
        "composite_pegrna_seq": peg.composite_pegrna_seq,
        "intended_mutation": peg.intended_mutation,
        "pbs_length_nt": peg.pbs_length_nt,
        "pbs_tm_celsius": peg.pbs_tm_celsius,
        "pbs_gc_pct": peg.pbs_gc_pct,
        "rtt_length_nt": peg.rtt_length_nt,
        "rtt_gc_pct": peg.rtt_gc_pct,
        "nick_to_edit_distance_bp": peg.nick_to_edit_distance_bp,
        "feasibility_score": peg.feasibility_score,
        "feasibility_tier": peg.feasibility_tier,
        "recommendations": peg.recommendations
    }


@app.post("/api/v1/delivery-sim", summary="Preclinical In-Vivo Delivery & Chromatin Kinetics")
def simulate_delivery_endpoint(req: DeliverySimRequest):
    sim = simulate_in_vivo_delivery(
        cell_type_choice=req.cell_type,
        delivery_choice=req.delivery_modality,
        on_target_score=req.on_target_score,
        high_risk_off_target_count=req.high_risk_off_targets
    )

    return {
        "cell_type": req.cell_type,
        "delivery_modality": req.delivery_modality,
        "fda_ind_feasibility_score": sim.fda_ind_feasibility_score,
        "feasibility_tier": sim.feasibility_tier,
        "transduction_efficiency_pct": sim.transduction_efficiency_pct,
        "chromatin_accessibility_pct": sim.chromatin_accessibility_pct,
        "effective_in_vivo_cutting_pct": sim.effective_in_vivo_cutting_pct,
        "cas9_exposure_duration": sim.cas9_exposure_duration,
        "preclinical_advantages": sim.preclinical_advantages,
        "preclinical_warnings": sim.preclinical_warnings,
        "regulatory_recommendation": sim.regulatory_recommendation
    }


@app.post("/api/v1/primers", summary="Automated PCR Primer Synthesis")
def generate_primers(req: PrimerRequest):
    primers = design_validation_primers(req.chromosome, req.cut_position, req.locus_sequence)
    return {
        "chromosome": primers.target_chrom,
        "cut_position": primers.cut_position,
        "forward_primer": primers.forward_primer,
        "forward_tm": primers.forward_tm,
        "forward_gc": primers.forward_gc,
        "reverse_primer": primers.reverse_primer,
        "reverse_tm": primers.reverse_tm,
        "reverse_gc": primers.reverse_gc,
        "amplicon_length_bp": primers.amplicon_length_bp
    }


@app.get("/api/v1/genes", summary="Search Clinical Human Gene & Disease Database")
def search_genes(query: str = Query("", description="Disease or gene symbol search term")):
    results = search_genes_and_diseases(query)
    formatted = []
    for g in results:
        leads = extract_and_design_sgrnas_for_gene(g, max_candidates=2)
        formatted.append({
            "gene_symbol": g.gene_symbol,
            "disease_indication": g.disease_indication,
            "chromosome": g.chromosome,
            "therapeutic_phase": g.therapeutic_phase,
            "clinical_summary": g.clinical_summary,
            "recommended_guides": [
                {
                    "rank": l.rank,
                    "guide_seq": l.guide_seq,
                    "on_target_score": l.on_target_score,
                    "strand": l.strand,
                    "clinical_rationale": l.clinical_rationale
                }
                for l in leads
            ]
        })
    return {"query": query, "count": len(formatted), "results": formatted}


@app.get("/api/v1/nucleases", summary="List Registered Endonuclease Systems")
def list_nucleases():
    nucs = get_supported_nucleases()
    return {
        k: {
            "name": v.name,
            "organism": v.organism,
            "canonical_pam": v.canonical_pam,
            "pam_orientation": v.pam_orientation,
            "guide_length_nt": v.guide_length_nt,
            "cut_geometry": v.cut_geometry,
            "aav_packageable": v.aav_packageable,
            "clinical_use_case": v.clinical_use_case
        }
        for k, v in nucs.items()
    }


@app.get("/api/v1/base-editors", summary="List Registered Base Editing Systems")
def list_base_editors():
    beds = get_supported_base_editors()
    return {
        k: {
            "name": v.name,
            "editor_type": v.editor_type,
            "target_base": v.target_base,
            "product_base": v.product_base,
            "deamination_window": list(v.deamination_window),
            "peak_positions": v.peak_positions,
            "clinical_use_case": v.clinical_use_case
        }
        for k, v in beds.items()
    }


@app.post("/api/v1/dual-guide/excision", summary="Simulate Dual-Guide Genomic Excision & Nickase Kinetics")
def simulate_dual_guide_endpoint(req: DualGuideExcisionRequest):
    try:
        clean_a = validate_and_clean_guide(req.guide_a_sequence, expected_length=20)
        clean_b = validate_and_clean_guide(req.guide_b_sequence, expected_length=20)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    res = simulate_dual_guide_excision(
        guide_a_seq=clean_a,
        guide_b_seq=clean_b,
        cut_pos_a=req.cut_pos_a,
        cut_pos_b=req.cut_pos_b,
        is_nickase_mode=req.is_nickase_mode,
        strand_a=req.strand_a,
        strand_b=req.strand_b,
        pam_a=req.pam_a,
        pam_b=req.pam_b
    )

    return {
        "deletion_span_bp": res.deletion_span_bp,
        "is_nickase_mode": res.is_nickase_mode,
        "nickase_offset_bp": res.nickase_offset_bp,
        "nickase_orientation": res.nickase_orientation,
        "predicted_deletion_efficiency_pct": res.predicted_deletion_efficiency_pct,
        "predicted_inversion_rate_pct": res.predicted_inversion_rate_pct,
        "predicted_unedited_or_indel_rate_pct": res.predicted_unedited_or_indel_rate_pct,
        "guide_a_on_target_score": res.guide_a_on_target_score,
        "guide_b_on_target_score": res.guide_b_on_target_score,
        "composite_cutting_potency": res.composite_cutting_potency,
        "off_target_safety_advantage": res.off_target_safety_advantage,
        "wildtype_amplicon_bp": res.wildtype_amplicon_bp,
        "deleted_amplicon_bp": res.deleted_amplicon_bp,
        "deletion_junction_sequence": res.deletion_junction_sequence,
        "clinical_recommendation": res.clinical_recommendation,
        "biophysical_warnings": res.biophysical_warnings
    }


@app.post("/api/v1/dual-guide/primers", summary="Design Excision Junction PCR Screening Primers")
def design_junction_primers_endpoint(req: ExcisionPrimerRequest):
    primers = design_excision_junction_primers(
        chrom=req.chromosome,
        cut_pos_a=req.cut_pos_a,
        cut_pos_b=req.cut_pos_b,
        upstream_flank_dna=req.upstream_flank_dna or "GACGTACTGACGTACGTACTGACGTACTGACGTACGTACT",
        downstream_flank_dna=req.downstream_flank_dna or "TGCATGCACTGCATGCACTGTGCATGCACTGCATGCACTG"
    )
    return primers


@app.get("/api/v1/dual-guide/presets", summary="List Clinical Excision & Exon Skipping Benchmarks")
def list_excision_presets():
    return CLINICAL_EXCISION_PRESETS


@app.post("/api/v1/variants/audit", summary="Audit Patient Variants Against Target Guide for PAM/Seed Mutations")
def audit_variants_endpoint(req: VariantAuditRequest):
    try:
        clean_g = validate_and_clean_guide(req.guide_sequence, expected_length=20)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    domain_variants = [
        GenomicVariant(
            chrom=v.chrom,
            pos=v.pos,
            ref=v.ref,
            alt=v.alt,
            variant_id=v.variant_id,
            gene_symbol=v.gene_symbol,
            clinical_significance=v.clinical_significance,
            heterozygous=v.heterozygous
        )
        for v in req.variants
    ]

    dossier = audit_patient_variants(
        guide_seq=clean_g,
        pam_motif=req.pam_motif,
        variants=domain_variants,
        patient_id=req.patient_id,
        target_locus_start=req.target_locus_start
    )

    return {
        "patient_id": dossier.patient_id,
        "guide_seq": dossier.guide_seq,
        "pam_motif": dossier.pam_motif,
        "num_variants_screened": dossier.num_variants_screened,
        "therapy_viable_in_patient": dossier.therapy_viable_in_patient,
        "risk_level": dossier.risk_level,
        "key_findings": dossier.key_findings,
        "impacted_variants": [
            {
                "variant_id": r.variant.variant_id,
                "chrom": r.variant.chrom,
                "pos": r.variant.pos,
                "ref": r.variant.ref,
                "alt": r.variant.alt,
                "gene_symbol": r.variant.gene_symbol,
                "consequence": r.consequence.value,
                "ref_affinity_score": r.ref_affinity_score,
                "alt_affinity_score": r.alt_affinity_score,
                "allele_selectivity_ratio": r.allele_selectivity_ratio,
                "is_pam_impacted": r.is_pam_impacted,
                "is_seed_impacted": r.is_seed_impacted,
                "position_in_guide": r.position_in_guide,
                "description": r.description,
                "recommendation": r.recommendation
            }
            for r in dossier.impacted_variants
        ]
    }


@app.post("/api/v1/variants/parse-vcf", summary="Parse VCF Text into Structured Genomic Variants")
def parse_vcf_endpoint(req: VcfParseRequest):
    variants = parse_vcf_lines(req.vcf_text)
    return {
        "total_parsed": len(variants),
        "variants": [
            {
                "chrom": v.chrom,
                "pos": v.pos,
                "variant_id": v.variant_id,
                "ref": v.ref,
                "alt": v.alt,
                "gene_symbol": v.gene_symbol,
                "clinical_significance": v.clinical_significance,
                "heterozygous": v.heterozygous
            }
            for v in variants
        ]
    }


@app.post("/api/v1/variants/allele-selectivity", summary="Calculate Allele-Specific Discriminatory Selectivity Ratio")
def allele_selectivity_endpoint(req: AlleleSelectivityRequest):
    res = compute_allele_specific_window(
        wildtype_seq=req.wildtype_seq,
        mutant_seq=req.mutant_seq,
        snp_pos_in_window=req.snp_pos_in_window,
        pam_motif=req.pam_motif
    )
    return res


@app.get("/api/v1/variants/benchmarks", summary="Get Pre-loaded ClinVar / dbSNP Benchmark Variants")
def get_benchmarks_endpoint():
    return get_clinvar_benchmark_variants()


@app.post("/api/v1/epigenetics/modulate", summary="Simulate Epigenetic CRISPRa / CRISPRi Transcriptional Modulation")
def modulate_epigenetic_endpoint(req: EpigeneticModulationRequest):
    try:
        clean_g = validate_and_clean_guide(req.guide_sequence, expected_length=20)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    res = simulate_epigenetic_modulation(
        gene_symbol=req.gene_symbol,
        guide_seq=clean_g,
        distance_to_tss_bp=req.distance_to_tss_bp,
        effector_key=req.effector_key,
        guide_efficiency_score=req.guide_efficiency_score
    )

    return {
        "gene_symbol": res.gene_symbol,
        "modality": res.modality.value,
        "effector_name": res.effector_name,
        "guide_seq": res.guide_seq,
        "distance_to_tss_bp": res.distance_to_tss_bp,
        "is_in_optimal_window": res.is_in_optimal_window,
        "window_attenuation_factor": res.window_attenuation_factor,
        "predicted_fold_change": res.predicted_fold_change,
        "log2_fold_change": res.log2_fold_change,
        "percent_modulation": res.percent_modulation,
        "retained_expression_pct": res.retained_expression_pct,
        "off_target_epigenetic_risk": res.off_target_epigenetic_risk,
        "delivery_feasibility": res.delivery_feasibility,
        "clinical_recommendation": res.clinical_recommendation,
        "biophysical_notes": res.biophysical_notes
    }


@app.get("/api/v1/epigenetics/effectors", summary="List Available CRISPRa / CRISPRi Epigenetic Effector Profiles")
def list_epigenetic_effectors():
    return {
        k: {
            "name": v.name,
            "modality": v.modality.value,
            "domain": v.domain.value,
            "max_fold_modulation": v.max_fold_modulation,
            "optimal_window_start_bp": v.optimal_window_start_bp,
            "optimal_window_end_bp": v.optimal_window_end_bp,
            "peak_distance_bp": v.peak_distance_bp,
            "epigenetic_mark": v.epigenetic_mark,
            "delivery_size_kb": v.delivery_size_kb,
            "description": v.description
        }
        for k, v in EFFECTOR_PROFILES.items()
    }


@app.get("/api/v1/epigenetics/presets", summary="List Curated Clinical Epigenetic Modulation Benchmarks")
def list_epigenetic_presets():
    return get_clinical_epigenetic_presets()


@app.get("/api/v1/epigenetics/promoter-curve", summary="Generate TSS Promoter Distance Modulation Curve Points")
def get_promoter_curve(gene_symbol: str = "HBG1", effector_key: str = "dcas9_vpr", step_bp: int = 25):
    return generate_promoter_window_profile(gene_symbol=gene_symbol, effector_key=effector_key, step_bp=step_bp)

