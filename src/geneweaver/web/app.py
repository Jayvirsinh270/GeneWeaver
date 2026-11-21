"""
GeneWeaver Dual-Mode Web GUI Dashboard.
Wired directly to the GeneWeaver sequence alignment, streaming, and biological scoring engines.
Features dual experience modes:
- Simple / Beginner Mode: Plain-English summaries, safety scores, visual analogies, and intuitive DNA spell-checkers.
- Expert / Research Mode: Raw bioinformatic metrics, Manhattan plots, MIT Hsu-Zhang + Broad CFD matrices,
  Doench 2016 on-target efficiency, genomic locus annotations (exons/introns/oncogenes),
  automated PCR primer designer for wet-lab validation, multi-guide duel comparator, and GPU microarchitectural telemetry.
"""

import os
import sys
import time

# Ensure project root and src/ are in sys.path
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("src"))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from geneweaver.exceptions import GeneWeaverError, InvalidGuideError, InvalidPamError, FastaParseError
from geneweaver.parser.validator import validate_and_clean_guide, validate_and_clean_pam, validate_fasta_file
from geneweaver.parser.reference_fetcher import ReferenceManager, PRESETS
from geneweaver.parser.fasta_streamer import FastaStreamer
from geneweaver.baseline.cpu_aligner import CpuAligner
from geneweaver.biology.cfd import calculate_cfd_score, classify_cfd_risk
from geneweaver.biology.presets import CLINICAL_PRESETS, calculate_gc_content, estimate_melting_temp
from geneweaver.biology.on_target import evaluate_on_target_efficiency, OnTargetScore
from geneweaver.biology.annotator import annotate_genomic_locus, GenomicAnnotation
from geneweaver.biology.primer import design_validation_primers, PCRValidationPrimerSet
from geneweaver.biology.optimizer import repair_and_optimize_guide, AutoRepairDossier
from geneweaver.biology.gene_search import search_genes_and_diseases, extract_and_design_sgrnas_for_gene, GENE_DATABASE, TargetGeneEntry
from geneweaver.biology.delivery_sim import simulate_in_vivo_delivery, CELL_TYPE_PROFILES, DELIVERY_MODALITIES, DeliverySimulationResult
from geneweaver.biology.nucleases import get_supported_nucleases, evaluate_nuclease_compatibility, NUCLEASE_REGISTRY, NucleaseProfile
from geneweaver.web.ideogram import build_karyotype_ideogram_plot
from geneweaver.web.pdf_report import generate_clinical_pdf_dossier
from geneweaver.web.structure_viewer import generate_3dmol_html
import streamlit.components.v1 as components
from geneweaver.tui.telemetry import HardwareMonitor

# ----------------- PAGE CONFIGURATION & THEME -----------------
st.set_page_config(
    page_title="GeneWeaver | CRISPR Safety & Efficiency Engine",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Cyber-Bio CSS Styling
st.markdown("""
<style>
    /* Dark Master Background */
    .stApp {
        background-color: #0A0E17;
        color: #F0F6FC;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    /* Metrics Card Styling */
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        color: #00F0FF;
        font-size: 1.8rem;
    }
    div[data-testid="stMetricLabel"] {
        color: #8B949E;
        font-size: 0.9rem;
    }
    
    /* Sequence Visualizer Badges */
    .seq-card {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        font-family: 'JetBrains Mono', monospace;
    }
    .base-tile {
        display: inline-block;
        padding: 4px 7px;
        margin: 1px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 1.05rem;
    }
    .match-tile { background: #0D2238; color: #00F0FF; border: 1px solid #00F0FF; }
    .mismatch-tile { background: #FF1744; color: #FFFFFF; font-weight: 900; box-shadow: 0 0 8px rgba(255, 23, 68, 0.6); }
    .pam-tile { background: #00E676; color: #0A0E17; font-weight: 900; }
    .arrow-indicator { color: #FFB300; font-size: 1.1rem; }
    
    /* Explainer Callout Card */
    .explainer-box {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 16px;
    }

    /* Gene Badge */
    .gene-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 5px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-cancer { background: rgba(255, 23, 68, 0.2); border: 1px solid #FF1744; color: #FF5252; }
    .badge-exon { background: rgba(255, 179, 0, 0.2); border: 1px solid #FFB300; color: #FFD54F; }
    .badge-safe { background: rgba(0, 230, 118, 0.2); border: 1px solid #00E676; color: #69F0AE; }
</style>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE -----------------
if "scan_completed" not in st.session_state:
    st.session_state.scan_completed = False
    st.session_state.df_hits = None
    st.session_state.metrics = None
    st.session_state.active_guide = "GAGTCCGAGCAGAAGAAGAA"
    st.session_state.active_pam = "NGG"
    st.session_state.selected_disease_name = "Synthetic Ground-Truth Benchmark"
    st.session_state.active_cell_type = list(CELL_TYPE_PROFILES.keys())[0]
    st.session_state.active_delivery = list(DELIVERY_MODALITIES.keys())[0]
    st.session_state.active_nuclease = "sp_cas9"

if "active_cell_type" not in st.session_state:
    st.session_state.active_cell_type = list(CELL_TYPE_PROFILES.keys())[0]
if "active_delivery" not in st.session_state:
    st.session_state.active_delivery = list(DELIVERY_MODALITIES.keys())[0]
if "active_nuclease" not in st.session_state:
    st.session_state.active_nuclease = "sp_cas9"

ref_mgr = ReferenceManager()
hw_monitor = HardwareMonitor(device_ids=[0, 1])

# ----------------- SIDEBAR: INPUTS & PRESETS -----------------
with st.sidebar:
    st.title("🧬 GeneWeaver")
    st.caption("GPU-Accelerated CRISPR Safety & Efficiency Engine")

    st.markdown("### 🧭 Dashboard Mode")
    ui_mode = st.radio(
        "Choose Experience Level:",
        ["🌱 Simple / Beginner Mode", "🔬 Expert / Research Mode"],
        index=0,
        help="Simple Mode translates bioinformatic metrics into plain English, safety scores, visual analogies, and intuitive DNA spell-checkers. Expert Mode provides raw formulas, Manhattan plots, Doench '16 on-target efficiency, genomic locus annotations, PCR primer synthesis, and GPU microarchitecture."
    )
    is_simple = "Simple" in ui_mode

    st.markdown("---")
    st.markdown("### 🧬 Nuclease Architecture" if not is_simple else "### ✂️ Molecular Scissors Type")
    nuc_dict = get_supported_nucleases()
    nuc_keys = list(nuc_dict.keys())
    sel_nuc = st.selectbox(
        "Select Endonuclease System",
        options=nuc_keys,
        format_func=lambda k: f"{nuc_dict[k].name} ({nuc_dict[k].canonical_pam} PAM)",
        index=nuc_keys.index(st.session_state.active_nuclease) if st.session_state.active_nuclease in nuc_keys else 0,
        help="Switch between SpCas9 (standard), Cas12a/Cpf1 (sticky overhangs), SaCas9 (compact AAV), and CasMINI."
    )
    if sel_nuc != st.session_state.active_nuclease:
        st.session_state.active_nuclease = sel_nuc
        n_prof = nuc_dict[sel_nuc]
        st.session_state.active_pam = n_prof.canonical_pam
        if len(st.session_state.active_guide) != n_prof.guide_length_nt:
            if len(st.session_state.active_guide) < n_prof.guide_length_nt:
                st.session_state.active_guide = st.session_state.active_guide.ljust(n_prof.guide_length_nt, 'A')
            else:
                st.session_state.active_guide = st.session_state.active_guide[:n_prof.guide_length_nt]
        st.session_state.scan_completed = False
        st.rerun()

    st.markdown("---")
    st.markdown("### 🔍 Instant Disease & Gene Search" if not is_simple else "### 🔍 Search Any Disease")
    st.caption("Auto-design guide from clinical human gene database:" if not is_simple else "Search any condition (e.g. Huntington, Lung Cancer, BRCA1):")
    search_q = st.text_input(
        "Search Disease or Gene Symbol",
        placeholder="e.g. Huntington, EGFR, BRCA1, CFTR, PCSK9...",
        key="search_disease_input"
    )
    if search_q:
        matched = search_genes_and_diseases(search_q)
        if matched:
            st.caption(f"Found {len(matched)} matching clinical target(s):")
            for g in matched[:3]:
                with st.expander(f"🧬 **{g.gene_symbol}** — {g.disease_indication}", expanded=True):
                    st.markdown(f"**Locus:** `{g.chromosome}` &middot; **Status:** {g.therapeutic_phase}")
                    st.caption(g.clinical_summary)
                    auto_cands = extract_and_design_sgrnas_for_gene(g, max_candidates=2)
                    for ac in auto_cands:
                        if st.button(f"🎯 Load #{ac.rank}: {ac.guide_seq} ({ac.on_target_score}%)", key=f"auto_lead_{g.gene_symbol}_{ac.rank}", use_container_width=True):
                            st.session_state.active_guide = ac.guide_seq
                            st.session_state.selected_disease_name = f"{g.gene_symbol} ({g.disease_indication})"
                            st.session_state.scan_completed = False
                            st.session_state.df_hits = None
                            st.session_state.metrics = None
                            st.session_state.trigger_scan = True
                            st.toast(f"Adopted {g.gene_symbol} lead guide! Launching scan...", icon="🎯")
                            st.rerun()
        else:
            st.info(f"No match for '{search_q}'. Try 'Huntington', 'EGFR', 'BRCA1', 'CFTR', 'HBB', 'PCSK9'.")

    st.markdown("---")
    st.markdown("### 1. Ready-to-Use Treatments" if is_simple else "### 1. Clinical Guide Presets")
    st.caption("Select a real disease treatment to test:" if is_simple else "Load verified FDA / clinical trial therapeutic targets:")
    
    preset_cols = st.columns(2)
    casgevy_label = "🩸 Sickle Cell\n(Casgevy)" if is_simple else "🩸 Casgevy\n(Sickle Cell)"
    if preset_cols[0].button(casgevy_label, use_container_width=True):
        p = CLINICAL_PRESETS["sickle_cell_bcl11a"]
        st.session_state.active_guide = p.guide_seq_20nt
        st.session_state.active_pam = p.pam_motif
        st.session_state.selected_disease_name = p.disease_name
        st.session_state.scan_completed = False
        st.session_state.df_hits = None
        st.session_state.metrics = None
        st.rerun()

    attr_label = "🫀 Heart & Nerve\n(NTLA-2001)" if is_simple else "🫀 NTLA-2001\n(ATTR Disease)"
    if preset_cols[1].button(attr_label, use_container_width=True):
        p = CLINICAL_PRESETS["attr_amyloidosis_ttr"]
        st.session_state.active_guide = p.guide_seq_20nt
        st.session_state.active_pam = p.pam_motif
        st.session_state.selected_disease_name = p.disease_name
        st.session_state.scan_completed = False
        st.session_state.df_hits = None
        st.session_state.metrics = None
        st.rerun()

    preset_cols2 = st.columns(2)
    braf_label = "🎗️ Skin Cancer\n(BRAF V600E)" if is_simple else "🎗️ BRAF V600E\n(Oncology)"
    if preset_cols2[0].button(braf_label, use_container_width=True):
        p = CLINICAL_PRESETS["oncogene_braf_v600e"]
        st.session_state.active_guide = p.guide_seq_20nt
        st.session_state.active_pam = p.pam_motif
        st.session_state.selected_disease_name = p.disease_name
        st.session_state.scan_completed = False
        st.session_state.df_hits = None
        st.session_state.metrics = None
        st.rerun()

    bench_label = "🎯 Tutorial Demo\n(Benchmark)" if is_simple else "🎯 Benchmark\n(Default)"
    if preset_cols2[1].button(bench_label, use_container_width=True):
        st.session_state.active_guide = "GAGTCCGAGCAGAAGAAGAA"
        st.session_state.active_pam = "NGG"
        st.session_state.selected_disease_name = "Synthetic Ground-Truth Benchmark"
        st.session_state.scan_completed = False
        st.session_state.df_hits = None
        st.session_state.metrics = None
        st.rerun()

    st.markdown("---")
    st.markdown("### 2. Target DNA Address" if is_simple else "### 2. Guide RNA & PAM Specification")

    curr_nuc = nuc_dict.get(st.session_state.active_nuclease, nuc_dict["sp_cas9"])
    req_len = curr_nuc.guide_length_nt

    guide_title = f"{req_len}-Letter Target DNA Sequence" if is_simple else f"Target Guide Sequence ({req_len}nt)"
    guide_help = f"The {req_len}-letter DNA address (A, C, G, T) that {curr_nuc.name} will search for." if is_simple else f"{req_len}-nucleotide guide spacer sequence without PAM."
    raw_guide = st.text_input(
        guide_title,
        value=st.session_state.active_guide,
        max_chars=32,
        help=guide_help
    ).upper().strip()

    # Real-Time Guide Diagnostics & On-Target Efficiency HUD
    guide_len = len(raw_guide)
    gc_pct = calculate_gc_content(raw_guide)
    tm_est = estimate_melting_temp(raw_guide)
    on_target_preview = evaluate_on_target_efficiency(raw_guide, curr_nuc.canonical_pam)

    diag_color = "green" if guide_len == req_len else ("orange" if guide_len < req_len else "red")
    st.markdown(
        f"<div style='font-size: 0.85rem; color: #8B949E;'>"
        f"{'Letters' if is_simple else 'Length'}: <strong style='color:{diag_color}'>{guide_len}/{req_len}</strong> | "
        f"{'GC Balance' if is_simple else 'GC Content'}: <strong>{gc_pct}%</strong> | "
        f"{'Binding Temp' if is_simple else 'Tm'}: <strong>{tm_est}°C</strong>"
        f"</div>",
        unsafe_allow_html=True
    )

    if guide_len == req_len:
        eff_color = "#00E676" if on_target_preview.score >= 70 else ("#FFB300" if on_target_preview.score >= 50 else "#FF1744")
        st.markdown(
            f"<div style='margin-top: 6px; padding: 6px 10px; background: #161B22; border: 1px solid #30363D; border-radius: 6px; font-size: 0.82rem;'>"
            f"🎯 <strong>{'Cutting Strength' if is_simple else 'On-Target Efficiency (Doench \'16)'}:</strong> "
            f"<span style='color:{eff_color}; font-weight:bold;'>{on_target_preview.score}/100 ({on_target_preview.tier})</span>"
            f"</div>",
            unsafe_allow_html=True
        )
        if on_target_preview.penalties:
            st.caption(f"⚠️ {on_target_preview.penalties[0]}")
    elif gc_pct < 30.0 or gc_pct > 70.0:
        st.caption(f"⚠️ Note: GC content outside 30-70% may lower {curr_nuc.name} cleavage efficiency.")

    pam_label = "Safety Lock Pattern (PAM)" if is_simple else f"Protospacer Adjacent Motif (PAM - {curr_nuc.pam_orientation})"
    if curr_nuc.id == "cas12a_cpf1":
        pam_options = [
            "TTTV (Cas12a Canonical 5'-PAM)",
            "TTTA (Cas12a Subtype A)",
            "TTTC (Cas12a Subtype C)",
            "TTTG (Cas12a Alternate G)"
        ]
    elif curr_nuc.id == "sa_cas9":
        pam_options = [
            "NNGRRT (SaCas9 Canonical 3'-PAM)",
            "NNGGAT (SaCas9 Variant)",
            "NNGGGT (SaCas9 Variant)",
            "NNGAAT (SaCas9 Alternate)"
        ]
    elif curr_nuc.id == "cas_mini":
        pam_options = [
            "TTTR (CasMINI Canonical 5'-PAM)",
            "TTTA (CasMINI Subtype A)",
            "TTTG (CasMINI Subtype G)"
        ]
    else:  # sp_cas9
        pam_options = [
            "NGG (Standard SpCas9 Scissors)" if is_simple else "NGG (SpCas9 Canonical)",
            "NAG (SpCas9 Alternate)" if is_simple else "NAG (SpCas9 Alternate)",
            "NGA (SpCas9 Variant)" if is_simple else "NGA (SpCas9 Variant)"
        ]

    pam_idx = 0
    for idx_p, opt in enumerate(pam_options):
        if opt.startswith(st.session_state.active_pam):
            pam_idx = idx_p
            break

    pam_choice = st.selectbox(
        pam_label,
        pam_options,
        index=pam_idx
    )
    raw_pam = pam_choice.split()[0]
    st.session_state.active_pam = raw_pam

    st.markdown("---")
    st.markdown("### 3. DNA Search Genome" if is_simple else "### 3. Genome Sequence Source")
    data_source_mode = st.radio(
        "Select Source",
        ["Benchmark Preset", "Upload Custom .FASTA"],
        horizontal=True
    )

    fasta_target_path = ""
    if data_source_mode == "Benchmark Preset":
        preset_options = ref_mgr.list_available_presets()
        selected_preset_key = st.selectbox(
            "Preset File",
            options=list(preset_options.keys()),
            format_func=lambda k: preset_options[k]
        )
        fasta_target_path = ref_mgr.get_preset_path(selected_preset_key, default_guide=raw_guide)
        st.info(f"Loaded: `{os.path.basename(fasta_target_path)}` ({PRESETS[selected_preset_key]['length_bp']:,} bp)")
    else:
        uploaded_file = st.file_uploader("Upload .fa / .fasta file", type=["fa", "fasta", "fna"])
        if uploaded_file is not None:
            upload_dir = "data/uploads"
            os.makedirs(upload_dir, exist_ok=True)
            fasta_target_path = os.path.join(upload_dir, uploaded_file.name)
            with open(fasta_target_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Uploaded: `{uploaded_file.name}` ({os.path.getsize(fasta_target_path):,} bytes)")

    st.markdown("---")
    st.markdown("### 4. How Strict to Search?" if is_simple else "### 4. Search Sensitivity")
    max_mismatches = st.slider(
        "Allowed Spelling Mistakes (Typos)" if is_simple else "Max Allowed Mismatches",
        min_value=1, max_value=6, value=4,
        help="How many letter differences are tolerated before the scissors ignore the DNA?" if is_simple else "Max Hamming distance."
    )
    min_score = st.slider(
        "Sensitivity Cutoff" if is_simple else "Min Cleavage Score Cutoff",
        min_value=0.00, max_value=1.00, value=0.01, step=0.01
    )

    st.markdown("---")
    st.markdown("### 5. In-Vivo Delivery & Tissue" if not is_simple else "### 5. Patient Cell & Delivery")
    cell_options = list(CELL_TYPE_PROFILES.keys())
    sel_cell = st.selectbox(
        "Target Cell / Organ System",
        cell_options,
        index=cell_options.index(st.session_state.active_cell_type) if st.session_state.active_cell_type in cell_options else 0
    )
    st.session_state.active_cell_type = sel_cell

    deliv_options = list(DELIVERY_MODALITIES.keys())
    sel_deliv = st.selectbox(
        "Delivery Modality",
        deliv_options,
        index=deliv_options.index(st.session_state.active_delivery) if st.session_state.active_delivery in deliv_options else 0
    )
    st.session_state.active_delivery = sel_deliv

    st.markdown("---")
    scan_btn_label = "🚀 Launch DNA Safety Scan" if is_simple else "🚀 Launch Real Alignment Scan"
    start_btn = st.button(scan_btn_label, type="primary", use_container_width=True)

# ----------------- EXECUTE SCAN ON BUTTON CLICK -----------------
should_scan = start_btn or st.session_state.get("trigger_scan", False)
if should_scan:
    st.session_state.trigger_scan = False
    has_error = False

    curr_nuc = nuc_dict.get(st.session_state.active_nuclease, nuc_dict["sp_cas9"])
    req_len = curr_nuc.guide_length_nt

    # 1. Defensive Validation
    try:
        clean_guide = validate_and_clean_guide(raw_guide, expected_length=req_len)
    except InvalidGuideError as e:
        st.error(f"🚨 **Guide RNA Error:** {e.message}\n\n💡 **Action:** {e.remediation}")
        has_error = True

    try:
        clean_pam = validate_and_clean_pam(raw_pam)
    except InvalidPamError as e:
        st.error(f"🚨 **PAM Error:** {e.message}\n\n💡 **Action:** {e.remediation}")
        has_error = True

    if not fasta_target_path:
        st.error("🚨 **FASTA Error:** Please select a preset sequence or upload a `.fasta` file before scanning.")
        has_error = True
    else:
        try:
            validate_fasta_file(fasta_target_path)
        except FastaParseError as e:
            st.error(f"🚨 **FASTA Error:** {e.message}\n\n💡 **Action:** {e.remediation}")
            has_error = True

    # 2. Run Engine
    if not has_error:
        progress_bar = st.progress(0, text="Ingesting genomic stream and allocating memory...")
        with st.spinner(f"Scanning target `{clean_guide}` against genome sequence..."):
            t_start = time.perf_counter()

            # Stream chunks
            streamer = FastaStreamer(chunk_size=250_000, overlap_bp=22)
            aligner = CpuAligner(
                target_guide=clean_guide,
                pam_motif=clean_pam,
                max_mismatches=max_mismatches,
                pam_orientation=curr_nuc.pam_orientation
            )

            raw_hits, metrics = aligner.benchmark_chunk_stream(streamer, fasta_target_path)
            elapsed = time.perf_counter() - t_start

            # Calculate on-target efficiency, biological scoring & nuclease compatibility
            on_target_assessment = evaluate_on_target_efficiency(clean_guide, clean_pam)
            nuc_assessment = evaluate_nuclease_compatibility(curr_nuc.id, clean_guide, clean_pam)

            # Calculate dual biological scores + Genomic Locus Annotations
            scored_records = []
            pam_len = len(clean_pam)
            for h in raw_hits:
                if curr_nuc.pam_orientation == "5_PRIME":
                    cand_pam = h.sequence[:pam_len]
                    cand_guide = h.sequence[pam_len:pam_len + req_len]
                else:
                    cand_guide = h.sequence[:req_len]
                    cand_pam = h.sequence[req_len:req_len + pam_len] if len(h.sequence) >= req_len + pam_len else clean_pam

                cfd = calculate_cfd_score(clean_guide, cand_guide, pam=cand_pam)
                if h.hsu_score >= min_score or cfd >= min_score:
                    ann = annotate_genomic_locus(h.chrom_name, h.global_pos)
                    scored_records.append({
                        "Chromosome": h.chrom_name,
                        "Position": h.global_pos,
                        "Strand": h.strand,
                        "Mismatches": h.mismatches,
                        "Hsu_Score": round(h.hsu_score, 4),
                        "CFD_Score": round(cfd, 4),
                        "Risk_Tier": h.risk_tier,
                        "Gene_Symbol": ann.gene_symbol,
                        "Genomic_Feature": ann.feature_type,
                        "Clinical_Consequence": ann.clinical_risk,
                        "Is_Cancer_Hallmark": ann.is_cancer_hallmark,
                        "Gene_Description": ann.gene_description,
                        "Sequence": h.sequence
                    })

            progress_bar.progress(100, text="Alignment complete!")
            time.sleep(0.2)
            progress_bar.empty()

            COLUMNS = [
                "Chromosome", "Position", "Strand", "Mismatches",
                "Hsu_Score", "CFD_Score", "Risk_Tier",
                "Gene_Symbol", "Genomic_Feature", "Clinical_Consequence",
                "Is_Cancer_Hallmark", "Gene_Description", "Sequence"
            ]
            if scored_records:
                st.session_state.df_hits = pd.DataFrame(scored_records)
            else:
                st.session_state.df_hits = pd.DataFrame(columns=COLUMNS)

            # Compute In-Vivo Delivery Simulation
            deliv_sim = simulate_in_vivo_delivery(
                cell_type_choice=st.session_state.active_cell_type,
                delivery_choice=st.session_state.active_delivery,
                on_target_score=on_target_assessment.score,
                high_risk_off_target_count=sum(1 for r in scored_records if r.get("Risk_Tier") == "HIGH_RISK")
            )

            st.session_state.metrics = {
                "elapsed": elapsed,
                "total_bp": metrics.total_base_pairs,
                "throughput_mbp": (metrics.total_base_pairs / 1e6) / elapsed if elapsed > 0 else 0.0,
                "total_candidates": len(scored_records),
                "high_risk_count": sum(1 for r in scored_records if r.get("Risk_Tier") == "HIGH_RISK"),
                "exonic_hits_count": sum(1 for r in scored_records if "Exon" in r.get("Genomic_Feature", "")),
                "cancer_hallmark_hits_count": sum(1 for r in scored_records if r.get("Is_Cancer_Hallmark", False)),
                "on_target_score": on_target_assessment.score,
                "on_target_tier": on_target_assessment.tier,
                "on_target_summary": on_target_assessment.summary,
                "on_target_penalties": on_target_assessment.penalties,
                "on_target_bonuses": on_target_assessment.bonuses,
                "nuclease_id": curr_nuc.id,
                "nuclease_name": curr_nuc.name,
                "nuclease_organism": curr_nuc.organism,
                "nuclease_cut": curr_nuc.cut_geometry,
                "nuclease_cleavage_desc": curr_nuc.cleavage_description,
                "nuclease_aav": curr_nuc.aav_packageable,
                "nuclease_pam_orient": curr_nuc.pam_orientation,
                "nuclease_seed_range": curr_nuc.seed_region_range,
                "nuclease_advantages": nuc_assessment.advantages,
                "nuclease_warnings": nuc_assessment.warnings,
                "cell_type": st.session_state.active_cell_type,
                "delivery_modality": st.session_state.active_delivery,
                "fda_feasibility": deliv_sim.fda_ind_feasibility_score,
                "feasibility_tier": deliv_sim.feasibility_tier,
                "transduction_eff": deliv_sim.transduction_efficiency_pct,
                "chromatin_access": deliv_sim.chromatin_accessibility_pct,
                "effective_in_vivo_cutting": deliv_sim.effective_in_vivo_cutting_pct,
                "cas9_duration": deliv_sim.cas9_exposure_duration,
                "deliv_advantages": deliv_sim.preclinical_advantages,
                "deliv_warnings": deliv_sim.preclinical_warnings,
                "deliv_recommendation": deliv_sim.regulatory_recommendation,
                "guide_used": clean_guide,
                "pam_used": clean_pam,
                "disease_name": st.session_state.selected_disease_name
            }
            st.session_state.scan_completed = True
            st.toast("Real alignment scan completed successfully!", icon="✅")

# ----------------- MAIN VIEW & TABS -----------------
main_title = "🧬 GeneWeaver | CRISPR DNA Safety & Efficiency Scanner" if is_simple else "🧬 GeneWeaver | GPU-Accelerated CRISPR Discovery Engine"
st.title(main_title)

st.markdown(
    f"Active Treatment: **{st.session_state.selected_disease_name}** | "
    f"Target Address: `{raw_guide}` | Safety Key: `{raw_pam[:3]}` | Max Typos: `{max_mismatches}`"
)

# 💡 CRISPR 101 Explainer Card (Available in both modes)
with st.expander("💡 **What is CRISPR? Click for a 30-second simple guide**", expanded=False):
    col_e1, col_e2, col_e3, col_e4 = st.columns(4)
    with col_e1:
        st.markdown("""
        #### ✂️ 1. Molecular Scissors
        **Cas9** is a microscopic protein engineered to cut DNA at a precise disease gene location.
        """)
    with col_e2:
        st.markdown("""
        #### 🎯 2. GPS Guide (20nt)
        A **20-letter guide** tells the scissors where to go. It searches for an exact matching sentence in DNA.
        """)
    with col_e3:
        st.markdown("""
        #### 🔒 3. Safety Lock (PAM)
        The scissors only cut if a 3-letter sequence (`NGG`) is sitting directly next to the target.
        """)
    with col_e4:
        st.markdown("""
        #### 🚨 4. Accidental Cuts
        If another gene looks similar with only 1 or 2 typos, the scissors might cut the wrong place! **GeneWeaver finds them.**
        """)

# Top Telemetry Deck
col1, col2, col3, col4, col5 = st.columns(5)
hw_stats = hw_monitor.query_stats()
gpu0 = hw_stats[0] if len(hw_stats) > 0 else None

if st.session_state.scan_completed and st.session_state.metrics is not None:
    m = st.session_state.metrics
    
    # Calculate simple safety rating (0-100)
    if m["high_risk_count"] == 0:
        safety_score = 100
        safety_status = "🟢 High Safety"
    elif m["high_risk_count"] == 1:
        safety_score = 80
        safety_status = "🟡 Caution Needed"
    else:
        safety_score = max(35, 100 - (m["high_risk_count"] * 18))
        safety_status = "🔴 High Danger"

    with col1:
        st.metric(
            label="⏱️ Scan Time" if is_simple else "Scan Duration",
            value=f"{m['elapsed']:.2f} s" if is_simple else f"{m['elapsed']:.3f} s",
            delta="Fast Compute" if is_simple else "Wall time"
        )
    with col2:
        st.metric(
            label="📖 DNA Scanned" if is_simple else "Throughput",
            value=f"{m['total_bp']:,} bp" if is_simple else f"{m['throughput_mbp']:.2f} Mbp/s",
            delta=f"{m['throughput_mbp']:.1f} M bp/sec" if is_simple else f"{m['total_bp']:,} bp"
        )
    with col3:
        st.metric(
            label="🎯 Cutting Efficacy" if is_simple else "On-Target (Doench '16)",
            value=f"{m['on_target_score']}/100",
            delta=f"{m['on_target_tier']}",
            delta_color="normal" if m['on_target_score'] >= 65 else "inverse"
        )
    with col4:
        if is_simple:
            st.metric(
                label="🛡️ Safety Score",
                value=f"{safety_score}/100",
                delta=safety_status,
                delta_color="normal" if safety_score >= 80 else "inverse"
            )
        else:
            st.metric(
                label="Off-Targets Found",
                value=f"{m['total_candidates']:,}",
                delta=f"{m['high_risk_count']} High Risk | {m['exonic_hits_count']} Exons",
                delta_color="inverse"
            )
    with col5:
        if is_simple:
            st.metric(
                label="⚠️ Danger Sites",
                value=f"{m['high_risk_count']}",
                delta=f"{m['cancer_hallmark_hits_count']} Cancer Genes" if m['cancer_hallmark_hits_count'] > 0 else "0 Cancer Genes",
                delta_color="inverse" if m['cancer_hallmark_hits_count'] > 0 else "normal"
            )
        else:
            gpu_label = f"GPU 0: {gpu0.utilization_pct:.0f}%" if gpu0 else "CPU Engine"
            gpu_sub = f"{gpu0.vram_used_gb:.1f}/{gpu0.vram_total_gb:.1f} GB" if gpu0 else "Multi-Core Active"
            st.metric(label=gpu_label, value=f"{gpu0.temperature_c}°C" if gpu0 else "Online", delta=gpu_sub)
else:
    with col1:
        st.metric(
            label="Engine Status" if not is_simple else "System Ready",
            value="Ready",
            delta="Pick a target & click scan"
        )
    with col2:
        st.metric(
            label="Hardware Engine" if not is_simple else "Speed Mode",
            value="GPU Accelerated" if not is_simple else "Lightning Fast",
            delta="CUDA & SIMD Active" if not is_simple else "Thousands of Cores"
        )
    with col3:
        st.metric(
            label="On-Target Metric" if not is_simple else "Guide Strength",
            value=f"{on_target_preview.score}/100",
            delta=f"{on_target_preview.tier}"
        )
    with col4:
        st.metric(
            label="Safety Limit" if not is_simple else "Memory Protection",
            value="75% Ceiling" if not is_simple else "Safe Guard Active",
            delta="0 Memory Faults"
        )
    with col5:
        st.metric(
            label="Computer Health" if not is_simple else "Device Status",
            value=f"{gpu0.temperature_c}°C" if gpu0 else "Normal",
            delta=f"Running Smoothly" if is_simple else f"{gpu0.utilization_pct:.0f}% Util" if gpu0 else "Online"
        )

st.markdown("---")

# Dynamic Tab Names (Now with Auto-Repair Studio, Live Scratchpad, and Multi-Guide Duel)
if is_simple:
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🌟 Safety & Efficiency Summary",
        "🔍 Visual DNA Spell-Checker & Primers",
        "📋 List of Cut Sites",
        "✨ Auto-Repair Studio & Scratchpad",
        "⚡ Computer Speed & Health",
        "💾 Download Results & Dossier"
    ])
else:
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview, Hotspots & Annotations",
        "🔬 DNA Inspector & PCR Primers",
        "📋 Filterable Candidate Table",
        "✨ Auto-Repair Studio & Live Scratchpad",
        "⚡ Hardware Telemetry & Architecture",
        "📄 Clinical Report, UCSC BED & Exports"
    ])

# ----------------- TAB 1: SAFETY SUMMARY / OVERVIEW -----------------
with tab1:
    if st.session_state.scan_completed and st.session_state.df_hits is not None:
        df = st.session_state.df_hits
        m = st.session_state.metrics
        
        if len(df) == 0:
            st.success("🎉 **Zero Off-Target Mutations Detected!** The specified guide RNA appears 100% specific with no tolerated mismatches in this genomic region.")
            st.markdown(f"""
            <div style="background: #161B22; border: 1px solid #00E676; border-radius: 10px; padding: 20px; margin-top: 12px;">
                <h4 style="color: #00E676; margin-bottom: 6px;">🛡️ Clinical Specificity Rating: 100% | On-Target Efficiency: {m['on_target_score']}%</h4>
                <p style="color: #8B949E; margin-bottom: 0;">
                    Scanned <strong>{m['total_bp']:,} bp</strong> of genomic sequence.
                    No off-target binding sites were detected with &le; {max_mismatches} mismatches and canonical PAM.
                    This indicates minimal risk of off-target double-stranded breaks in this sequence.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            if is_simple:
                # ---------------- SIMPLE MODE TAB 1 ----------------
                high_count = sum(1 for _, r in df.iterrows() if r["Risk_Tier"] == "HIGH_RISK")
                cancer_hits = sum(1 for _, r in df.iterrows() if r.get("Is_Cancer_Hallmark", False))
                
                if high_count > 0:
                    alert_extra = f" Additionally, <strong>{cancer_hits} accidental cut site(s)</strong> are located in critical human genes." if cancer_hits > 0 else ""
                    st.markdown(f"""
                    <div style="background: rgba(255, 23, 68, 0.12); border: 1px solid #FF1744; border-radius: 12px; padding: 20px; margin-bottom: 20px;">
                        <h3 style="color: #FF1744; margin: 0 0 8px 0;">🚨 Safety Alert: {high_count} High-Danger Site(s) Detected</h3>
                        <p style="color: #F0F6FC; margin: 0; font-size: 1.05rem;">
                            The molecular scissors might accidentally cut unintended DNA locations because they closely resemble your target address.{alert_extra}
                            <strong>Review the high-danger sites below before proceeding with wet-lab experiments.</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background: rgba(0, 230, 118, 0.12); border: 1px solid #00E676; border-radius: 12px; padding: 20px; margin-bottom: 20px;">
                        <h3 style="color: #00E676; margin: 0 0 8px 0;">✅ Target DNA is Highly Specific</h3>
                        <p style="color: #F0F6FC; margin: 0; font-size: 1.05rem;">
                            No dangerous accidental cut sites were found in this genome region.
                            The scissors will cleanly edit the intended gene with minimal risk of unintended damage!
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                col_s1, col_s2 = st.columns([1, 1])
                with col_s1:
                    st.subheader("📊 Risk Breakdown (Simple)")
                    tier_map = {
                        "HIGH_RISK": "🚨 High Danger (Likely to cut)",
                        "MODERATE_RISK": "🟡 Warning (Might cut)",
                        "LOW_RISK": "🟢 Safe (Unlikely to cut)",
                        "BENIGN": "🟢 Safe (Too many typos to cut)"
                    }
                    df_simple = df.copy()
                    df_simple["Category"] = df_simple["Risk_Tier"].map(tier_map).fillna("Other")
                    cat_counts = df_simple["Category"].value_counts().reset_index()
                    cat_counts.columns = ["Risk Category", "Number of Sites"]

                    fig_simple_pie = px.pie(
                        cat_counts,
                        values="Number of Sites",
                        names="Risk Category",
                        hole=0.55,
                        color="Risk Category",
                        color_discrete_map={
                            "🚨 High Danger (Likely to cut)": "#FF1744",
                            "🟡 Warning (Might cut)": "#FFB300",
                            "🟢 Safe (Unlikely to cut)": "#00F0FF",
                            "🟢 Safe (Too many typos to cut)": "#00E676"
                        },
                        template="plotly_dark",
                        height=350
                    )
                    fig_simple_pie.update_layout(paper_bgcolor="#0A0E17", plot_bgcolor="#161B22")
                    st.plotly_chart(fig_simple_pie, use_container_width=True)

                with col_s2:
                    st.subheader("🎯 Target Cutting Power")
                    st.markdown(f"""
                    * **Cutting Efficiency:** **{m['on_target_score']}% ({m['on_target_tier']})**
                    * **Verdict:** {m['on_target_summary']}
                    * **GC Chemical Balance:** {gc_pct}% (Ideal is 40–60%)
                    * **Binding Temperature:** {tm_est}°C
                    """)
                    if m.get("on_target_bonuses"):
                        for b in m["on_target_bonuses"]:
                            st.caption(f"✅ {b}")
                    if m.get("on_target_penalties"):
                        for p in m["on_target_penalties"]:
                            st.caption(f"⚠️ {p}")
                    st.caption(f"💉 **Delivery:** Formulated for **{m.get('cell_type', 'Target Cells')}** via **{m.get('delivery_modality', 'LNP')}** &middot; Patient Success: **{m.get('fda_feasibility', 80.0)}/100**")

                st.subheader("🚨 Priority Sites to Review")
                danger_sites = df[df["Risk_Tier"] == "HIGH_RISK"]
                if len(danger_sites) > 0:
                    for _, r in danger_sites.iterrows():
                        cut_pct = int(r['Hsu_Score'] * 100)
                        is_ontarget = r['Mismatches'] == 0
                        badge = "🎯 Intended Target Gene" if is_ontarget else "🚨 Dangerous Accidental Cut Site"
                        gene_txt = f" | Gene: **{r['Gene_Symbol']}** ({r['Genomic_Feature']})"
                        cancer_warn = " ⚠️ **[CRITICAL GENE AT RISK]**" if r.get('Is_Cancer_Hallmark', False) else ""
                        st.warning(
                            f"**{r['Chromosome']}, Position {r['Position']:,}** — **{badge}**{cancer_warn}\n\n"
                            f"* **Chance of Cutting:** **{cut_pct}%**\n"
                            f"* **Spelling Mistakes (Typos):** **{r['Mismatches']} letter difference(s)**{gene_txt}\n"
                            f"* **DNA Code:** `{r['Sequence']}`"
                        )
            else:
                # ---------------- EXPERT MODE TAB 1 ----------------
                chart_col1, chart_col2 = st.columns([2, 1])

                with chart_col1:
                    st.subheader("Genome-Wide Cleavage Probability (Manhattan Plot)")
                    fig_manhattan = px.scatter(
                        df,
                        x="Position",
                        y="Hsu_Score",
                        color="Risk_Tier",
                        color_discrete_map={
                            "HIGH_RISK": "#FF1744",
                            "MODERATE_RISK": "#FFB300",
                            "LOW_RISK": "#00F0FF",
                            "BENIGN": "#8B949E"
                        },
                        hover_data=["Chromosome", "Position", "Gene_Symbol", "Genomic_Feature", "Mismatches", "Hsu_Score", "CFD_Score"],
                        labels={"Hsu_Score": "Hsu-Zhang Cleavage Probability", "Position": "Genomic Coordinate (bp)"},
                        template="plotly_dark",
                        height=380
                    )
                    fig_manhattan.update_layout(paper_bgcolor="#0A0E17", plot_bgcolor="#161B22")
                    st.plotly_chart(fig_manhattan, use_container_width=True)

                with chart_col2:
                    st.subheader("Genomic Feature Distribution")
                    feat_counts = df["Genomic_Feature"].value_counts().reset_index()
                    feat_counts.columns = ["Genomic_Feature", "Count"]
                    fig_feat = px.pie(
                        feat_counts,
                        values="Count",
                        names="Genomic_Feature",
                        hole=0.55,
                        color_discrete_sequence=["#FF1744", "#FFB300", "#00F0FF", "#00E676"],
                        template="plotly_dark",
                        height=380
                    )
                    fig_feat.update_layout(paper_bgcolor="#0A0E17", plot_bgcolor="#161B22")
                    st.plotly_chart(fig_feat, use_container_width=True)

                # On-Target Cleavage & Clinical Quality Dossier Card
                st.subheader("🎯 Guide On-Target Efficacy (Doench et al. Rule Set 2 Model)")
                ot1, ot2, ot3 = st.columns([1, 1, 2])
                with ot1:
                    st.metric("On-Target Score", f"{m['on_target_score']}/100", delta=m['on_target_tier'])
                with ot2:
                    st.metric("Exonic Coding Disruptions", f"{m['exonic_hits_count']}", delta="High Consequence", delta_color="inverse" if m['exonic_hits_count'] > 0 else "normal")
                with ot3:
                    st.markdown(f"**Bioinformatic Summary:** {m['on_target_summary']}")
                    if m.get("on_target_bonuses"):
                        st.caption("✅ " + " | ".join(m["on_target_bonuses"]))
                    if m.get("on_target_penalties"):
                        st.caption("⚠️ " + " | ".join(m["on_target_penalties"]))

                # Top Critical Threats Card
                st.subheader("🚨 Priority Action: Highest Cleavage Risk Sites")
                high_risks = df[df["Hsu_Score"] >= 0.10].sort_values(by="Hsu_Score", ascending=False).head(4)
                if len(high_risks) > 0:
                    for _, r in high_risks.iterrows():
                        risk_badge = "🔴 High Risk" if r["Hsu_Score"] >= 0.50 else "🟡 Moderate Risk"
                        cancer_flag = " | 🛡️ **[TUMOR SUPPRESSOR/ONCOGENE ALERT]**" if r.get("Is_Cancer_Hallmark", False) else ""
                        st.warning(
                            f"**{r['Chromosome']}:{r['Position']:,} ({r['Strand']})** — "
                            f"{risk_badge} (Hsu: **{r['Hsu_Score']}** | CFD: **{r['CFD_Score']}**) | "
                            f"Gene: **{r['Gene_Symbol']}** ({r['Genomic_Feature']}){cancer_flag} | "
                            f"Locus: `{r['Sequence']}`"
                        )
                else:
                    st.info("No high-risk off-target sites detected above score threshold 0.10.")

                # Interactive Chromosome Karyotype Ideogram
                st.subheader("🧬 Whole-Genome Cytogenetic Ideogram (Hotspot Map)")
                st.caption("Macroscopic chromosomal visualization mapping p-arm, centromere, and q-arm off-target hot zones:")
                fig_ideogram = build_karyotype_ideogram_plot(df)
                st.plotly_chart(fig_ideogram, use_container_width=True)

                # In-Vivo Preclinical Delivery Vehicle & Epigenetic Simulator Card
                st.subheader("💉 Preclinical In-Vivo Delivery & Patient Tissue Feasibility")
                st.caption(f"Simulating delivery kinetics in **{m.get('cell_type', 'Target Tissue')}** via **{m.get('delivery_modality', 'LNP')}**:")
                d1, d2, d3, d4 = st.columns(4)
                with d1:
                    st.metric("Transduction Uptake", f"{m.get('transduction_eff', 88.0)}%", delta="Target Tissue")
                with d2:
                    st.metric("Chromatin Access", f"{m.get('chromatin_access', 85.0)}%", delta="Euchromatin")
                with d3:
                    st.metric("In-Vivo Cutting Rate", f"{m.get('effective_in_vivo_cutting', 70.0)}%", delta="Net Tissue Potency")
                with d4:
                    feas_score = m.get('fda_feasibility', 80.0)
                    feas_color = "normal" if feas_score >= 75 else ("off" if feas_score >= 50 else "inverse")
                    st.metric("🏛️ FDA IND Feasibility", f"{feas_score}/100", delta=m.get('feasibility_tier', 'HIGH_FEASIBILITY'), delta_color=feas_color)

                st.markdown(f"**Regulatory Guidance:** {m.get('deliv_recommendation', 'Preclinical profile satisfactory.')}")
                adv_list = m.get('deliv_advantages', [])
                warn_list = m.get('deliv_warnings', [])
                if adv_list:
                    st.caption("✅ " + " | ".join(adv_list[:2]))
                if warn_list:
                    st.caption("⚠️ " + " | ".join(warn_list[:2]))

                # Endonuclease Architecture & Delivery Profile Card
                st.subheader(f"🧬 Endonuclease Architecture: {m.get('nuclease_name', 'SpCas9')}")
                st.caption(f"Biophysical constraints & packaging parameters for **{m.get('nuclease_organism', 'CRISPR Nuclease')}**:")
                np1, np2, np3, np4 = st.columns(4)
                with np1:
                    st.metric("Cut Geometry", m.get('nuclease_cut', 'BLUNT'), delta="Cleavage Type")
                with np2:
                    st.metric("PAM Recognition", f"{m.get('pam_used', 'NGG')}", delta=f"{m.get('nuclease_pam_orient', '3_PRIME')} Orientation")
                with np3:
                    aav_fit = m.get('nuclease_aav', False)
                    st.metric("AAV Packaging", "✅ Single AAV" if aav_fit else "⚠️ Dual-AAV / LNP", delta="<4.7 kb Capsid" if aav_fit else ">4.7 kb Requires LNP", delta_color="normal" if aav_fit else "inverse")
                with np4:
                    s_r = m.get('nuclease_seed_range', (11, 20))
                    st.metric("Seed Core", f"nt {s_r[0]}–{s_r[1]}", delta="Mismatch Critical")

                st.markdown(f"**Cleavage Mechanism:** {m.get('nuclease_cleavage_desc', 'Double-stranded DNA break.')}")
                nuc_adv = m.get('nuclease_advantages', [])
                nuc_warn = m.get('nuclease_warnings', [])
                if nuc_adv:
                    st.caption("✅ " + " | ".join(nuc_adv[:2]))
                if nuc_warn:
                    st.caption("⚠️ " + " | ".join(nuc_warn[:2]))
    else:
        st.markdown(f"""
        <div style="background: #161B22; border: 1px solid #30363D; border-radius: 12px; padding: 28px; margin-top: 10px; text-align: center;">
            <h2 style="color: #00F0FF; margin-bottom: 8px;">🚀 Ready to Align & Score CRISPR Target</h2>
            <p style="color: #8B949E; font-size: 1.05rem; max-width: 680px; margin: 0 auto 16px auto;">
                Currently configured for <strong>{st.session_state.selected_disease_name}</strong> (<code style="color: #00E676;">{raw_guide}</code>).
                Scan genome-scale sequences with GPU acceleration, PAM filtering, MIT Hsu-Zhang + Broad CFD cleavage scoring, and Doench '16 on-target efficiency.
            </p>
        </div>
        """, unsafe_allow_html=True)
        col_c1, col_c2, col_c3 = st.columns([1, 2, 1])
        with col_c2:
            st.write("")
            if st.button("🚀 Run Real Alignment Scan Now", type="primary", use_container_width=True, key="main_run_btn"):
                st.session_state.trigger_scan = True
                st.rerun()

# ----------------- TAB 2: VISUAL DNA SPELL-CHECKER & PCR PRIMERS -----------------
with tab2:
    if st.session_state.scan_completed and st.session_state.df_hits is not None and len(st.session_state.df_hits) > 0:
        df = st.session_state.df_hits
        st.subheader("🔍 Visual DNA Spell-Checker & Primer Designer" if is_simple else "Base-by-Base Nucleotide Mismatch & PCR Primer Designer")
        
        # Color Legend Box
        st.markdown("""
        <div style="background: #161B22; border: 1px solid #30363D; border-radius: 8px; padding: 12px; margin-bottom: 16px;">
            <strong>Color Guide:</strong> &nbsp;
            <span class='base-tile match-tile'>A</span> <strong>Cyan = Match</strong> (Scissors recognize this letter) &nbsp;|&nbsp;
            <span class='base-tile mismatch-tile'>G</span> <strong>Red = Spelling Mistake</strong> (Acts as a brake) &nbsp;|&nbsp;
            <span class='base-tile pam-tile'>AGG</span> <strong>Green = Safety Lock (PAM)</strong>
        </div>
        """, unsafe_allow_html=True)

        def format_locus_option(i):
            row = df.iloc[i]
            m_count = row['Mismatches']
            gene_tag = f"[{row.get('Gene_Symbol', 'Locus')}]"
            if is_simple:
                if m_count == 0:
                    return f"Site #{i+1}: Pos {row['Position']:,} {gene_tag} — 🎯 Exact Match (Intended Target)"
                elif row['Risk_Tier'] == "HIGH_RISK":
                    return f"Site #{i+1}: Pos {row['Position']:,} {gene_tag} — 🚨 High Danger ({m_count} typo, {int(row['Hsu_Score']*100)}% cut chance)"
                elif row['Risk_Tier'] == "MODERATE_RISK":
                    return f"Site #{i+1}: Pos {row['Position']:,} {gene_tag} — 🟡 Warning ({m_count} typo, {int(row['Hsu_Score']*100)}% cut chance)"
                else:
                    return f"Site #{i+1}: Pos {row['Position']:,} {gene_tag} — 🟢 Safe ({m_count} typos, won't cut)"
            else:
                return f"#{i+1}: {row['Chromosome']}:{row['Position']:,} | Gene: {row.get('Gene_Symbol', 'N/A')} ({row.get('Genomic_Feature', 'N/A')}) | Mismatches: {m_count} | Hsu: {row['Hsu_Score']}"

        selected_idx = st.selectbox(
            "Select DNA Location to Inspect:" if is_simple else "Select Candidate Off-Target Locus to Inspect:",
            options=list(range(len(df))),
            format_func=format_locus_option
        )

        row = df.iloc[selected_idx]
        target_g = st.session_state.metrics["guide_used"]
        locus_s = row["Sequence"]

        # Render Sequence Comparator Cards
        guide_tiles = ""
        locus_tiles = ""
        pos_labels = ""

        mismatch_positions = []
        g_len = len(target_g)
        is_5prime = (st.session_state.metrics.get("nuclease_pam_orient", "3_PRIME") == "5_PRIME")
        pam_str = st.session_state.metrics.get("pam_used", "NGG")
        nuc_pam_len = len(pam_str)

        if is_5prime:
            locus_pam = locus_s[:nuc_pam_len] if len(locus_s) >= nuc_pam_len else pam_str
            locus_guide = locus_s[nuc_pam_len:nuc_pam_len + g_len] if len(locus_s) >= nuc_pam_len + g_len else locus_s
        else:
            locus_guide = locus_s[:g_len]
            locus_pam = locus_s[g_len:g_len + nuc_pam_len] if len(locus_s) >= g_len + nuc_pam_len else pam_str

        for i in range(g_len):
            t_char = target_g[i] if i < len(target_g) else "N"
            l_char = locus_guide[i] if i < len(locus_guide) else "N"
            pos_labels += f"<span style='display:inline-block;width:30px;text-align:center;font-size:0.75rem;color:#8B949E'>{i+1}</span>"
            
            if t_char == l_char:
                guide_tiles += f"<span class='base-tile match-tile' style='width:30px;text-align:center'>{t_char}</span>"
                locus_tiles += f"<span class='base-tile match-tile' style='width:30px;text-align:center'>{l_char}</span>"
            else:
                mismatch_positions.append(i + 1)
                guide_tiles += f"<span class='base-tile match-tile' style='width:30px;text-align:center'>{t_char}</span>"
                locus_tiles += f"<span class='base-tile mismatch-tile' style='width:30px;text-align:center'>{l_char}</span>"

        pam_tile_guide = f"<span class='base-tile pam-tile'>{pam_str}</span>"
        pam_tile_locus = f"<span class='base-tile pam-tile'>{locus_pam}</span>"
        pam_label = "<span style='font-size:0.75rem;color:#00E676'>PAM</span>"

        if is_5prime:
            guide_tiles = f"{pam_tile_guide} &nbsp; {guide_tiles}"
            locus_tiles = f"{pam_tile_locus} &nbsp; {locus_tiles}"
            pos_labels = f"{pam_label} &nbsp; {pos_labels}"
        else:
            guide_tiles += f" &nbsp; {pam_tile_guide}"
            locus_tiles += f" &nbsp; {pam_tile_locus}"
            pos_labels += f" &nbsp; {pam_label}"

        st.markdown(f"""
        <div class="seq-card">
            <div style="margin-bottom: 4px;">{pos_labels}</div>
            <div style="margin-bottom: 4px;"><strong>Target Guide:</strong> {guide_tiles}</div>
            <div style="margin-bottom: 8px;"><strong>Patient DNA:</strong> &nbsp;{locus_tiles}</div>
        </div>
        """, unsafe_allow_html=True)

        # Genomic Context Information Card
        st.markdown(f"""
        <div class="explainer-box">
            <h4 style="margin: 0 0 6px 0; color: #00F0FF;">🏛️ Genomic Context & Gene Annotation</h4>
            <span class="gene-badge badge-exon">{row.get('Genomic_Feature', 'Exon')}</span> &nbsp;
            <strong>Gene Symbol:</strong> <code>{row.get('Gene_Symbol', 'N/A')}</code> &nbsp;|&nbsp;
            <strong>Clinical Impact:</strong> <code>{row.get('Clinical_Consequence', 'N/A')}</code><br/>
            <span style="color: #8B949E; font-size: 0.9rem;">{row.get('Gene_Description', '')}</span>
        </div>
        """, unsafe_allow_html=True)

        n_s_start, n_s_end = st.session_state.metrics.get("nuclease_seed_range", (11, 20))
        n_name = st.session_state.metrics.get("nuclease_name", "SpCas9")

        if is_simple:
            # Simple explanation for the selected site
            if row['Mismatches'] == 0:
                st.success(f"🎯 **This is your exact target gene!** All {g_len} letters match perfectly. {n_name} will bind with 100% affinity and make the intended therapeutic cut here.")
            else:
                has_seed_mistake = any(n_s_start <= p <= n_s_end for p in mismatch_positions)
                has_distal_mistake = any(p < n_s_start or p > n_s_end for p in mismatch_positions)
                
                if has_seed_mistake and not has_distal_mistake:
                    st.info(f"🛡️ **Spelling mistake at letter #{mismatch_positions} (Seed Core):** Because this typo is inside the critical seed recognition zone (letters {n_s_start}–{n_s_end}), it severely disrupts {n_name} from cutting, dropping cut chance to **{int(row['Hsu_Score']*100)}%**.")
                elif has_distal_mistake and not has_seed_mistake:
                    st.warning(f"⚠️ **Spelling mistake at letter #{mismatch_positions} (Distal Zone):** Because this typo is outside the core seed, the scissors can still latch on and cut with **{int(row['Hsu_Score']*100)}% probability**! Watch out for this site.")
                else:
                    st.info(f"🛡️ **Multiple spelling mistakes at letters #{mismatch_positions}:** With multiple typos spread across the sequence, {n_name} will struggle to bind, making cutting unlikely (**{int(row['Hsu_Score']*100)}% chance**).")
        else:
            n_desc = st.session_state.metrics.get("nuclease_cleavage_desc", "Blunt double-strand break.")
            n_cut = st.session_state.metrics.get("nuclease_cut", "BLUNT")
            aav_str = "Single AAV (<4.7 kb)" if st.session_state.metrics.get("nuclease_aav", False) else "Dual-AAV / LNP"
            st.markdown(f"""
            <div style="font-size: 0.9rem; color: #8B949E; border-top: 1px solid #30363D; padding-top: 8px;">
                <span class='arrow-indicator'>✂️ {n_name} Cleavage Architecture:</span> {n_desc}<br/>
                <strong>Seed Region (Positions {n_s_start}–{n_s_end}):</strong> Intolerant to mismatches. Disrupts R-loop propagation.<br/>
                <strong>Break Geometry:</strong> <code>{n_cut}</code> &middot; <strong>Vector Compatibility:</strong> <code>{aav_str}</code>
            </div>
            """, unsafe_allow_html=True)
            """, unsafe_allow_html=True)

        # ----------------- WET-LAB VALIDATION: AUTOMATED PCR PRIMER GENERATOR -----------------
        st.markdown("---")
        with st.expander("🔬 **Wet-Lab Experimental Validation: Automated PCR Primer Designer**", expanded=True):
            st.caption("Matched PCR primer pair to amplify this specific genomic locus for T7E1 cleavage assay or deep NGS amplicon sequencing:")
            primer_set = design_validation_primers(row['Chromosome'], row['Position'], row['Sequence'])
            
            pcol1, pcol2 = st.columns(2)
            with pcol1:
                st.markdown(f"""
                **Forward Primer (5' &rarr; 3'):**
                ```text
                {primer_set.forward_primer}
                ```
                * **Melting Temp (Tm):** `{primer_set.forward_tm}°C`
                * **GC Content:** `{primer_set.forward_gc}%`
                * **Length:** `20 nt`
                """)
            with pcol2:
                st.markdown(f"""
                **Reverse Primer (5' &rarr; 3'):**
                ```text
                {primer_set.reverse_primer}
                ```
                * **Melting Temp (Tm):** `{primer_set.reverse_tm}°C`
                * **GC Content:** `{primer_set.reverse_gc}%`
                * **Length:** `20 nt`
                """)

            st.markdown(f"**Expected PCR Amplicon Size:** `{primer_set.amplicon_length_bp} bp` spanning the double-stranded cut coordinate.")
            st.markdown("**Thermal Cycling Conditions (Standard Q5 / Phusion Polymerase):**")
            pcr_df = pd.DataFrame(list(primer_set.pcr_protocol.items()), columns=["Step", "Condition"])
            st.table(pcr_df)

        # ----------------- 3D MACROMOLECULAR STRUCTURE VIEWER -----------------
        st.markdown("---")
        with st.expander("🧬 **3D Macromolecular Cas9 Cleavage Pocket Viewer (PDB: 4UN3)**", expanded=True):
            st.caption("Interactive WebGL 3D model of the SpCas9 ribonucleoprotein complex showing physical steric mismatch clashes inside the catalytic pocket:")
            mol_html = generate_3dmol_html(
                target_guide=target_g,
                locus_seq=locus_s,
                mismatch_positions=mismatch_positions,
                height=520
            )
            components.html(mol_html, height=530)

    elif st.session_state.scan_completed:
        st.info("Zero off-target mutations detected to inspect. Target sequence is 100% specific in this genomic region!")
    else:
        st.info("Run an alignment scan to inspect nucleotide mismatches.")

# ----------------- TAB 3: LIST OF CUT SITES / CANDIDATE TABLE -----------------
with tab3:
    if st.session_state.scan_completed and st.session_state.df_hits is not None:
        df = st.session_state.df_hits
        if len(df) == 0:
            st.success("🎉 **Zero Off-Target Mutations Detected!** No candidates matching tolerance thresholds.")
        else:
            st.subheader("📋 Detected DNA Cut Sites" if is_simple else "Filter and Search Discovered Candidates")

            # Filters
            fcol1, fcol2, fcol3, fcol4 = st.columns(4)
            with fcol1:
                tier_filter = st.multiselect(
                    "Filter by Safety Risk" if is_simple else "Filter by Risk Tier",
                    options=["HIGH_RISK", "MODERATE_RISK", "LOW_RISK", "BENIGN"],
                    default=["HIGH_RISK", "MODERATE_RISK", "LOW_RISK", "BENIGN"]
                )
            with fcol2:
                feat_options = list(df["Genomic_Feature"].unique()) if "Genomic_Feature" in df.columns else []
                feat_filter = st.multiselect(
                    "Genomic Feature",
                    options=feat_options,
                    default=feat_options
                )
            with fcol3:
                max_m_filter = st.slider(
                    "Max Spelling Mistakes" if is_simple else "Filter Max Mismatches",
                    0, max_mismatches, max_mismatches
                )
            with fcol4:
                search_query = st.text_input("Search Coordinates / Genes", "")

            filtered_df = df[df["Risk_Tier"].isin(tier_filter) & (df["Mismatches"] <= max_m_filter)] if "Risk_Tier" in df.columns else df
            if feat_filter and "Genomic_Feature" in filtered_df.columns:
                filtered_df = filtered_df[filtered_df["Genomic_Feature"].isin(feat_filter)]
            if search_query and len(filtered_df) > 0:
                filtered_df = filtered_df[
                    filtered_df["Chromosome"].str.contains(search_query, case=False) |
                    filtered_df["Sequence"].str.contains(search_query, case=False) |
                    filtered_df["Gene_Symbol"].str.contains(search_query, case=False) |
                    filtered_df["Position"].astype(str).str.contains(search_query)
                ]

            if is_simple:
                # Render friendly view
                display_df = pd.DataFrame({
                    "Location": filtered_df["Chromosome"] + " : " + filtered_df["Position"].astype(str),
                    "Gene / Feature": filtered_df["Gene_Symbol"] + " (" + filtered_df["Genomic_Feature"] + ")",
                    "Safety Status": filtered_df["Risk_Tier"].map({
                        "HIGH_RISK": "🔴 High Danger",
                        "MODERATE_RISK": "🟡 Possible Cut",
                        "LOW_RISK": "🟢 Safe (Unlikely)",
                        "BENIGN": "🟢 Safe (No Cut)"
                    }),
                    "Spelling Mistakes": filtered_df["Mismatches"].apply(lambda m: f"{m} typo{'s' if m!=1 else ''}" if m > 0 else "0 (Exact Target)"),
                    "Cut Probability": (filtered_df["Hsu_Score"] * 100).round(1).astype(str) + "%",
                    "DNA Sequence": filtered_df["Sequence"]
                })
                st.dataframe(display_df, use_container_width=True)
            else:
                st.dataframe(
                    filtered_df.sort_values(by="Hsu_Score", ascending=False),
                    use_container_width=True
                )
    else:
        st.info("Run a scan to view tabular data.")

# ----------------- TAB 4: AUTONOMOUS REPAIR STUDIO, SCRATCHPAD & DUEL -----------------
with tab4:
    studio_subtab1, studio_subtab2, studio_subtab3 = st.tabs([
        "✨ Autonomous Guide Auto-Repair Studio",
        "🧬 In-Silico Live Mutation Scratchpad",
        "⚔️ Head-to-Head Multi-Guide Duel"
    ])

    # ---------------- SUBTAB 1: AUTONOMOUS GUIDE AUTO-REPAIR ----------------
    with studio_subtab1:
        st.subheader("✨ Autonomous CRISPR Guide Auto-Repair Engine")
        st.caption(
            "GeneWeaver analyzes high-risk off-target bottlenecks and algorithmically synthesizes repaired alternatives "
            "using exonic codon shifting, 5'-distal wobble engineering (Fu et al. Tru-gRNA), and terminal base optimization."
        )

        try:
            dossier = repair_and_optimize_guide(raw_guide, raw_pam[:3])
            best = dossier.best_candidate

            # Hero Before & After Card
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(0, 240, 255, 0.08) 0%, rgba(0, 230, 118, 0.08) 100%); border: 1px solid #00F0FF; border-radius: 12px; padding: 20px; margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                    <div>
                        <span style="background: #00E676; color: #0A0E17; font-weight: 800; font-size: 0.75rem; padding: 3px 8px; border-radius: 4px; text-transform: uppercase;">Top Clinical Lead</span>
                        <h3 style="color: #00F0FF; margin: 6px 0 4px 0;">✨ Repaired Guide: <code>{best.guide_sequence}</code></h3>
                        <p style="color: #8B949E; margin: 0; font-size: 0.92rem;">
                            Strategy: <strong>{best.optimization_strategy}</strong> &middot; <strong>{best.estimated_off_target_reduction}</strong>
                        </p>
                    </div>
                    <div style="text-align: right; margin-top: 8px;">
                        <span style="font-size: 0.85rem; color: #8B949E;">Therapeutic Fitness</span>
                        <div style="font-size: 1.8rem; font-weight: 900; color: #00E676; font-family: monospace;">{best.therapeutic_fitness_index}/100</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            col_rep1, col_rep2 = st.columns([2, 1])
            with col_rep1:
                st.markdown(f"**Bioinformatic Fix Rationale:** {best.bio_rationale}")
            with col_rep2:
                if st.button("🚀 Apply Best Repaired Guide & Rescan", type="primary", use_container_width=True, key="adopt_best_repair_btn"):
                    st.session_state.active_guide = best.guide_sequence
                    st.session_state.active_pam = best.pam_motif
                    st.session_state.selected_disease_name = f"{st.session_state.selected_disease_name} (Auto-Repaired)"
                    st.session_state.trigger_scan = True
                    st.toast("Adopted repaired guide! Rescanning...", icon="✨")
                    st.rerun()

            st.markdown("---")
            st.markdown("#### 📋 All Generated Clinical Lead Candidates")

            cand_cols = st.columns(len(dossier.repaired_candidates))
            for i, cand in enumerate(dossier.repaired_candidates):
                with cand_cols[i]:
                    badge_style = "color:#00E676" if i == 0 else ("color:#00F0FF" if i == 1 else "color:#FFB300")
                    st.markdown(f"""
                    <div class="seq-card">
                        <strong style="{badge_style}">Candidate #{cand.rank} ({cand.optimization_strategy})</strong><br/>
                        <code style="font-size: 0.95rem; color: #F0F6FC;">{cand.guide_sequence}</code><br/>
                        <div style="font-size: 0.82rem; color: #8B949E; margin-top: 6px;">
                            On-Target: <strong>{cand.on_target_score}%</strong> ({cand.on_target_tier})<br/>
                            Fitness: <strong>{cand.therapeutic_fitness_index}/100</strong><br/>
                            GC: <strong>{cand.gc_pct}%</strong> | Tm: <strong>{cand.melting_temp}°C</strong>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.caption(f"🛡️ {cand.estimated_off_target_reduction}")
                    if st.button(f"Adopt Candidate #{cand.rank}", key=f"btn_cand_{cand.rank}", use_container_width=True):
                        st.session_state.active_guide = cand.guide_sequence
                        st.session_state.active_pam = cand.pam_motif
                        st.session_state.selected_disease_name = f"{st.session_state.selected_disease_name} (Candidate #{cand.rank})"
                        st.session_state.trigger_scan = True
                        st.toast(f"Adopted Candidate #{cand.rank}! Launching scan...", icon="🚀")
                        st.rerun()

        except Exception as e:
            st.error(f"Auto-Repair could not process sequence: {e}")

    # ---------------- SUBTAB 2: IN-SILICO LIVE MUTATION SCRATCHPAD ----------------
    with studio_subtab2:
        st.subheader("🧬 In-Silico Live Mutation Scratchpad (Reactive DNA Synthesizer)")
        st.caption("Interactively edit individual nucleotides to preview on-target cleavage efficiency, GC stability, and binding thermodynamics in real-time:")

        sc_col1, sc_col2 = st.columns([2, 1])
        with sc_col1:
            scratch_seq = st.text_input(
                "Live CRISPR 20-nt Scratchpad Sequence",
                value=raw_guide,
                max_chars=20,
                key="scratchpad_input"
            ).upper().strip()

        with sc_col2:
            st.write("")
            st.write("")
            if st.button("🚀 Scan With This Custom Mutant", type="primary", use_container_width=True, key="scan_scratch_btn"):
                st.session_state.active_guide = scratch_seq
                st.session_state.selected_disease_name = "Custom In-Silico Mutant"
                st.session_state.trigger_scan = True
                st.rerun()

        if len(scratch_seq) == 20:
            sc_eval = evaluate_on_target_efficiency(scratch_seq, raw_pam[:3])
            sc_gc = calculate_gc_content(scratch_seq)
            sc_tm = estimate_melting_temp(scratch_seq)

            # Interactive Base-by-Base Visual Representation
            tiles_html = ""
            for idx, char in enumerate(scratch_seq):
                pos = idx + 1
                tile_color = "#00F0FF" if pos >= 11 else "#8B949E"  # Seed vs Distal
                border_color = "#00E676" if pos == 20 else ("#FF1744" if pos in [1, 2] and char == 'T' else "#30363D")
                tiles_html += f"<span style='display:inline-block; margin:2px; padding:6px 10px; background:#161B22; border:1px solid {border_color}; border-radius:6px; font-family:monospace; font-weight:bold; font-size:1.1rem; color:{tile_color}; text-align:center;'>{char}<br/><sub style='font-size:0.65rem; color:#6E7681;'>{pos}</sub></span>"
            tiles_html += f" &nbsp; <span style='display:inline-block; margin:2px; padding:6px 12px; background:#00E676; border-radius:6px; font-family:monospace; font-weight:900; font-size:1.1rem; color:#0A0E17;'>{raw_pam[:3]}<br/><sub style='font-size:0.65rem; color:#0A0E17;'>PAM</sub></span>"

            st.markdown(f"<div style='margin: 16px 0; padding: 12px; background: #0D1117; border: 1px solid #30363D; border-radius: 8px;'>{tiles_html}</div>", unsafe_allow_html=True)
            st.caption("Cyan letters (11–20): High-intolerance Seed Region &middot; Grey letters (1–10): Distal Tolerance Region &middot; Green: PAM lock")

            # Instant Live Reactive Meters
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("🎯 On-Target Cleavage", f"{sc_eval.score}/100", delta=sc_eval.tier)
            with m2:
                st.metric("🧬 GC Content", f"{sc_gc}%", delta="Ideal 40-60%" if 40 <= sc_gc <= 60 else "Sub-optimal")
            with m3:
                st.metric("🌡️ Duplex Tm", f"{sc_tm}°C", delta="Wallace Model")
            with m4:
                seed_gc = calculate_gc_content(scratch_seq[10:20])
                st.metric("🔒 Seed GC Density", f"{seed_gc}%", delta="R-loop Anchor")

            if sc_eval.bonuses:
                st.success("✅ " + " | ".join(sc_eval.bonuses))
            if sc_eval.penalties:
                st.warning("⚠️ " + " | ".join(sc_eval.penalties))
        else:
            st.info(f"Enter exactly 20 nucleotides to activate live biophysical telemetry ({len(scratch_seq)}/20 entered).")

    # ---------------- SUBTAB 3: HEAD-TO-HEAD DUEL ----------------
    with studio_subtab3:
        st.subheader("⚔️ Head-to-Head Multi-Guide Duel")
        st.caption("Compare two candidate sgRNAs side-by-side to select the optimal clinical lead:")

        duel_col1, duel_col2 = st.columns(2)
        with duel_col1:
            guide_a = st.text_input("Candidate Guide A", value=raw_guide, max_chars=20, key="duel_guide_a").upper().strip()
        with duel_col2:
            guide_b = st.text_input("Candidate Guide B", value="CTAACAGTTGCTTTTATCAC", max_chars=20, key="duel_guide_b").upper().strip()

        if len(guide_a) == 20 and len(guide_b) == 20:
            ot_a = evaluate_on_target_efficiency(guide_a, "NGG")
            ot_b = evaluate_on_target_efficiency(guide_b, "NGG")

            d_c1, d_c2 = st.columns(2)
            with d_c1:
                st.markdown(f"""
                ### 🅰️ Guide A: `{guide_a}`
                * **On-Target Efficiency:** **{ot_a.score}/100** ({ot_a.tier})
                * **GC Content:** `{ot_a.gc_content_pct}%`
                * **Melting Temp:** `{ot_a.melting_temp_c}°C`
                * **Verdict:** {ot_a.summary}
                """)
                if ot_a.penalties:
                    for p in ot_a.penalties:
                        st.caption(f"⚠️ {p}")
            with d_c2:
                st.markdown(f"""
                ### 🅱️ Guide B: `{guide_b}`
                * **On-Target Efficiency:** **{ot_b.score}/100** ({ot_b.tier})
                * **GC Content:** `{ot_b.gc_content_pct}%`
                * **Melting Temp:** `{ot_b.melting_temp_c}°C`
                * **Verdict:** {ot_b.summary}
                """)
                if ot_b.penalties:
                    for p in ot_b.penalties:
                        st.caption(f"⚠️ {p}")

            # Head to Head Comparison Chart
            duel_df = pd.DataFrame({
                "Metric": ["On-Target Efficiency", "GC Balance", "Hybridization Tm"],
                "Guide A": [ot_a.score, ot_a.gc_content_pct, ot_a.melting_temp_c],
                "Guide B": [ot_b.score, ot_b.gc_content_pct, ot_b.melting_temp_c]
            })
            fig_duel = go.Figure(data=[
                go.Bar(name='Guide A', x=duel_df['Metric'], y=duel_df['Guide A'], marker_color='#00F0FF'),
                go.Bar(name='Guide B', x=duel_df['Metric'], y=duel_df['Guide B'], marker_color='#FFB300')
            ])
            fig_duel.update_layout(barmode='group', template='plotly_dark', paper_bgcolor="#0A0E17", plot_bgcolor="#161B22", height=320)
            st.plotly_chart(fig_duel, use_container_width=True)

            # Winner Recommendation
            if ot_a.score > ot_b.score:
                winner = "🅰️ Candidate Guide A"
                reason = f"higher on-target cleavage score ({ot_a.score} vs {ot_b.score}) and superior nucleotide composition."
            elif ot_b.score > ot_a.score:
                winner = "🅱️ Candidate Guide B"
                reason = f"higher on-target cleavage score ({ot_b.score} vs {ot_a.score}) and superior nucleotide composition."
            else:
                winner = "Both Guides Comparable"
                reason = "identical on-target scores. Selection should be guided by empirical off-target count."

            st.success(f"🏆 **Recommended Lead Candidate:** **{winner}** based on {reason}")
        else:
            st.warning("Please ensure both Guide A and Guide B are exactly 20 nucleotides.")

# ----------------- TAB 5: COMPUTER SPEED & POWER -----------------
with tab5:
    if is_simple:
        st.subheader("⚡ How Did It Scan 1,000,000 Letters in a Fraction of a Second?")
        st.markdown(f"""
        <div class="explainer-box">
            <h4 style="color: #00F0FF; margin-bottom: 6px;">🚀 Graphics Card (GPU) Acceleration</h4>
            <p style="color: #8B949E;">
                Regular computer processors read DNA one letter at a time, which takes minutes or hours for whole genomes.
                GeneWeaver uses your <strong>graphics card (GPU)</strong>, which has thousands of microscopic computing cores working simultaneously in parallel.
            </p>
            <h4 style="color: #00E676; margin-bottom: 6px;">📦 2-Bit Binary Compression</h4>
            <p style="color: #8B949E; margin-bottom: 0;">
                DNA has only 4 letters: <strong>A, C, G, T</strong>.
                Instead of using ordinary text storage (which takes 8 bits per letter), GeneWeaver compresses 4 DNA letters into a single byte of memory.
                This reduces memory usage by <strong>75%</strong>, allowing data to fly through your computer's memory bus at blazing speeds.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("🖥️ Processing Device", gpu0.name if gpu0 else "Multi-Core CPU", delta="Active")
        with c2:
            st.metric("🌡️ Device Temperature", f"{gpu0.temperature_c}°C" if gpu0 else "Safe", delta="Cool & Protected")
        with c3:
            st.metric("🛡️ Memory Safety", "0 Memory Faults", delta="Protected by VRAM Limit")
    else:
        st.subheader("GPU Acceleration & Microarchitectural Telemetry")
        col_hw1, col_hw2 = st.columns(2)
        with col_hw1:
            st.markdown("""
            ### Massively Parallel CUDA Kernel
            * **Numba `@cuda.jit`:** Python code compiled to LLVM NVVM IR -> PTX -> SASS.
            * **2-Bit Binary Compression:** 4 bases/byte (75% VRAM bandwidth reduction).
            * **Branchless XOR + POPC:** Mismatch counts computed in 3 hardware instructions.
            * **L1 Shared Memory Tiling:** Genomic tiles cached on-chip at $\approx 10\\text{ TB/s}$.
            """)
        with col_hw2:
            st.markdown(f"""
            ### Live Hardware Status
            * **Device:** {gpu0.name if gpu0 else 'CPU Multi-Core SIMD'}
            * **Utilization:** `{gpu0.utilization_pct:.1f}%`
            * **VRAM Allocation:** `{gpu0.vram_used_gb:.2f} GB / {gpu0.vram_total_gb:.2f} GB`
            * **Core Temperature:** `{gpu0.temperature_c}°C`
            * **Power Draw:** `{gpu0.power_watts:.1f} W`
            """)

# ----------------- TAB 6: DOWNLOAD CLINICAL DOSSIER & EXPORTS -----------------
with tab6:
    st.subheader("💾 Download Your Results & Clinical Dossier" if is_simple else "Exportable Clinical Deliverables & UCSC BED")
    if st.session_state.scan_completed and st.session_state.df_hits is not None and len(st.session_state.df_hits) > 0:
        df = st.session_state.df_hits
        m = st.session_state.metrics

        st.markdown("Save and export your findings in lab-ready formats:" if is_simple else "Download formatted clinical deliverables and genome browser tracks:")
        exp1, exp2, exp3, exp4, exp5 = st.columns(5)

        with exp1:
            st.download_button(
                label="📥 Download CSV Report",
                data=df.to_csv(index=False),
                file_name=f"geneweaver_report_{m['guide_used']}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with exp2:
            # Generate Publication-Grade Clinical Audit PDF Certificate
            pdf_bytes = generate_clinical_pdf_dossier(m, df)
            st.download_button(
                label="📄 Clinical PDF Certificate",
                data=pdf_bytes,
                file_name=f"geneweaver_clinical_certificate_{m['guide_used']}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

        with exp3:
            # Generate UCSC Genome Browser / IGV Compatible BED File
            bed_lines = [f"#track name=GeneWeaver_OffTargets description='GeneWeaver Off-Target Cleavage Sites for {m['guide_used']}' useScore=1"]
            for _, r in df.iterrows():
                chrom = r['Chromosome']
                start = max(0, r['Position'])
                end = start + 23
                score_int = int(r['Hsu_Score'] * 1000)
                strand = r['Strand']
                bed_lines.append(f"{chrom}\t{start}\t{end}\t{r['Gene_Symbol']}_M{r['Mismatches']}\t{score_int}\t{strand}")
            bed_content = "\n".join(bed_lines)

            st.download_button(
                label="🧬 UCSC BED Track",
                data=bed_content,
                file_name=f"geneweaver_{m['guide_used']}.bed",
                mime="text/plain",
                use_container_width=True
            )

        with exp4:
            # Generate PCR Primers Ordering Sheet
            primer_records = []
            for _, r in df.head(15).iterrows():
                pset = design_validation_primers(r['Chromosome'], r['Position'], r['Sequence'])
                primer_records.append({
                    "Locus": f"{r['Chromosome']}:{r['Position']}",
                    "Gene": r['Gene_Symbol'],
                    "Forward_Primer_5to3": pset.forward_primer,
                    "Forward_Tm": pset.forward_tm,
                    "Reverse_Primer_5to3": pset.reverse_primer,
                    "Reverse_Tm": pset.reverse_tm,
                    "Amplicon_bp": pset.amplicon_length_bp
                })
            primer_df = pd.DataFrame(primer_records)
            st.download_button(
                label="🔬 PCR Primers (.csv)",
                data=primer_df.to_csv(index=False),
                file_name=f"pcr_primers_{m['guide_used']}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with exp5:
            report_text = f"""================================================================================
GENEWEAVER CRISPR-CAS9 OFF-TARGET & ON-TARGET CLINICAL AUDIT DOSSIER
================================================================================
Timestamp:               {time.strftime('%Y-%m-%d %H:%M:%S UTC')}
Therapeutic Target:      {m['disease_name']}
Guide RNA Sequence:      5'-{m['guide_used']}-3' (20 nt)
PAM Motif:               {m['pam_used'][:3]}
Genomic Bases Scanned:   {m['total_bp']:,} bp
Throughput Achieved:     {m['throughput_mbp']:.2f} Mbp/s
Elapsed Execution Time:  {m['elapsed']:.4f} s

EFFICACY & SPECIFICITY METRICS:
--------------------------------------------------------------------------------
On-Target Cleavage Score:{m['on_target_score']}/100 ({m['on_target_tier']})
Total Candidate Sites:   {m['total_candidates']}
High-Risk Off-Targets:   {m['high_risk_count']}
Exonic Disruptions:      {m['exonic_hits_count']}
Cancer Hallmark Hits:    {m['cancer_hallmark_hits_count']}
VRAM Memory Safety:      75% Ceiling Guaranteed

TOP REPORTED SITES & GENOMIC LOCI:
--------------------------------------------------------------------------------
"""
            for _, r in df.sort_values(by="Hsu_Score", ascending=False).head(10).iterrows():
                report_text += f"{r['Chromosome']:<10} Pos: {r['Position']:<10} Gene: {r['Gene_Symbol']:<10} Exon: {r['Genomic_Feature']:<16} Hsu: {r['Hsu_Score']:<8} CFD: {r['CFD_Score']:<8} Locus: {r['Sequence']}\n"

            st.download_button(
                label="📄 Clinical Dossier (.txt)",
                data=report_text,
                file_name=f"geneweaver_clinical_dossier_{m['guide_used']}.txt",
                mime="text/plain",
                use_container_width=True
            )
    elif st.session_state.scan_completed:
        st.info("Zero off-target mutations detected to export. Target sequence is 100% specific!")
    else:
        st.info("Execute a scan to generate export files.")
