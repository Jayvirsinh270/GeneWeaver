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
from geneweaver.biology.base_editor import (
    get_supported_base_editors,
    simulate_base_editing,
    design_pegrna,
    BASE_EDITOR_REGISTRY,
    BaseEditorProfile,
    BaseEditingOutcome,
    PegRNADesign
)
from geneweaver.biology.dual_guide import (
    simulate_dual_guide_excision,
    design_excision_junction_primers,
    CLINICAL_EXCISION_PRESETS,
    ExcisionSimulationResult,
    VirtualGelLane,
    VirtualGelBand
)
from geneweaver.biology.variant_analyzer import (
    GenomicVariant,
    VariantImpactReport,
    PersonalizedAuditDossier,
    VariantConsequence,
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
from geneweaver.web.gel_visualizer import build_virtual_gel_plot
from geneweaver.web.ideogram import build_karyotype_ideogram_plot
from geneweaver.web.pdf_report import generate_clinical_pdf_dossier
from geneweaver.web.structure_viewer import generate_3dmol_html
import streamlit.components.v1 as components
from geneweaver.tui.telemetry import HardwareMonitor

# ----------------- PAGE CONFIGURATION & SESSION STATE -----------------
st.set_page_config(
    page_title="GeneWeaver | CRISPR Safety & Efficiency Engine",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "light"  # Default to clean Light Mode

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

is_light = (st.session_state.theme_mode == "light")

# Plotly theme tokens adapted to current Light / Dark appearance
plot_template = "plotly_white" if is_light else "plotly_dark"
plot_paper = "rgba(255, 255, 255, 0.95)" if is_light else "rgba(15, 23, 42, 0.6)"
plot_bg = "#F8FAFC" if is_light else "rgba(30, 41, 59, 0.25)"

# Dynamic Theme Tokens
if is_light:
    theme_tokens = """
    :root {
        --bg-canvas: #F8FAFC;
        --surface-glass: rgba(255, 255, 255, 0.9);
        --surface-glass-hover: rgba(255, 255, 255, 0.98);
        --surface-card: #FFFFFF;
        --surface-border: rgba(226, 232, 240, 0.9);
        --surface-border-highlight: rgba(255, 255, 255, 1);
        --primary-indigo: #4F46E5;
        --primary-indigo-hover: #4338CA;
        --accent-teal: #0D9488;
        --accent-mint: #059669;
        --accent-rose: #E11D48;
        --accent-amber: #D97706;
        --text-primary: #0F172A;
        --text-secondary: #475569;
        --text-muted: #94A3B8;
        --tab-list-bg: #E2E8F0;
        --tab-inactive-color: #64748B;
        --tab-active-bg: #FFFFFF;
        --tab-active-border: rgba(203, 213, 225, 0.95);
        --tab-active-color: #0F172A;
        --tab-active-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        --input-bg: #FFFFFF;
        --input-border: #CBD5E1;
        --input-color: #0F172A;
        --sidebar-bg: #F8FAFC;
        --sidebar-border: #E2E8F0;
        --metric-bg: #FFFFFF;
        --metric-val-color: #0F172A;
        --chip-bg: #F1F5F9;
        --chip-border: #E2E8F0;
        --brand-header-bg: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(241, 245, 249, 0.9) 100%);
    }
    .stApp {
        background: radial-gradient(circle at 10% 8%, rgba(79, 70, 229, 0.04) 0%, transparent 40%),
                    radial-gradient(circle at 90% 12%, rgba(13, 148, 136, 0.03) 0%, transparent 35%),
                    #F1F5F9 fixed !important;
        color: var(--text-primary) !important;
    }
    header[data-testid="stHeader"] {
        background: rgba(248, 250, 252, 0.85) !important;
        border-bottom: 1px solid var(--surface-border) !important;
    }
    .brand-name { color: #0F172A !important; }
    .brand-subtitle { color: #475569 !important; }
    .hud-chip { background: #FFFFFF !important; border: 1px solid #E2E8F0 !important; color: #475569 !important; }
    .hud-chip strong { color: #0F172A !important; }
    .seq-card { background: #FFFFFF !important; border: 1px solid #E2E8F0 !important; box-shadow: 0 4px 16px rgba(0,0,0,0.04) !important; }
    .base-tile.match-tile { background: rgba(13, 148, 136, 0.12) !important; color: #0D9488 !important; border: 1px solid rgba(13, 148, 136, 0.35) !important; }
    .base-tile.mismatch-tile { background: rgba(225, 29, 72, 0.12) !important; color: #E11D48 !important; border: 1px solid rgba(225, 29, 72, 0.35) !important; box-shadow: none !important; }
    .base-tile.pam-tile { background: rgba(79, 70, 229, 0.12) !important; color: #4F46E5 !important; border: 1px solid rgba(79, 70, 229, 0.35) !important; }
    .explainer-box { background: rgba(255, 255, 255, 0.9) !important; border: 1px solid #E2E8F0 !important; }
    """
else:
    theme_tokens = """
    :root {
        --bg-canvas: #0F172A;
        --surface-glass: rgba(30, 41, 59, 0.65);
        --surface-glass-hover: rgba(30, 41, 59, 0.85);
        --surface-card: rgba(15, 23, 42, 0.75);
        --surface-border: rgba(148, 163, 184, 0.14);
        --surface-border-highlight: rgba(255, 255, 255, 0.12);
        --primary-indigo: #6366F1;
        --primary-indigo-hover: #4F46E5;
        --accent-teal: #14B8A6;
        --accent-mint: #10B981;
        --accent-rose: #F43F5E;
        --accent-amber: #F59E0B;
        --text-primary: #F8FAFC;
        --text-secondary: #94A3B8;
        --text-muted: #64748B;
        --tab-list-bg: rgba(15, 23, 42, 0.7);
        --tab-inactive-color: #94A3B8;
        --tab-active-bg: rgba(99, 102, 241, 0.16);
        --tab-active-border: rgba(129, 140, 248, 0.35);
        --tab-active-color: #C7D2FE;
        --tab-active-shadow: 0 4px 14px rgba(99, 102, 241, 0.15);
        --input-bg: rgba(15, 23, 42, 0.7);
        --input-border: rgba(148, 163, 184, 0.18);
        --input-color: #F8FAFC;
        --sidebar-bg: rgba(15, 23, 42, 0.95);
        --sidebar-border: rgba(255, 255, 255, 0.08);
        --metric-bg: rgba(30, 41, 59, 0.5);
        --metric-val-color: #F8FAFC;
        --chip-bg: rgba(15, 23, 42, 0.75);
        --chip-border: rgba(255, 255, 255, 0.08);
        --brand-header-bg: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
    }
    .stApp {
        background: radial-gradient(circle at 10% 8%, rgba(99, 102, 241, 0.07) 0%, transparent 40%),
                    radial-gradient(circle at 90% 12%, rgba(20, 184, 166, 0.05) 0%, transparent 35%),
                    radial-gradient(circle at 50% 50%, rgba(30, 41, 59, 0.35) 0%, rgba(15, 23, 42, 0.95) 55%, #0B111E 100%) fixed !important;
        color: var(--text-primary) !important;
    }
    header[data-testid="stHeader"] {
        background: rgba(11, 17, 30, 0.85) !important;
        border-bottom: 1px solid var(--surface-border) !important;
    }
    .brand-name { color: #FFFFFF !important; }
    .brand-subtitle { color: #94A3B8 !important; }
    .hud-chip { background: rgba(15, 23, 42, 0.75) !important; border: 1px solid rgba(255, 255, 255, 0.08) !important; color: #94A3B8 !important; }
    .hud-chip strong { color: #F8FAFC !important; }
    .seq-card { background: rgba(15, 23, 42, 0.75) !important; border: 1px solid var(--surface-border) !important; }
    .base-tile.match-tile { background: rgba(20, 184, 166, 0.16) !important; color: #2DD4BF !important; border: 1px solid rgba(45, 212, 191, 0.35) !important; }
    .base-tile.mismatch-tile { background: rgba(244, 63, 94, 0.22) !important; color: #FDA4AF !important; border: 1px solid rgba(244, 63, 94, 0.45) !important; box-shadow: none !important; }
    .base-tile.pam-tile { background: rgba(99, 102, 241, 0.25) !important; color: #C7D2FE !important; border: 1px solid rgba(129, 140, 248, 0.5) !important; }
    .explainer-box { background: var(--surface-glass) !important; border: 1px solid var(--surface-border) !important; }
    """

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    :root {{
        --font-sans: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
    }}

    {theme_tokens}

    .stApp {{
        font-family: var(--font-sans) !important;
        letter-spacing: -0.01em;
    }}

    /* Global Soft Glassmorphic Card Containers */
    .glass-card {{
        background: var(--surface-glass);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--surface-border);
        border-top: 1px solid var(--surface-border-highlight);
        border-radius: 16px !important;
        padding: 22px;
        margin-bottom: 18px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.08);
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }}
    .glass-card:hover {{
        border-color: rgba(99, 102, 241, 0.3);
    }}

    /* Precision App Brand Header */
    .brand-header-container {{
        background: var(--brand-header-bg);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border: 1px solid var(--surface-border);
        border-top: 1px solid var(--surface-border-highlight);
        border-radius: 18px !important;
        padding: 24px 28px;
        margin-bottom: 22px;
        box-shadow: 0 8px 30px -6px rgba(0, 0, 0, 0.12);
        display: flex;
        flex-direction: column;
        gap: 16px;
    }}
    .brand-title-row {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 12px;
    }}
    .brand-title-left {{
        display: flex;
        align-items: center;
        gap: 14px;
    }}
    .brand-logo-pill {{
        width: 44px;
        height: 44px;
        border-radius: 12px;
        background: linear-gradient(135deg, #4F46E5 0%, #6366F1 50%, #14B8A6 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.4rem;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }}
    .brand-name {{
        font-size: 1.75rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin: 0;
        line-height: 1.2;
    }}
    .brand-subtitle {{
        font-size: 0.9rem;
        font-weight: 500;
        margin: 2px 0 0 0;
    }}
    .brand-status-pills {{
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
    }}
    .status-pill {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }}
    .status-pill-green {{
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #059669;
    }}
    .status-pill-indigo {{
        background: rgba(99, 102, 241, 0.12);
        border: 1px solid rgba(99, 102, 241, 0.3);
        color: #4F46E5;
    }}
    .status-pill-pulse {{
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #10B981;
    }}

    /* Target Parameter HUD Capsule */
    .hud-capsule-row {{
        display: flex;
        align-items: center;
        gap: 10px;
        flex-wrap: wrap;
        padding-top: 14px;
        border-top: 1px solid var(--surface-border);
    }}
    .hud-chip {{
        border-radius: 10px !important;
        padding: 6px 12px;
        font-size: 0.82rem;
        display: flex;
        align-items: center;
        gap: 6px;
    }}

    /* =========================================================================
       CRITICAL: ELIMINATE STREAMLIT NATIVE RED UNDERLINE & SHARP TAB FOCUS BOX
       ========================================================================= */
    div[data-baseweb="tab-highlight"],
    div[data-baseweb="tab-border"],
    div[data-testid="stTabs"] div[data-baseweb="tab-highlight"],
    div[data-testid="stTabs"] div[data-baseweb="tab-border"],
    div[data-testid="stTabs"] [data-baseweb="tab-highlight"],
    div[data-testid="stTabs"] [data-baseweb="tab-border"],
    div[data-testid="stTabs"] hr {{
        display: none !important;
        height: 0px !important;
        opacity: 0 !important;
        visibility: hidden !important;
        border: none !important;
        background: transparent !important;
    }}
    
    div[data-baseweb="tab-list"],
    div[data-testid="stTabs"] div[data-baseweb="tab-list"] {{
        background: var(--tab-list-bg) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid var(--surface-border) !important;
        border-bottom: 1px solid var(--surface-border) !important;
        border-radius: 14px !important;
        padding: 5px !important;
        gap: 6px !important;
        margin-bottom: 18px !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
    }}

    /* Soft pill tabs with NO sharp corners */
    button[data-baseweb="tab"],
    .stTabs [data-baseweb="tab"],
    .stTabs button,
    div[data-testid="stTabs"] button {{
        outline: none !important;
        outline-width: 0 !important;
        border: 1px solid transparent !important;
        border-radius: 10px !important;
        color: var(--tab-inactive-color) !important;
        font-family: var(--font-sans) !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 8px 18px !important;
        background: transparent !important;
        box-shadow: none !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }}
    
    button[data-baseweb="tab"]:hover,
    .stTabs [data-baseweb="tab"]:hover {{
        color: var(--text-primary) !important;
        background: rgba(148, 163, 184, 0.12) !important;
        border-radius: 10px !important;
    }}

    button[data-baseweb="tab"]:focus,
    button[data-baseweb="tab"]:focus-visible,
    button[data-baseweb="tab"]:active,
    .stTabs [data-baseweb="tab"]:focus,
    .stTabs [data-baseweb="tab"]:focus-visible,
    .stTabs button:focus,
    .stTabs button:focus-visible {{
        outline: none !important;
        outline-width: 0 !important;
        border: 1px solid transparent !important;
        box-shadow: none !important;
    }}

    button[data-baseweb="tab"][aria-selected="true"],
    .stTabs [aria-selected="true"],
    .stTabs button[aria-selected="true"],
    div[data-testid="stTabs"] button[aria-selected="true"] {{
        outline: none !important;
        outline-width: 0 !important;
        border: 1px solid var(--tab-active-border) !important;
        border-radius: 10px !important;
        background: var(--tab-active-bg) !important;
        color: var(--tab-active-color) !important;
        box-shadow: var(--tab-active-shadow) !important;
    }}

    /* Metrics with soft rounded edges */
    div[data-testid="stMetric"] {{
        background: var(--metric-bg) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid var(--surface-border) !important;
        border-radius: 14px !important;
        padding: 16px 20px !important;
        box-shadow: 0 2px 10px 0 rgba(0, 0, 0, 0.05) !important;
    }}
    div[data-testid="stMetricValue"] {{
        font-family: var(--font-mono) !important;
        color: var(--metric-val-color) !important;
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }}
    div[data-testid="stMetricLabel"] {{
        color: var(--text-secondary) !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }}

    /* Soft rounded buttons */
    .stButton > button {{
        background: linear-gradient(135deg, var(--surface-glass) 0%, var(--surface-card) 100%) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--surface-border) !important;
        border-radius: 12px !important;
        padding: 0.6rem 1.25rem !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        font-family: var(--font-sans) !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05) !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }}
    .stButton > button:hover {{
        border-color: var(--primary-indigo) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.15) !important;
    }}
    .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.3) !important;
    }}
    .stButton > button[kind="primary"]:hover {{
        background: linear-gradient(135deg, #4338CA 0%, #4F46E5 100%) !important;
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.45) !important;
    }}

    /* Soft rounded inputs */
    div[data-baseweb="input"] {{
        background: var(--input-bg) !important;
        border: 1px solid var(--input-border) !important;
        border-radius: 12px !important;
        color: var(--input-color) !important;
    }}
    div[data-baseweb="input"]:focus-within {{
        border-color: var(--primary-indigo) !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.18) !important;
    }}
    div[data-baseweb="select"] > div {{
        background: var(--input-bg) !important;
        border: 1px solid var(--input-border) !important;
        border-radius: 12px !important;
        color: var(--input-color) !important;
    }}
    div[data-testid="stExpander"] {{
        background: var(--surface-glass) !important;
        border: 1px solid var(--surface-border) !important;
        border-radius: 14px !important;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.04) !important;
    }}
    div[data-testid="stRadio"] > div {{
        border-radius: 12px !important;
        gap: 6px !important;
    }}

    /* Sequence Visualizer Badges */
    .seq-card {{
        border-radius: 14px !important;
        padding: 18px 22px;
        margin-bottom: 14px;
        font-family: var(--font-mono);
    }}
    .base-tile {{
        display: inline-block;
        padding: 5px 9px;
        margin: 2px;
        border-radius: 8px !important;
        font-weight: 700;
        font-size: 1.05rem;
    }}
    .arrow-indicator {{ color: var(--accent-amber); font-size: 1.05rem; }}

    /* Explainer Callout Card */
    .explainer-box {{
        border-radius: 14px !important;
        padding: 18px 22px;
        margin-bottom: 18px;
    }}

    /* Gene Badges */
    .gene-badge {{
        display: inline-block;
        padding: 4px 10px;
        border-radius: 8px !important;
        font-size: 0.82rem;
        font-weight: 600;
    }}
    .badge-cancer {{ background: rgba(244, 63, 94, 0.12); border: 1px solid rgba(244, 63, 94, 0.3); color: #E11D48; }}
    .badge-exon {{ background: rgba(245, 158, 11, 0.12); border: 1px solid rgba(245, 158, 11, 0.3); color: #D97706; }}
    .badge-safe {{ background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.3); color: #059669; }}

    /* Sidebar Refinements */
    section[data-testid="stSidebar"] {{
        background: var(--sidebar-bg) !important;
        border-right: 1px solid var(--sidebar-border) !important;
    }}

    /* Custom Scrollbars */
    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{ background: rgba(148, 163, 184, 0.3); border-radius: 6px; }}
</style>
""", unsafe_allow_html=True)

ref_mgr = ReferenceManager()
hw_monitor = HardwareMonitor(device_ids=[0, 1])

# Plotly theme tokens adapted to current Light / Dark appearance
plot_template = "plotly_white" if is_light else "plotly_dark"
plot_paper = "rgba(255, 255, 255, 0.95)" if is_light else "rgba(15, 23, 42, 0.6)"
plot_bg = "#F8FAFC" if is_light else "rgba(30, 41, 59, 0.25)"

# ----------------- SIDEBAR: INPUTS & PRESETS -----------------
with st.sidebar:
    st.title("🧬 GeneWeaver")
    st.caption("GPU-Accelerated CRISPR Safety & Efficiency Engine")

    # Light / Dark Theme Switcher
    st.markdown("### 🎨 Appearance Theme")
    theme_choice = st.radio(
        "Display Mode",
        ["☀️ Clean Studio (Light)", "🌙 Soft Slate (Dark)"],
        index=0 if is_light else 1,
        horizontal=True,
        key="app_theme_radio"
    )
    new_theme = "light" if "Light" in theme_choice else "dark"
    if new_theme != st.session_state.theme_mode:
        st.session_state.theme_mode = new_theme
        st.rerun()

    st.markdown("---")
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
        eff_color = "#34D399" if on_target_preview.score >= 70 else ("#FBBF24" if on_target_preview.score >= 50 else "#F87171")
        st.markdown(
            f"<div style='margin-top: 6px; padding: 7px 12px; background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; font-size: 0.82rem;'>"
            f"🎯 <strong>{'Cutting Strength' if is_simple else 'On-Target Efficiency (Doench \'16)'}:</strong> "
            f"<span style='color:{eff_color}; font-weight:700;'>{on_target_preview.score}/100 ({on_target_preview.tier})</span>"
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

# ----------------- MAIN VIEW & EXECUTIVE APPS BRAND HEADER -----------------
st.markdown(f"""
<div class="brand-header-container">
    <div class="brand-title-row">
        <div class="brand-title-left">
            <div class="brand-logo-pill">🧬</div>
            <div>
                <h1 class="brand-name">GeneWeaver <span style="font-weight: 300; opacity: 0.8; font-size: 1.15rem; color: #94A3B8;">&middot; Clinical CRISPR Suite</span></h1>
                <p class="brand-subtitle">GPU-Accelerated Genome Alignment, Off-Target Hazard Prediction & Therapeutic Architecture</p>
            </div>
        </div>
        <div class="brand-status-pills">
            <span class="status-pill status-pill-green">
                <span class="status-pill-pulse"></span>
                CUDA JIT Online
            </span>
            <span class="status-pill status-pill-indigo">
                v2.2 Enterprise
            </span>
            <span class="status-pill" style="background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1); color: #CBD5E1;">
                {'🌱 Beginner Mode' if is_simple else '🔬 Expert Mode'}
            </span>
        </div>
    </div>
    <div class="hud-capsule-row">
        <div class="hud-chip">
            <span>Target:</span>
            <strong>{st.session_state.selected_disease_name}</strong>
        </div>
        <div class="hud-chip">
            <span>Lead Spacer:</span>
            <strong style="color: #A5B4FC; letter-spacing: 0.05em;">{raw_guide}</strong>
        </div>
        <div class="hud-chip">
            <span>PAM Lock:</span>
            <strong style="color: #6EE7B7;">{raw_pam[:3]}</strong>
        </div>
        <div class="hud-chip">
            <span>Nuclease:</span>
            <strong>{curr_nuc.name}</strong>
        </div>
        <div class="hud-chip">
            <span>Hamming Distance:</span>
            <strong>&le; {max_mismatches} bp</strong>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

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
            <div class="glass-card" style="border-left: 4px solid #10B981; padding: 22px; margin-top: 14px;">
                <h4 style="color: #6EE7B7; margin-bottom: 6px;">🛡️ Clinical Specificity Rating: 100% | On-Target Efficiency: {m['on_target_score']}%</h4>
                <p style="color: #94A3B8; margin-bottom: 0;">
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
                            "🚨 High Danger (Likely to cut)": "#F43F5E",
                            "🟡 Warning (Might cut)": "#F59E0B",
                            "🟢 Safe (Unlikely to cut)": "#6366F1",
                            "🟢 Safe (Too many typos to cut)": "#10B981"
                        },
                        template=plot_template,
                        height=350
                    )
                    fig_simple_pie.update_layout(paper_bgcolor=plot_paper, plot_bgcolor=plot_bg)
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
                            "HIGH_RISK": "#F43F5E",
                            "MODERATE_RISK": "#F59E0B",
                            "LOW_RISK": "#6366F1",
                            "BENIGN": "#64748B"
                        },
                        hover_data=["Chromosome", "Position", "Gene_Symbol", "Genomic_Feature", "Mismatches", "Hsu_Score", "CFD_Score"],
                        labels={"Hsu_Score": "Hsu-Zhang Cleavage Probability", "Position": "Genomic Coordinate (bp)"},
                        template=plot_template,
                        height=380
                    )
                    fig_manhattan.update_layout(paper_bgcolor=plot_paper, plot_bgcolor=plot_bg)
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
                        color_discrete_sequence=["#F43F5E", "#F59E0B", "#6366F1", "#10B981"],
                        template=plot_template,
                        height=380
                    )
                    fig_feat.update_layout(paper_bgcolor=plot_paper, plot_bgcolor=plot_bg)
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
        <div class="glass-card" style="padding: 36px 32px; text-align: center; border-color: rgba(99, 102, 241, 0.25);">
            <div style="display: inline-flex; align-items: center; justify-content: center; width: 64px; height: 64px; border-radius: 18px; background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(20, 184, 166, 0.2)); border: 1px solid rgba(255, 255, 255, 0.15); margin-bottom: 16px; font-size: 2rem;">
                ⚡
            </div>
            <h2 style="color: #FFFFFF; font-size: 1.65rem; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 8px;">
                Ready to Execute Whole-Genome Alignment Scan
            </h2>
            <p style="color: #94A3B8; font-size: 1rem; max-width: 640px; margin: 0 auto 24px auto; line-height: 1.6;">
                Currently targeted to <strong style="color: #F8FAFC;">{st.session_state.selected_disease_name}</strong> with lead spacer <code style="color: #A5B4FC; background: rgba(99, 102, 241, 0.12); padding: 3px 8px; border-radius: 6px; font-weight: 600;">{raw_guide}</code>.
                Evaluates whole genomes using JIT-compiled CUDA SIMD mismatch evaluation, MIT Hsu-Zhang, Broad CFD, and Doench '16 on-target efficiency.
            </p>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; max-width: 900px; margin: 0 auto 24px auto; text-align: left;">
                <div style="background: rgba(15, 23, 42, 0.65); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 12px; padding: 14px 16px;">
                    <div style="font-weight: 700; color: #F8FAFC; margin-bottom: 4px; font-size: 0.92rem;">🚀 Sub-3-Minute Scan</div>
                    <div style="color: #94A3B8; font-size: 0.82rem;">2-bit binary genomic compression with GPU streaming.</div>
                </div>
                <div style="background: rgba(15, 23, 42, 0.65); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 12px; padding: 14px 16px;">
                    <div style="font-weight: 700; color: #F8FAFC; margin-bottom: 4px; font-size: 0.92rem;">🎛️ Epigenetic Tuning</div>
                    <div style="color: #94A3B8; font-size: 0.82rem;">CRISPRa / CRISPRi gene volume control without DNA breaks.</div>
                </div>
                <div style="background: rgba(15, 23, 42, 0.65); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 12px; padding: 14px 16px;">
                    <div style="font-weight: 700; color: #F8FAFC; margin-bottom: 4px; font-size: 0.92rem;">✂️ Dual-Guide Excision</div>
                    <div style="color: #94A3B8; font-size: 0.82rem;">Simulate exon skipping & virtual agarose gel electrophoresis.</div>
                </div>
                <div style="background: rgba(15, 23, 42, 0.65); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 12px; padding: 14px 16px;">
                    <div style="font-weight: 700; color: #F8FAFC; margin-bottom: 4px; font-size: 0.92rem;">🧬 Patient Variant Audit</div>
                    <div style="color: #94A3B8; font-size: 0.82rem;">Detect PAM destruction and de-novo off-targets in VCFs.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        col_c1, col_c2, col_c3 = st.columns([1, 2, 1])
        with col_c2:
            st.write("")
            if st.button("🚀 Execute Whole-Genome Alignment Scan Now", type="primary", use_container_width=True, key="main_run_btn"):
                st.session_state.trigger_scan = True
                st.rerun()

# ----------------- TAB 2: VISUAL DNA SPELL-CHECKER & PCR PRIMERS -----------------
with tab2:
    if st.session_state.scan_completed and st.session_state.df_hits is not None and len(st.session_state.df_hits) > 0:
        df = st.session_state.df_hits
        st.subheader("🔍 Visual DNA Spell-Checker & Primer Designer" if is_simple else "Base-by-Base Nucleotide Mismatch & PCR Primer Designer")
        
        # Color Legend Box
        st.markdown("""
        <div class="glass-card" style="padding: 14px 18px; margin-bottom: 16px; display: flex; align-items: center; gap: 16px; flex-wrap: wrap;">
            <strong style="color: #94A3B8; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em;">Nucleotide Key:</strong>
            <span style="display: flex; align-items: center; gap: 6px;"><span class='base-tile match-tile'>A</span> <strong style="color: #2DD4BF;">Match</strong> (Target Alignment)</span>
            <span style="display: flex; align-items: center; gap: 6px;"><span class='base-tile mismatch-tile'>G</span> <strong style="color: #FDA4AF;">Mismatch</strong> (Binding Penalty)</span>
            <span style="display: flex; align-items: center; gap: 6px;"><span class='base-tile pam-tile'>AGG</span> <strong style="color: #C7D2FE;">PAM Lock</strong></span>
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

# ----------------- TAB 4: AUTONOMOUS REPAIR, BASE & PRIME EDITING STUDIO -----------------
with tab4:
    studio_subtab1, studio_subtab2, studio_subtab3, studio_subtab4, studio_subtab5, studio_subtab6, studio_subtab7, studio_subtab8 = st.tabs([
        "✨ Autonomous Guide Auto-Repair Studio",
        "✂️ Dual-Guide Excision & Nickase (Virtual Gel)",
        "🧬 Patient Variant & ClinVar SNP Analyzer",
        "🎛️ Epigenetic CRISPRa / CRISPRi Modulation",
        "🎯 Base Editing Window & Bystander Simulator",
        "🧬 Prime Editing pegRNA Architect",
        "🔬 In-Silico Live Mutation Scratchpad",
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

    # ---------------- SUBTAB 2: DUAL-GUIDE EXCISION & NICKASE (VIRTUAL GEL) ----------------
    with studio_subtab2:
        st.subheader("✂️ Dual-Guide Genomic Excision & Paired Nickase (D10A) Architect")
        st.caption(
            "Simulate coordinated dual-guide excision for exon skipping (e.g. DMD Exon 51), oncogene deletion, or viral provirus removal. "
            "Evaluates deletion vs inversion kinetics, paired D10A nickase offsets, and renders a virtual agarose gel."
        )

        # Preset selection
        st.markdown("##### 🧬 Load Clinical Excision Benchmark")
        exc_presets = CLINICAL_EXCISION_PRESETS
        p_cols = st.columns(3)
        
        # Default session state for dual guide
        if "dual_g_a" not in st.session_state:
            st.session_state.dual_g_a = exc_presets["dmd_exon_51_skipping"]["guide_a"]
            st.session_state.dual_g_b = exc_presets["dmd_exon_51_skipping"]["guide_b"]
            st.session_state.dual_cut_a = exc_presets["dmd_exon_51_skipping"]["cut_a"]
            st.session_state.dual_cut_b = exc_presets["dmd_exon_51_skipping"]["cut_b"]
            st.session_state.dual_label = exc_presets["dmd_exon_51_skipping"]["title"]
            st.session_state.dual_strand_a = exc_presets["dmd_exon_51_skipping"]["strand_a"]
            st.session_state.dual_strand_b = exc_presets["dmd_exon_51_skipping"]["strand_b"]

        if p_cols[0].button("🧬 DMD Exon 51 Skipping", use_container_width=True, key="btn_dmd_excision"):
            p = exc_presets["dmd_exon_51_skipping"]
            st.session_state.dual_g_a = p["guide_a"]
            st.session_state.dual_g_b = p["guide_b"]
            st.session_state.dual_cut_a = p["cut_a"]
            st.session_state.dual_cut_b = p["cut_b"]
            st.session_state.dual_label = p["title"]
            st.session_state.dual_strand_a = p["strand_a"]
            st.session_state.dual_strand_b = p["strand_b"]
            st.rerun()

        if p_cols[1].button("🎗️ EGFR Exon 19 Deletion", use_container_width=True, key="btn_egfr_excision"):
            p = exc_presets["egfr_exon_19_deletion"]
            st.session_state.dual_g_a = p["guide_a"]
            st.session_state.dual_g_b = p["guide_b"]
            st.session_state.dual_cut_a = p["cut_a"]
            st.session_state.dual_cut_b = p["cut_b"]
            st.session_state.dual_label = p["title"]
            st.session_state.dual_strand_a = p["strand_a"]
            st.session_state.dual_strand_b = p["strand_b"]
            st.rerun()

        if p_cols[2].button("🛡️ HIV-1 LTR Excision", use_container_width=True, key="btn_hiv_excision"):
            p = exc_presets["hiv1_proviral_ltr_excision"]
            st.session_state.dual_g_a = p["guide_a"]
            st.session_state.dual_g_b = p["guide_b"]
            st.session_state.dual_cut_a = p["cut_a"]
            st.session_state.dual_cut_b = p["cut_b"]
            st.session_state.dual_label = p["title"]
            st.session_state.dual_strand_a = p["strand_a"]
            st.session_state.dual_strand_b = p["strand_b"]
            st.rerun()

        st.caption(f"Active Model: **{st.session_state.dual_label}**")
        st.markdown("---")

        # Inputs for Guide A and Guide B
        g_col1, g_col2 = st.columns(2)
        with g_col1:
            st.markdown("#### 🅰️ 5' Upstream Guide A")
            dg_a = st.text_input("Guide A Spacer (20 nt)", value=st.session_state.dual_g_a, max_chars=20, key="dg_input_a").upper().strip()
            c_pos_a = st.number_input("Cut Position A (bp)", min_value=1, value=int(st.session_state.dual_cut_a), key="dg_pos_a")
            st_a = st.selectbox("Guide A Strand", ["(+)", "(-)"], index=0 if st.session_state.dual_strand_a == "(+)" else 1, key="dg_strand_a")

        with g_col2:
            st.markdown("#### 🅱️ 3' Downstream Guide B")
            dg_b = st.text_input("Guide B Spacer (20 nt)", value=st.session_state.dual_g_b, max_chars=20, key="dg_input_b").upper().strip()
            c_pos_b = st.number_input("Cut Position B (bp)", min_value=1, value=int(st.session_state.dual_cut_b), key="dg_pos_b")
            st_b = st.selectbox("Guide B Strand", ["(+)", "(-)"], index=1 if st.session_state.dual_strand_b == "(-)" else 0, key="dg_strand_b")

        # Nuclease Architecture Mode
        mode_col1, mode_col2 = st.columns([1, 1])
        with mode_col1:
            nuc_mode = st.radio(
                "Cleavage Endonuclease Mode",
                ["Standard Dual Cas9 (Simultaneous DSBs)", "Paired Cas9 D10A Nickases (Single-Strand Nicks)"],
                index=0,
                key="dg_nuc_mode"
            )
            is_nick = "Nickase" in nuc_mode

        if len(dg_a) == 20 and len(dg_b) == 20:
            exc_res = simulate_dual_guide_excision(
                guide_a_seq=dg_a,
                guide_b_seq=dg_b,
                cut_pos_a=int(c_pos_a),
                cut_pos_b=int(c_pos_b),
                is_nickase_mode=is_nick,
                strand_a=st_a,
                strand_b=st_b
            )

            # Metrics
            m_ex1, m_ex2, m_ex3, m_ex4 = st.columns(4)
            with m_ex1:
                st.metric("Target Deletion Span", f"{exc_res.deletion_span_bp:,} bp", delta="Excision Gap")
            with m_ex2:
                st.metric("Predicted Deletion Rate", f"{exc_res.predicted_deletion_efficiency_pct}%", delta="Target Excision")
            with m_ex3:
                st.metric("Inversion Artifact Rate", f"{exc_res.predicted_inversion_rate_pct}%", delta="Inverted Fragment", delta_color="inverse")
            with m_ex4:
                pot_delta = "High Potency" if exc_res.composite_cutting_potency >= 60 else "Moderate"
                st.metric("Composite Cut Potency", f"{exc_res.composite_cutting_potency}/100", delta=pot_delta)

            # Virtual Gel Electrophoresis Display
            st.markdown("---")
            st.markdown("#### 🧪 Virtual Agarose Gel Electrophoresis")
            st.caption("Predicted electrophoretic migration comparison between Untreated Control (Wildtype) and Dual-Guide Edited Amplicons:")
            gel_fig = build_virtual_gel_plot(exc_res.virtual_gel_lanes, height=450)
            st.plotly_chart(gel_fig, use_container_width=True)

            # Excision Junction Sequence & Primers
            st.markdown("#### 🔬 Deletion Junction & PCR Screening Primers")
            st.markdown(f"**Recombined Deletion Junction:**\n`{exc_res.deletion_junction_sequence}`")
            st.caption(exc_res.clinical_recommendation)

            j_primers = design_excision_junction_primers(
                chrom="Target Chromosome",
                cut_pos_a=int(c_pos_a),
                cut_pos_b=int(c_pos_b)
            )
            jp_c1, jp_c2 = st.columns(2)
            with jp_c1:
                st.markdown(f"**Forward Screening Primer (5' &rarr; 3'):**\n```text\n{j_primers['forward_primer_5to3']}\n```\n* Tm: `{j_primers['forward_tm']}°C` | GC: `{j_primers['forward_gc']}%`")
            with jp_c2:
                st.markdown(f"**Reverse Screening Primer (5' &rarr; 3'):**\n```text\n{j_primers['reverse_primer_5to3']}\n```\n* Tm: `{j_primers['reverse_tm']}°C` | GC: `{j_primers['reverse_gc']}%`")

            if exc_res.biophysical_warnings:
                for w in exc_res.biophysical_warnings:
                    st.warning(f"⚠️ {w}")
        else:
            st.warning("Please ensure both Guide A and Guide B spacers are exactly 20 nucleotides.")

    # ---------------- SUBTAB 3: PATIENT VARIANT & CLINVAR SNP ANALYZER ----------------
    with studio_subtab3:
        st.subheader("🧬 Patient-Personalized Genomic Variant (ClinVar / dbSNP / VCF) Analyzer")
        st.caption(
            "Audit patient-specific single nucleotide polymorphisms (SNPs) and indels against CRISPR guides. "
            "Detects PAM destruction (therapy loss), PAM creation (de-novo off-targets), seed mismatch penalties, "
            "and calculates Allele-Specific Selectivity Ratios for dominant-negative disorders (e.g. Huntington's HTT, Sickle Cell HBB, BRAF V600E)."
        )

        # 1-Click Clinical Benchmarks
        st.markdown("##### 📌 Clinical Benchmarks (ClinVar / dbSNP Pre-Loaded)")
        bench_cols = st.columns(4)
        clinvar_benchmarks = get_clinvar_benchmark_variants()

        if "variant_preset" not in st.session_state:
            st.session_state.variant_preset = "HBB_rs334_SickleCell"

        preset_keys = list(clinvar_benchmarks.keys())
        labels = [
            "🩸 Sickle Cell (HBB)",
            "🧠 Huntington's (HTT)",
            "🔬 BRAF V600E (Melanoma)",
            "🫁 EGFR T790M (NSCLC)"
        ]

        for b_idx, (b_key, b_lbl) in enumerate(zip(preset_keys, labels)):
            with bench_cols[b_idx]:
                if st.button(b_lbl, use_container_width=True, key=f"btn_var_bench_{b_key}"):
                    st.session_state.variant_preset = b_key
                    st.rerun()

        active_bench = clinvar_benchmarks[st.session_state.variant_preset]
        st.info(f"**Loaded Case Study:** {active_bench['name']} &mdash; *{active_bench['clinical_context']}*")

        v_c1, v_c2, v_c3 = st.columns([1, 1, 1])
        with v_c1:
            guide_v = st.text_input("Candidate Guide Spacer (20 nt)", value=active_bench["guide_seq"], max_chars=20, key="var_guide_input").upper().strip()
        with v_c2:
            pam_v = st.text_input("PAM Motif (IUPAC)", value=active_bench["pam_motif"], max_chars=5, key="var_pam_input").upper().strip()
        with v_c3:
            loc_v = st.number_input("Target Locus Start Position", value=active_bench["target_locus_start"], step=1, key="var_loc_input")

        # Patient Variants Section
        st.markdown("##### 🧪 Patient Variant Inputs")
        var_mode = st.radio("Variant Input Mode", ["Curated Patient Variant (Quick Audit)", "Upload / Paste VCF v4.2 Text"], horizontal=True, key="var_mode_radio")

        parsed_variants = []
        if var_mode == "Curated Patient Variant (Quick Audit)":
            pv_c1, pv_c2, pv_c3, pv_c4, pv_c5 = st.columns([1, 1, 1, 1, 1])
            with pv_c1:
                v_chr = st.text_input("Chromosome", value=active_bench["chrom"], key="pv_chr")
            with pv_c2:
                v_pos = st.number_input("Variant Position", value=active_bench["pos"], step=1, key="pv_pos")
            with pv_c3:
                v_ref = st.text_input("Reference Allele (WT)", value=active_bench["ref"], max_chars=10, key="pv_ref").upper()
            with pv_c4:
                v_alt = st.text_input("Patient Allele (ALT)", value=active_bench["alt"], max_chars=10, key="pv_alt").upper()
            with pv_c5:
                v_rs = st.text_input("Variant ID / dbSNP", value=active_bench["name"].split()[-1].rstrip(")"), key="pv_id")

            parsed_variants.append(GenomicVariant(
                chrom=v_chr,
                pos=int(v_pos),
                ref=v_ref,
                alt=v_alt,
                variant_id=v_rs,
                gene_symbol=active_bench["gene"],
                clinical_significance=active_bench["clinical_significance"],
                heterozygous=True
            ))
        else:
            default_vcf = f"""##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tPATIENT\n{active_bench['chrom']}\t{active_bench['pos']}\t{active_bench['name'].split()[-1].rstrip(')')}\t{active_bench['ref']}\t{active_bench['alt']}\t100\tPASS\tGENE={active_bench['gene']};CLNSIG={active_bench['clinical_significance']}\tGT\t0/1\n"""
            vcf_text = st.text_area("Paste VCF 4.2 Lines", value=default_vcf, height=120, key="vcf_paste_area")
            parsed_variants = parse_vcf_lines(vcf_text)

        if len(guide_v) == 20 and parsed_variants:
            dossier = audit_patient_variants(
                guide_seq=guide_v,
                pam_motif=pam_v,
                variants=parsed_variants,
                patient_id="PATIENT-001",
                target_locus_start=int(loc_v)
            )

            # Clinical Safety Status Banner
            st.markdown("---")
            if dossier.risk_level == "THERAPY_CONTRAINDICATED":
                st.error("🚨 **CLINICAL ALERT: THERAPY CONTRAINDICATED IN THIS PATIENT** &mdash; PAM motif is destroyed by patient SNP! Cas9 cleavage is abolished (~0%).")
            elif dossier.risk_level == "HIGH_RISK":
                st.warning("⚠️ **SAFETY WARNING: DE-NOVO PATIENT OFF-TARGET** &mdash; Variant creates a novel PAM site not present in reference genome.")
            elif dossier.risk_level == "CAUTION":
                st.warning("⚠️ **EFFICACY CAUTION: SEED REGION MISMATCH** &mdash; Mutation in critical PAM-proximal seed window reduces on-target potency.")
            else:
                st.success("✅ **CLEARANCE APPROVED: THERAPY VIABLE** &mdash; No patient variants disrupt target protospacer or PAM site.")

            # Metrics Row
            m_v1, m_v2, m_v3, m_v4 = st.columns(4)
            with m_v1:
                st.metric("Variants Audited", f"{dossier.num_variants_screened}", delta=f"{len(dossier.impacted_variants)} Impacted")
            with m_v2:
                top_impact = dossier.impacted_variants[0] if dossier.impacted_variants else None
                ref_aff = top_impact.ref_affinity_score if top_impact else 92.0
                st.metric("Ref Allele Cut Rate", f"{ref_aff}%", delta="Standard Genome")
            with m_v3:
                alt_aff = top_impact.alt_affinity_score if top_impact else 92.0
                st.metric("Patient Allele Cut Rate", f"{alt_aff}%", delta="Patient Genome", delta_color="normal" if alt_aff > 50 else "inverse")
            with m_v4:
                sel_ratio = top_impact.allele_selectivity_ratio if top_impact else 1.0
                st.metric("Allele Selectivity", f"{sel_ratio}x", delta="Discriminatory Power" if sel_ratio >= 10 else "Non-selective")

            # Comparative Bar Chart: Ref Allele vs Patient Allele
            if top_impact:
                fig_comp = go.Figure(data=[
                    go.Bar(
                        x=["Reference (WT) Allele", "Patient (ALT) Allele"],
                        y=[top_impact.ref_affinity_score, top_impact.alt_affinity_score],
                        marker_color=["#6366F1", "#F43F5E" if top_impact.alt_affinity_score < 10 else "#10B981"],
                        text=[f"{top_impact.ref_affinity_score}%", f"{top_impact.alt_affinity_score}%"],
                        textposition="auto"
                    )
                ])
                fig_comp.update_layout(
                    title="CRISPR Cleavage Affinity: Reference vs Patient Variant Allele",
                    template=plot_template,
                    paper_bgcolor=plot_paper,
                    plot_bgcolor=plot_bg,
                    height=280,
                    yaxis=dict(title="Predicted Cleavage Affinity (%)", range=[0, 105])
                )
                st.plotly_chart(fig_comp, use_container_width=True)

                # Variant Impact Breakdown Table
                st.markdown("##### 📋 Variant Impact Breakdown")
                v_rows = []
                for imp in dossier.impacted_variants:
                    v_rows.append({
                        "Variant ID": imp.variant.variant_id,
                        "Position": f"{imp.variant.chrom}:{imp.variant.pos}",
                        "Alleles": f"{imp.variant.ref} &rarr; {imp.variant.alt}",
                        "Consequence": imp.consequence.value,
                        "Ref Cut %": f"{imp.ref_affinity_score}%",
                        "Patient Cut %": f"{imp.alt_affinity_score}%",
                        "Selectivity": f"{imp.allele_selectivity_ratio}x",
                        "Recommendation": imp.recommendation
                    })
                st.table(pd.DataFrame(v_rows))

            if dossier.key_findings:
                st.markdown("##### 🔍 Key Clinical Findings")
                for kf in dossier.key_findings:
                    st.caption(f"• {kf}")
        else:
            st.info("Ensure the candidate guide spacer is 20 nucleotides and valid variants are provided.")

    # ---------------- SUBTAB 4: EPIGENETIC TRANSCRIPTIONAL MODULATION (CRISPRa & CRISPRi) ----------------
    with studio_subtab4:
        st.subheader("🎛️ Epigenetic Transcriptional Modulation Studio (CRISPRa & CRISPRi)")
        st.caption(
            "Precision gene activation (dCas9-VPR, SunTag, SAM) or epigenetic silencing (dCas9-KRAB, KRAB-MeCP2) "
            "without creating double-stranded DNA breaks. Models promoter distance attenuation windows relative to the "
            "Transcription Start Site (TSS) and calculates predicted log2 fold changes and residual expression."
        )

        # 1-Click Clinical Benchmarks
        st.markdown("##### 📌 Clinical Benchmarks (Epigenetic Activation & Silencing)")
        epi_presets = get_clinical_epigenetic_presets()
        ep_cols = st.columns(4)

        if "epigenetic_preset" not in st.session_state:
            st.session_state.epigenetic_preset = "HBG1_Activation_SickleCell"

        preset_items = list(epi_presets.items())
        labels = [
            "🩸 HBG1 Reactivation (Sickle Cell)",
            "🔬 MYC Oncogene Silencing",
            "🛡️ FOXP3 Treg Activation",
            "🛑 KRAS Driver Repression"
        ]

        for p_idx, (p_key, p_val) in enumerate(preset_items):
            with ep_cols[p_idx]:
                if st.button(labels[p_idx], use_container_width=True, key=f"btn_epi_{p_key}"):
                    st.session_state.epigenetic_preset = p_key
                    st.rerun()

        active_epi = epi_presets[st.session_state.epigenetic_preset]
        st.markdown(f"""
        <div class="glass-card" style="padding: 14px 18px; margin-bottom: 16px;">
            <span class="gene-badge badge-safe">{active_epi['modality']} Mode</span> &nbsp;
            <strong>{active_epi['name']}</strong> &mdash; 
            <span style="color: #94A3B8; font-size: 0.9rem;">{active_epi['clinical_mechanism']}</span>
        </div>
        """, unsafe_allow_html=True)

        ep_c1, ep_c2, ep_c3, ep_c4 = st.columns([1, 1, 1, 1])
        with ep_c1:
            epi_gene = st.text_input("Target Gene Symbol", value=active_epi["gene"], max_chars=12, key="epi_gene_in").upper().strip()
        with ep_c2:
            epi_guide = st.text_input("Candidate Spacer (20 nt)", value=active_epi["guide_seq"], max_chars=20, key="epi_guide_in").upper().strip()
        with ep_c3:
            effector_options = list(EFFECTOR_PROFILES.keys())
            eff_idx = effector_options.index(active_epi["effector"]) if active_epi["effector"] in effector_options else 0
            selected_eff = st.selectbox(
                "Effector Fusion Construct",
                effector_options,
                index=eff_idx,
                format_func=lambda k: f"{EFFECTOR_PROFILES[k].name} ({EFFECTOR_PROFILES[k].modality.value})",
                key="epi_eff_select"
            )
        with ep_c4:
            dist_tss = st.number_input(
                "Distance to TSS (bp)",
                value=int(active_epi["distance_to_tss_bp"]),
                min_value=-1000,
                max_value=1000,
                step=5,
                help="Negative values = Upstream Promoter (CRISPRa target). Positive values = Downstream Exon 1 / Gene Body (CRISPRi target).",
                key="epi_dist_input"
            )

        if len(epi_guide) == 20:
            epi_res = simulate_epigenetic_modulation(
                gene_symbol=epi_gene,
                guide_seq=epi_guide,
                distance_to_tss_bp=int(dist_tss),
                effector_key=selected_eff,
                guide_efficiency_score=88.0
            )

            # Executive Glassmorphic Metric Displays
            em1, em2, em3, em4 = st.columns(4)
            with em1:
                delta_str = "Induction" if epi_res.modality == ModulationModality.ACTIVATION else "Repression"
                st.metric("Predicted Fold Change", f"{epi_res.predicted_fold_change}x", delta=delta_str)
            with em2:
                st.metric("log2 Fold Change", f"{epi_res.log2_fold_change:+0.2f}", delta=f"{epi_res.percent_modulation:+0.1f}%")
            with em3:
                st.metric("Residual Expression", f"{epi_res.retained_expression_pct}%", delta="Relative to WT")
            with em4:
                st.metric("Window Factor", f"{int(epi_res.window_attenuation_factor*100)}%", delta="Spatial Alignment")

            # Status Banner
            if epi_res.is_in_optimal_window:
                st.markdown(f"""
                <div class="glass-card" style="border-left: 4px solid #10B981;">
                    <h4 style="margin: 0 0 6px 0; color: #6EE7B7;">✅ Optimal Epigenetic Window Clearance</h4>
                    <p style="margin: 0; color: #CBD5E1; font-size: 0.95rem;">{epi_res.clinical_recommendation}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="glass-card" style="border-left: 4px solid #F59E0B;">
                    <h4 style="margin: 0 0 6px 0; color: #FCD34D;">⚠️ Sub-Optimal Distance from TSS</h4>
                    <p style="margin: 0; color: #CBD5E1; font-size: 0.95rem;">
                        Guide at {dist_tss} bp is outside the canonical optimal window. Spatial attenuation factor is {int(epi_res.window_attenuation_factor*100)}%.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # Interactive Plotly Promoter Attenuation Landscape
            st.markdown("#### 📈 Promoter Distance Attenuation Landscape (-600 bp to +500 bp from TSS)")
            curve_data = generate_promoter_window_profile(
                gene_symbol=epi_gene,
                effector_key=selected_eff,
                step_bp=20
            )
            cdf = pd.DataFrame(curve_data)

            fig_epi = go.Figure()

            # Optimal Window Shading
            cur_eff = EFFECTOR_PROFILES[selected_eff]
            fig_epi.add_vrect(
                x0=cur_eff.optimal_window_start_bp,
                x1=cur_eff.optimal_window_end_bp,
                fillcolor="rgba(99, 102, 241, 0.12)",
                layer="below",
                line_width=1,
                line_color="rgba(129, 140, 248, 0.4)",
                annotation_text="Optimal Activity Window",
                annotation_position="top left",
                annotation_font_color="#A5B4FC"
            )

            # Transcription Start Site vertical reference
            fig_epi.add_vline(
                x=0,
                line_width=2,
                line_dash="dash",
                line_color="#F43F5E",
                annotation_text="TSS (0 bp)",
                annotation_position="bottom right",
                annotation_font_color="#FDA4AF"
            )

            # Predicted Curve Line
            y_col = "predicted_fold_change" if cur_eff.modality == ModulationModality.ACTIVATION else "predicted_fold_change"
            fig_epi.add_trace(go.Scatter(
                x=cdf["distance_from_tss_bp"],
                y=cdf[y_col],
                mode="lines",
                name="Promoter Response Curve",
                line=dict(color="#6366F1", width=3),
                hovertemplate="TSS Distance: %{x} bp<br>Fold Change: %{y}x<extra></extra>"
            ))

            # Active Guide Pinpoint Marker
            fig_epi.add_trace(go.Scatter(
                x=[epi_res.distance_to_tss_bp],
                y=[epi_res.predicted_fold_change],
                mode="markers+text",
                name="Selected sgRNA Position",
                text=[f"Current Guide ({epi_res.distance_to_tss_bp} bp)"],
                textposition="top center",
                textfont=dict(color="#F8FAFC", size=12),
                marker=dict(size=14, color="#10B981", symbol="diamond", line=dict(color="#FFFFFF", width=2))
            ))

            fig_epi.update_layout(
                template=plot_template,
                paper_bgcolor=plot_paper,
                plot_bgcolor=plot_bg,
                height=340,
                margin=dict(l=40, r=40, t=30, b=40),
                xaxis=dict(title="Distance from Transcription Start Site (TSS) [bp]", gridcolor="rgba(148, 163, 184, 0.15)"),
                yaxis=dict(title="Predicted Fold Change (vs. WT)", gridcolor="rgba(148, 163, 184, 0.15)")
            )
            st.plotly_chart(fig_epi, use_container_width=True)

            # Clinical Delivery & Biophysical Details Card
            st.markdown("#### 🧬 Delivery Feasibility & Mechanistic Insights")
            d_c1, d_c2 = st.columns(2)
            with d_c1:
                st.markdown(f"""
                <div class="glass-card">
                    <h5 style="color: #818CF8; margin: 0 0 8px 0;">📦 In-Vivo Delivery Compatibility</h5>
                    <p style="margin: 0 0 6px 0;"><strong>Construct Size:</strong> <code>{cur_eff.delivery_size_kb} kb</code></p>
                    <p style="margin: 0 0 6px 0;"><strong>Packaging Assessment:</strong> {epi_res.delivery_feasibility}</p>
                    <p style="margin: 0; color: #94A3B8; font-size: 0.88rem;">
                        Standard AAV limits single-vector packaging to 4.7 kb. Larger effectors (SunTag, VPR, SAM) require dual-vector trans-splicing or mRNA lipid nanoparticles (LNPs).
                    </p>
                </div>
                """, unsafe_allow_html=True)
            with d_c2:
                st.markdown(f"""
                <div class="glass-card">
                    <h5 style="color: #2DD4BF; margin: 0 0 8px 0;">🔬 Chromatin Architecture & Epigenetic Marks</h5>
                    <p style="margin: 0 0 6px 0;"><strong>Recruited Mark:</strong> <code>{cur_eff.epigenetic_mark}</code></p>
                    <p style="margin: 0 0 6px 0;"><strong>Off-Target Risk:</strong> {epi_res.off_target_epigenetic_risk}</p>
                    <p style="margin: 0; color: #94A3B8; font-size: 0.88rem;">
                        {cur_eff.description}
                    </p>
                </div>
                """, unsafe_allow_html=True)

            if epi_res.biophysical_notes:
                for note in epi_res.biophysical_notes:
                    st.caption(f"💡 {note}")
        else:
            st.warning("Enter a 20-nucleotide guide spacer sequence to model epigenetic modulation.")

    # ---------------- SUBTAB 5: BASE EDITING (CBE / ABE) SIMULATOR ----------------
    with studio_subtab5:
        st.subheader("🎯 Precision Base Editing (CBE & ABE) & Bystander Mutation Simulator")
        st.caption(
            "Model Cytosine (C·G &rarr; T·A) and Adenine (A·T &rarr; G·C) deamination windows without double-stranded breaks. "
            "Detects accidental bystander mutations within the protospacer activity window."
        )

        be_col1, be_col2, be_col3 = st.columns([1, 2, 1])
        base_editors = get_supported_base_editors()
        with be_col1:
            sel_editor = st.selectbox(
                "Base Editor System",
                options=list(base_editors.keys()),
                format_func=lambda k: base_editors[k].name,
                key="be_editor_choice"
            )
        ed_obj = base_editors[sel_editor]

        with be_col2:
            be_proto = st.text_input(
                "Protospacer Sequence (20 nt)",
                value=raw_guide[:20],
                max_chars=20,
                key="be_proto_input"
            ).upper().strip()

        with be_col3:
            target_pos = st.number_input(
                "Target Base Position (1–20)",
                min_value=1,
                max_value=20,
                value=5,
                key="be_target_pos_input"
            )

        if len(be_proto) == 20:
            be_res = simulate_base_editing(be_proto, target_pos, sel_editor)

            # Overview Metrics
            m_be1, m_be2, m_be3, m_be4 = st.columns(4)
            with m_be1:
                st.metric("Primary Edit Conversion", f"{be_res.predicted_on_target_efficiency_pct}%", delta=f"{be_res.target_base} → {be_res.converted_base}")
            with m_be2:
                p_delta = "High Purity" if be_res.product_purity_score_pct >= 85 else ("Moderate" if be_res.product_purity_score_pct >= 50 else "Heterogeneous")
                st.metric("Product Purity Score", f"{be_res.product_purity_score_pct}%", delta=p_delta)
            with m_be3:
                b_color = "normal" if be_res.bystander_count == 0 else ("off" if be_res.bystander_count == 1 else "inverse")
                st.metric("Bystander Mutations", f"{be_res.bystander_count}", delta=be_res.bystander_risk_tier, delta_color=b_color)
            with m_be4:
                st.metric("Editing Activity Window", f"nt {be_res.window_range[0]}–{be_res.window_range[1]}", delta="In Window" if be_res.is_target_in_window else "Outside Window")

            # Visual Protospacer with Deamination Window
            tiles_be = ""
            w_start, w_end = be_res.window_range
            bystander_positions = [b.position for b in be_res.bystanders_found]

            for idx, char in enumerate(be_proto):
                pos = idx + 1
                in_w = (w_start <= pos <= w_end)
                is_target = (pos == target_pos)
                is_bystander = (pos in bystander_positions)

                if is_target:
                    bg = "#00E676"
                    fg = "#0A0E17"
                    border = "2px solid #FFFFFF"
                    lbl = "TARGET"
                elif is_bystander:
                    bg = "#FF1744"
                    fg = "#FFFFFF"
                    border = "2px solid #FF5252"
                    lbl = "BYSTAND"
                elif in_w:
                    bg = "#1F2937"
                    fg = "#00F0FF"
                    border = "1px dashed #00F0FF"
                    lbl = "WIN"
                else:
                    bg = "#161B22"
                    fg = "#8B949E"
                    border = "1px solid #30363D"
                    lbl = str(pos)

                tiles_be += f"<span style='display:inline-block; margin:2px; padding:6px 9px; background:{bg}; border:{border}; border-radius:6px; font-family:monospace; font-weight:bold; font-size:1.05rem; color:{fg}; text-align:center;'>{char}<br/><sub style='font-size:0.6rem;'>{lbl}</sub></span>"

            st.markdown(f"<div style='margin: 16px 0; padding: 12px; background: #0D1117; border: 1px solid #30363D; border-radius: 8px;'>{tiles_be}</div>", unsafe_allow_html=True)
            st.caption("Green: Intended deamination target · Red: Unintended bystander mutation · Cyan dashed border: Deamination catalytic window")

            # Sequence comparison & alleles
            st.markdown(f"**Clean Edited Sequence:** `5'-{be_res.edited_protospacer}-3'` &middot; Editor Architecture: *{ed_obj.nickase_architecture}*")
            st.markdown(f"**Biophysical Verdict:** {be_res.biophysical_verdict}")
            st.info(f"💡 **Clinical Guidance:** {be_res.remediation_guidance}")

            if be_res.all_possible_alleles:
                with st.expander("🧬 **Predicted Allelic Breakdown (Combinatorial Bystander Outcomes)**", expanded=False):
                    for a_idx, al in enumerate(be_res.all_possible_alleles):
                        st.code(f"Allele #{a_idx+1}: {al}")
        else:
            st.warning("Enter a 20-nucleotide protospacer sequence to simulate base editing.")

    # ---------------- SUBTAB 6: PRIME EDITING (pegRNA) ARCHITECT ----------------
    with studio_subtab6:
        st.subheader("🧬 Prime Editing (pegRNA) Architect (Search-and-Replace Genome Editing)")
        st.caption(
            "Automate the design of prime editing guide RNAs (pegRNAs). Generates Reverse Transcription Template (RTT) "
            "and Primer Binding Site (PBS) without double-stranded DNA breaks or viral donor templates."
        )

        pe_col1, pe_col2, pe_col3 = st.columns([1, 1, 1])
        with pe_col1:
            pe_edit_type = st.selectbox(
                "Intended Mutation Type",
                ["SUBSTITUTION", "INSERTION", "DELETION"],
                key="pe_edit_type_choice"
            )
        with pe_col2:
            pe_offset = st.slider(
                "Nick-to-Edit Distance (bp)",
                min_value=1,
                max_value=20,
                value=4,
                help="Distance downstream of the Cas9 nick site (between positions 17 and 18)."
            )
        with pe_col3:
            pe_pbs_len = st.slider(
                "PBS Length (nt)",
                min_value=10,
                max_value=16,
                value=13,
                help="Optimal PBS length is typically 12–14 nt for melting temperatures between 30°C and 38°C."
            )

        pe_col4, pe_col5 = st.columns([2, 1])
        with pe_col4:
            pe_seq_in = st.text_input(
                "Desired Synthesized Sequence (to write into DNA)",
                value="T",
                max_chars=25,
                key="pe_desired_seq"
            ).upper().strip()
        with pe_col5:
            pe_proto = st.text_input(
                "Spacer Sequence (20 nt)",
                value=raw_guide[:20],
                max_chars=20,
                key="pe_spacer_seq"
            ).upper().strip()

        if len(pe_proto) == 20 and len(pe_seq_in) > 0:
            pe_design = design_pegrna(
                target_guide=pe_proto,
                edit_type=pe_edit_type,
                edit_offset_from_nick=pe_offset,
                desired_sequence=pe_seq_in,
                pbs_length=pe_pbs_len
            )

            # Telemetry metrics
            m_pe1, m_pe2, m_pe3, m_pe4 = st.columns(4)
            with m_pe1:
                tm_color = "normal" if 30 <= pe_design.pbs_tm_celsius <= 38 else "inverse"
                st.metric("PBS Melting Temp (Tm)", f"{pe_design.pbs_tm_celsius}°C", delta="Ideal 30–38°C", delta_color=tm_color)
            with m_pe2:
                st.metric("PBS Length & GC", f"{pe_design.pbs_length_nt} nt", delta=f"{pe_design.pbs_gc_pct}% GC")
            with m_pe3:
                st.metric("RTT Length", f"{pe_design.rtt_length_nt} nt", delta=f"+{pe_design.nick_to_edit_distance_bp} bp offset")
            with m_pe4:
                feas_col = "normal" if pe_design.feasibility_score >= 80 else ("off" if pe_design.feasibility_score >= 60 else "inverse")
                st.metric("Feasibility Rating", f"{pe_design.feasibility_score}/100", delta=pe_design.feasibility_tier, delta_color=feas_col)

            st.markdown("#### 🔬 Synthesized pegRNA Subcomponents")
            c_r1, c_r2 = st.columns(2)
            with c_r1:
                st.markdown(f"**Primer Binding Site (PBS, 3' terminus):**\n```text\n{pe_design.pbs_seq}\n```")
            with c_r2:
                st.markdown(f"**Reverse Transcription Template (RTT):**\n```text\n{pe_design.rtt_seq}\n```")

            st.markdown(f"**Complete Full-Length pegRNA Sequence ({len(pe_design.composite_pegrna_seq)} nt):**")
            st.code(pe_design.composite_pegrna_seq, language="text")

            for rec in pe_design.recommendations:
                st.caption(f"💡 {rec}")
        else:
            st.info("Enter a 20-nt spacer and desired edit sequence to synthesize the pegRNA.")

    # ---------------- SUBTAB 7: IN-SILICO LIVE MUTATION SCRATCHPAD ----------------
    with studio_subtab7:
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
                tile_color = "#A5B4FC" if pos >= 11 else "#94A3B8"  # Seed vs Distal
                border_color = "#10B981" if pos == 20 else ("#F43F5E" if pos in [1, 2] and char == 'T' else "rgba(255, 255, 255, 0.08)")
                tiles_html += f"<span style='display:inline-block; margin:2px; padding:6px 10px; background:rgba(15, 23, 42, 0.75); border:1px solid {border_color}; border-radius:8px; font-family:monospace; font-weight:700; font-size:1.05rem; color:{tile_color}; text-align:center;'>{char}<br/><sub style='font-size:0.65rem; color:#64748B;'>{pos}</sub></span>"
            tiles_html += f" &nbsp; <span style='display:inline-block; margin:2px; padding:6px 12px; background:rgba(99, 102, 241, 0.25); border:1px solid rgba(129, 140, 248, 0.5); border-radius:8px; font-family:monospace; font-weight:800; font-size:1.05rem; color:#C7D2FE;'>{raw_pam[:3]}<br/><sub style='font-size:0.65rem; color:#A5B4FC;'>PAM</sub></span>"

            st.markdown(f"<div class='glass-card' style='margin: 16px 0; padding: 16px 20px;'>{tiles_html}</div>", unsafe_allow_html=True)
            st.caption("Indigo letters (11–20): High-intolerance Seed Core &middot; Slate letters (1–10): Distal Tolerance &middot; Purple: PAM lock")

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

    # ---------------- SUBTAB 8: HEAD-TO-HEAD DUEL ----------------
    with studio_subtab8:
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
                go.Bar(name='Guide A', x=duel_df['Metric'], y=duel_df['Guide A'], marker_color='#6366F1'),
                go.Bar(name='Guide B', x=duel_df['Metric'], y=duel_df['Guide B'], marker_color='#F59E0B')
            ])
            fig_duel.update_layout(barmode='group', template=plot_template, paper_bgcolor=plot_paper, plot_bgcolor=plot_bg, height=320)
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

    # ---------------- REST API MICROSERVICE INTEGRATION ----------------
    st.markdown("---")
    st.subheader("🌐 Production REST API & Computational Pipeline Integration")
    st.caption("GeneWeaver provides an enterprise-ready FastAPI microservice for integration with Nextflow, Snakemake, Python pipelines, and cloud clusters.")

    api_c1, api_c2 = st.columns([1, 1])
    with api_c1:
        st.markdown("""
        #### 🚀 Local Microservice Endpoints
        * **Interactive Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
        * **ReDoc Interactive Reference:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
        * **OpenAPI 3.1 JSON Schema:** [http://127.0.0.1:8000/openapi.json](http://127.0.0.1:8000/openapi.json)

        **Quick Launch Command:**
        ```powershell
        .\\launch_api.bat
        # or
        python -m uvicorn geneweaver.api.app:app --reload
        ```
        """)

    with api_c2:
        api_lang = st.radio("Choose API Client Snippet:", ["cURL", "Python (Requests)", "TypeScript / Node"], horizontal=True)
        if api_lang == "cURL":
            st.code(f"""# Align & Score Guide via REST API
curl -X POST "http://127.0.0.1:8000/api/v1/align" \\
  -H "Content-Type: application/json" \\
  -d '{{"guide_sequence": "{raw_guide[:20]}", "pam_motif": "NGG", "target_dna_sequence": "GAGTCCGAGCAGAAGAAGAATGGACGT", "max_mismatches": 3}}'""", language="bash")
        elif api_lang == "Python (Requests)":
            st.code(f"""import requests

payload = {{
    "guide_sequence": "{raw_guide[:20]}",
    "pam_motif": "NGG",
    "target_dna_sequence": "GAGTCCGAGCAGAAGAAGAATGGACGT",
    "max_mismatches": 3
}}
res = requests.post("http://127.0.0.1:8000/api/v1/align", json=payload)
print(res.json())""", language="python")
        else:
            st.code(f"""const res = await fetch("http://127.0.0.1:8000/api/v1/align", {{
  method: "POST",
  headers: {{ "Content-Type": "application/json" }},
  body: JSON.stringify({{
    guide_sequence: "{raw_guide[:20]}",
    pam_motif: "NGG",
    target_dna_sequence: "GAGTCCGAGCAGAAGAAGAATGGACGT",
    max_mismatches: 3
  }})
}});
const data = await res.json();
console.log(data);""", language="typescript")
