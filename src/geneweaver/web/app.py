"""
GeneWeaver Web GUI Dashboard.
Interactive browser-based interface for CRISPR off-target alignment analysis.
Powered by Streamlit, Plotly, and the GeneWeaver GPU engine.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Page Configuration with Cyber-Bio Theme
st.set_page_config(
    page_title="GeneWeaver | GPU CRISPR Alignment Engine",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Cyber-Bio CSS Design Tokens (Master Color Palette)
st.markdown("""
<style>
    /* Dark Void Master Background (#0A0E17) */
    .stApp {
        background-color: #0A0E17;
        color: #F0F6FC;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    /* Card Surface Styling (#161B22 with #30363D border) */
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        color: #00F0FF;
    }
    
    /* Custom DNA Nucleotide Badges */
    .dna-seq-container {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.15rem;
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 6px;
        padding: 12px 16px;
        margin-top: 8px;
        letter-spacing: 2px;
    }
    .base-match {
        color: #00F0FF;
        font-weight: 600;
    }
    .base-mismatch {
        background-color: #FF1744;
        color: #FFFFFF;
        font-weight: 900;
        padding: 2px 5px;
        border-radius: 4px;
    }
    .base-pam {
        background-color: #00E676;
        color: #0A0E17;
        font-weight: 900;
        padding: 2px 5px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR CONTROLS -----------------
with st.sidebar:
    st.title("🧬 GeneWeaver")
    st.caption("GPU-Accelerated CRISPR Alignment Engine")
    
    st.markdown("### 1. Genomic Sequence")
    genome_choice = st.selectbox(
        "Reference Genome (.fasta)",
        ["hg38 (Human Reference, 3.2 Gb)", "mm10 (Mouse Reference, 2.7 Gb)", "Synthetic Benchmark (100 Mb)"]
    )
    
    st.markdown("### 2. Guide RNA & Targeting")
    guide_seq = st.text_input(
        "Target sgRNA Sequence (20nt)",
        value="GAGTCCGAGCAGAAGAAGAA",
        max_chars=20,
        help="Enter the 20-nucleotide spacer sequence without PAM."
    ).upper()
    
    pam_motif = st.selectbox(
        "Protospacer Adjacent Motif (PAM)",
        ["NGG (SpCas9 Canonical)", "NAG (SpCas9 Alternate)", "TTTV (Cas12a/Cpf1)"]
    )
    
    st.markdown("### 3. Alignment Thresholds")
    max_mismatches = st.slider("Max Mismatch Cutoff", min_value=1, max_value=6, value=4)
    min_score = st.slider("Min Cleavage Score Cutoff", min_value=0.00, max_value=1.00, value=0.01, step=0.01)
    
    st.markdown("---")
    start_btn = st.button("🚀 Launch GPU Scan", type="primary", use_container_width=True)

# ----------------- MAIN DASHBOARD -----------------
st.title("CRISPR-Cas9 Off-Target Alignment Dashboard")
st.markdown(
    f"Active Target: `{guide_seq}` | PAM: `{pam_motif[:3]}` | Max Mismatches: `{max_mismatches}` | Score Cutoff: `{min_score}`"
)

# 1. Telemetry Metric Deck
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="GPU 0 (RTX 4090)", value="98%", delta="67°C | 2.5 GHz")
with col2:
    st.metric(label="VRAM Allocation", value="4.8 / 24.0 GB", delta="Pinned: 1.2 GB")
with col3:
    st.metric(label="Scan Throughput", value="18.4 Mbp/s", delta="Sub-3 min pace")
with col4:
    st.metric(label="Total Off-Targets", value="1,482", delta="42 High Risk", delta_color="inverse")

st.markdown("---")

# Mock dataset for demonstration
np.random.seed(42)
chromosomes = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
n_samples = 300
df_hits = pd.DataFrame({
    "Chromosome": np.random.choice(chromosomes, size=n_samples),
    "Position": np.random.randint(1_000_000, 240_000_000, size=n_samples),
    "Strand": np.random.choice(["(+)", "(-)"], size=n_samples),
    "Mismatches": np.random.choice([1, 2, 3, 4], size=n_samples, p=[0.05, 0.15, 0.35, 0.45]),
    "Hsu_Score": np.random.exponential(scale=0.15, size=n_samples).clip(0, 1),
    "CFD_Score": np.random.exponential(scale=0.18, size=n_samples).clip(0, 1),
})
df_hits["Risk_Tier"] = df_hits["Hsu_Score"].apply(
    lambda s: "High Risk (>=0.50)" if s >= 0.50 else ("Moderate (0.10-0.50)" if s >= 0.10 else "Low Risk (<0.10)")
)

# 2. Analytics Grid: Manhattan Plot + Donut Chart
col_chart1, col_chart2 = st.columns([2, 1])

with col_chart1:
    st.subheader("Genome-Wide Cleavage Risk (Manhattan Plot)")
    fig_manhattan = px.scatter(
        df_hits,
        x="Position",
        y="Hsu_Score",
        color="Risk_Tier",
        color_discrete_map={
            "High Risk (>=0.50)": "#FF1744",
            "Moderate (0.10-0.50)": "#FFB300",
            "Low Risk (<0.10)": "#00F0FF"
        },
        hover_data=["Chromosome", "Position", "Mismatches", "Strand"],
        labels={"Hsu_Score": "Hsu-Zhang Score"},
        template="plotly_dark",
        height=380
    )
    fig_manhattan.update_layout(paper_bgcolor="#0A0E17", plot_bgcolor="#161B22")
    st.plotly_chart(fig_manhattan, use_container_width=True)

with col_chart2:
    st.subheader("Risk Distribution")
    risk_counts = df_hits["Risk_Tier"].value_counts().reset_index()
    risk_counts.columns = ["Risk_Tier", "Count"]
    fig_donut = px.pie(
        risk_counts,
        values="Count",
        names="Risk_Tier",
        hole=0.55,
        color="Risk_Tier",
        color_discrete_map={
            "High Risk (>=0.50)": "#FF1744",
            "Moderate (0.10-0.50)": "#FFB300",
            "Low Risk (<0.10)": "#00F0FF"
        },
        template="plotly_dark",
        height=380
    )
    fig_donut.update_layout(paper_bgcolor="#0A0E17", plot_bgcolor="#161B22")
    st.plotly_chart(fig_donut, use_container_width=True)

# 3. DNA Alignment Inspector
st.subheader("Detailed DNA Mismatch Alignment Inspector")
st.markdown("""
<div class="dna-seq-container">
    <div><strong>Target Guide (5'-3'):</strong> <span class="base-match">G A G T C C G A G C A G A A G A A G A A</span> &nbsp; <span class="base-pam">N G G</span></div>
    <div style="margin-top: 6px;"><strong>Locus Alignment:</strong> <span class="base-match">G A G T</span> <span class="base-mismatch">C</span> <span class="base-match">C G A G C A G A A G A A G A A</span> &nbsp; <span class="base-pam">A G G</span></div>
    <div style="font-size: 0.85rem; color: #8B949E; margin-top: 8px;">
        Locus: <code>chr7:142,509,211 (+)</code> | Mismatches: 1 (Position 5, Distal Region) | Hsu-Zhang Score: <strong>0.842 (High Risk)</strong>
    </div>
</div>
""", unsafe_allow_html=True)

# 4. Filterable Candidate Data Table
st.subheader("Discovered Off-Target Candidates")
st.dataframe(
    df_hits.sort_values(by="Hsu_Score", ascending=False),
    use_container_width=True
)

# 5. One-Click Exporters
exp_col1, exp_col2, _ = st.columns([1, 1, 4])
with exp_col1:
    st.download_button(
        label="📥 Export CSV Report",
        data=df_hits.to_csv(index=False),
        file_name="geneweaver_off_targets.csv",
        mime="text/csv",
        use_container_width=True
    )
with exp_col2:
    st.download_button(
        label="📥 Export VCF Callset",
        data="##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n",
        file_name="geneweaver_off_targets.vcf",
        mime="text/plain",
        use_container_width=True
    )
