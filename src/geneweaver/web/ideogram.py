"""
Chromosome Karyotype Ideogram & Chromosomal Hotspot Visualizer.
Builds an interactive cytogenetic map of genomic coordinates and highlights
off-target cleavage hotspots across chromosomal arms (p-arm, centromere, q-arm).
"""

from typing import Dict, List, Optional
import pandas as pd
import plotly.graph_objects as go


# Standard chromosome length and centromere approximations (GRCh38 / synthetic mapping)
CHROM_CENTROMERES: Dict[str, Dict[str, float]] = {
    "chr1":  {"len": 248.9, "cent": 123.4},
    "chr2":  {"len": 242.1, "cent": 93.9},
    "chr3":  {"len": 198.2, "cent": 90.9},
    "chr4":  {"len": 190.2, "cent": 50.0},
    "chr5":  {"len": 181.5, "cent": 48.8},
    "chr6":  {"len": 170.8, "cent": 59.8},
    "chr7":  {"len": 159.3, "cent": 60.1},
    "chr8":  {"len": 145.1, "cent": 45.2},
    "chr9":  {"len": 138.3, "cent": 43.0},
    "chr10": {"len": 133.7, "cent": 39.8},
    "chr11": {"len": 135.0, "cent": 53.4},
    "chr12": {"len": 133.2, "cent": 35.5},
    "chr13": {"len": 114.3, "cent": 17.7},
    "chr14": {"len": 107.0, "cent": 17.2},
    "chr15": {"len": 101.9, "cent": 19.0},
    "chr16": {"len": 90.3,  "cent": 36.8},
    "chr17": {"len": 83.2,  "cent": 25.1},
    "chr18": {"len": 80.3,  "cent": 18.5},
    "chr19": {"len": 58.6,  "cent": 26.2},
    "chr20": {"len": 64.4,  "cent": 28.1},
    "chr21": {"len": 46.7,  "cent": 12.0},
    "chr22": {"len": 50.8,  "cent": 15.0},
    "chrX":  {"len": 156.0, "cent": 61.0},
    "chrY":  {"len": 57.2,  "cent": 10.4},
}


def build_karyotype_ideogram_plot(df_hits: pd.DataFrame, height: int = 420) -> go.Figure:
    """
    Creates an interactive chromosome ideogram plot displaying chromosomes
    and pinpointing off-target loci as glowing hot spots.
    """
    fig = go.Figure()

    if df_hits is None or len(df_hits) == 0:
        fig.add_annotation(
            text="No off-target hits detected to plot on ideogram.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color="#8B949E", size=14)
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0A0E17",
            plot_bgcolor="#161B22",
            height=height
        )
        return fig

    # Find chromosomes present in hits
    active_chroms = list(df_hits["Chromosome"].unique())
    # Sort chromosomes naturally
    def chrom_sort_key(c):
        num = c.replace("chr", "").replace("Chr", "").strip()
        if num.isdigit():
            return int(num)
        return 99

    sorted_chroms = sorted(active_chroms, key=chrom_sort_key)

    # Plot chromosome backbone bars
    for idx, chrom in enumerate(sorted_chroms):
        info = CHROM_CENTROMERES.get(chrom.lower(), {"len": 150.0, "cent": 60.0})
        total_len = info["len"]
        cent = info["cent"]

        y_pos = idx * 2.0

        # Draw p-arm (short arm)
        fig.add_trace(go.Bar(
            y=[y_pos],
            x=[cent],
            base=0,
            orientation='h',
            marker=dict(
                color='#1E2638',
                line=dict(color='#3B82F6', width=1.5)
            ),
            hoverinfo='text',
            hovertext=f"{chrom} p-arm (0 - {cent:.1f} Mb)",
            showlegend=False
        ))

        # Draw Centromere marker
        fig.add_trace(go.Bar(
            y=[y_pos],
            x=[3.0],
            base=cent,
            orientation='h',
            marker=dict(
                color='#64748B',
                line=dict(color='#94A3B8', width=1)
            ),
            hoverinfo='text',
            hovertext=f"{chrom} Centromere (~{cent:.1f} Mb)",
            showlegend=False
        ))

        # Draw q-arm (long arm)
        q_len = total_len - (cent + 3.0)
        fig.add_trace(go.Bar(
            y=[y_pos],
            x=[q_len],
            base=cent + 3.0,
            orientation='h',
            marker=dict(
                color='#1E2638',
                line=dict(color='#3B82F6', width=1.5)
            ),
            hoverinfo='text',
            hovertext=f"{chrom} q-arm ({cent+3:.1f} - {total_len:.1f} Mb)",
            showlegend=False
        ))

    # Overlay off-target hits as pinpoint scatter markers
    color_map = {
        "HIGH_RISK": "#FF1744",
        "MODERATE_RISK": "#FFB300",
        "LOW_RISK": "#00F0FF",
        "BENIGN": "#00E676"
    }

    for tier, color in color_map.items():
        sub_df = df_hits[df_hits["Risk_Tier"] == tier]
        if len(sub_df) == 0:
            continue

        x_coords = []
        y_coords = []
        hover_texts = []

        for _, r in sub_df.iterrows():
            c = r["Chromosome"]
            if c not in sorted_chroms:
                continue
            y_val = sorted_chroms.index(c) * 2.0
            # Position in Mb
            pos_mb = r["Position"] / 1e6
            x_coords.append(pos_mb)
            y_coords.append(y_val)

            gene = r.get("Gene_Symbol", "N/A")
            feat = r.get("Genomic_Feature", "Locus")
            cancer_alert = " 🚨 [ONCOGENE ALERT]" if r.get("Is_Cancer_Hallmark", False) else ""
            hover_texts.append(
                f"<b>{c}:{r['Position']:,}</b>{cancer_alert}<br>"
                f"Gene: {gene} ({feat})<br>"
                f"Risk Tier: {tier}<br>"
                f"Hsu Score: {r['Hsu_Score']} | CFD: {r['CFD_Score']}<br>"
                f"Mismatches: {r['Mismatches']}<br>"
                f"Sequence: {r['Sequence']}"
            )

        symbol = "circle" if tier != "HIGH_RISK" else "diamond"
        size = 14 if tier == "HIGH_RISK" else 10

        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers',
            name=tier.replace("_", " ").title(),
            marker=dict(
                size=size,
                color=color,
                symbol=symbol,
                line=dict(color='#FFFFFF', width=1.5),
                opacity=0.9
            ),
            text=hover_texts,
            hoverinfo='text'
        ))

    # Layout configuration
    y_ticks = [i * 2.0 for i in range(len(sorted_chroms))]
    fig.update_layout(
        title="Genomic Karyotype Ideogram (Off-Target Cleavage Distribution)",
        barmode='overlay',
        template="plotly_dark",
        paper_bgcolor="#0A0E17",
        plot_bgcolor="#161B22",
        height=height,
        margin=dict(l=70, r=30, t=50, b=40),
        xaxis=dict(
            title="Genomic Coordinate (Megabases / Mb)",
            gridcolor="#30363D",
            zeroline=False
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=y_ticks,
            ticktext=sorted_chroms,
            gridcolor="#21262D",
            autorange='reversed'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig
