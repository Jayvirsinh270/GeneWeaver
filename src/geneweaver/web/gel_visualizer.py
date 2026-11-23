"""
Virtual Agarose Gel Electrophoresis Plotly Visualizer.
Renders publication-grade UV transilluminator agarose gels with fluorescing ethidium bromide / GelGreen
DNA bands, molecular weight ladders, and lane labels.
"""

from typing import List
import plotly.graph_objects as go
from geneweaver.biology.dual_guide import VirtualGelLane, VirtualGelBand


def build_virtual_gel_plot(lanes: List[VirtualGelLane], height: int = 480) -> go.Figure:
    """
    Renders an interactive 2D agarose gel electrophoresis visualization.
    Y-axis represents relative mobility (Rf from 0.0 at top well to 1.0 at dye front).
    X-axis represents gel sample lanes.
    """
    fig = go.Figure()

    lane_x_coords = {lane.lane_number: lane.lane_number for lane in lanes}
    lane_labels = [lane.sample_label for lane in lanes]

    # Draw dark gel matrix background
    # Agarose well markers at top
    for lane in lanes:
        x = lane.lane_number
        # Draw well rectangle at y = 0.02
        fig.add_shape(
            type="rect",
            x0=x - 0.25, x1=x + 0.25,
            y0=0.01, y1=0.04,
            fillcolor="#1F2937",
            line=dict(color="#4B5563", width=1)
        )

        # Draw DNA bands in this lane
        for band in lane.bands:
            y = band.relative_migration_rf
            opac = max(0.25, min(1.0, band.intensity_pct / 100.0))
            
            # Draw horizontal band shape
            fig.add_shape(
                type="line",
                x0=x - 0.28, x1=x + 0.28,
                y0=y, y1=y,
                line=dict(
                    color=band.band_color,
                    width=4 if band.intensity_pct > 75 else 2.5
                ),
                opacity=opac
            )

            # Scatter hover point on band
            fig.add_trace(go.Scatter(
                x=[x],
                y=[y],
                mode="markers",
                marker=dict(
                    size=10,
                    color=band.band_color,
                    opacity=0.01  # invisible anchor for hover tooltip
                ),
                name=band.name,
                hovertemplate=f"<b>{band.name}</b><br>Size: {band.size_bp:,} bp<br>Relative Mobility (Rf): {y:.2f}<br>Signal Intensity: {band.intensity_pct:.1f}%<extra></extra>",
                showlegend=False
            ))

    # Add size annotations for Ladder in Lane 1
    if len(lanes) > 0 and len(lanes[0].bands) > 0:
        ladder_lane = lanes[0]
        for band in ladder_lane.bands:
            if band.size_bp in [10000, 6000, 3000, 1500, 1000, 500, 250]:
                fig.add_annotation(
                    x=0.55,
                    y=band.relative_migration_rf,
                    text=f"{band.size_bp} bp",
                    showarrow=False,
                    font=dict(size=10, color="#8B949E", family="JetBrains Mono, monospace"),
                    xanchor="right"
                )

    # Configure Dark UV Transilluminator Gel Theme
    num_lanes = len(lanes)
    fig.update_layout(
        title=dict(
            text="🧪 Virtual Agarose Gel Electrophoresis (UV Transilluminator View)",
            font=dict(size=14, color="#00F0FF", family="Inter, sans-serif"),
            x=0.02, y=0.98
        ),
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(1, num_lanes + 1)),
            ticktext=lane_labels,
            tickfont=dict(color="#F0F6FC", size=11, family="Inter, sans-serif"),
            range=[0.2, num_lanes + 0.6],
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            autorange="reversed",  # 0.0 at top, 1.0 at bottom
            range=[0.0, 1.0],
            showticklabels=False,
            showgrid=False,
            zeroline=False
        ),
        paper_bgcolor="#0A0E17",
        plot_bgcolor="#05070B",
        height=height,
        margin=dict(l=60, r=30, t=40, b=40),
        hovermode="closest"
    )

    return fig
