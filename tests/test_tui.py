"""
Unit Tests for Textual TUI & Telemetry Widgets.
"""

import pytest
from geneweaver.tui.telemetry import HardwareMonitor, GpuDeviceStats
from geneweaver.tui.app import AlignmentVisualizer, GeneWeaverApp
from geneweaver.baseline.cpu_aligner import OffTargetHit

def test_hardware_monitor_query():
    monitor = HardwareMonitor(device_ids=[0, 1])
    stats = monitor.query_stats()
    assert len(stats) >= 1
    s0 = stats[0]
    assert s0.utilization_pct >= 0.0
    assert s0.vram_total_gb > 0.0
    assert s0.temperature_c > 0

def test_alignment_visualizer_rendering():
    viz = AlignmentVisualizer()
    hit = OffTargetHit(
        chrom_name="chr7",
        global_pos=142509211,
        strand="(+)",
        mismatches=1,
        sequence="GAGTCCGAGCAGTAGAAGAAGGG",
        hsu_score=0.842,
        risk_tier="HIGH_RISK"
    )
    viz.selected_hit = hit
    panel = viz.render()
    assert panel is not None
    assert "chr7:142,509,211" in str(panel.title)

def test_app_instantiation():
    app = GeneWeaverApp(target_guide="GAGTCCGAGCAGAAGAAGAA")
    assert app.TITLE is not None
    assert app.target_guide == "GAGTCCGAGCAGAAGAAGAA"
