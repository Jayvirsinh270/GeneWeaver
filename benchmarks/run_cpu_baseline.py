"""
Phase 1: CPU Baseline Alignment Benchmark Runner.
Executes pure Python and BioPython sequence alignment to establish the baseline
performance metric against which Week 2 GPU acceleration will be evaluated.
"""

import sys
import os

# Ensure src/ is on Python search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import argparse
from geneweaver.parser.mock_genome import generate_mock_chromosome, write_fasta
from geneweaver.parser.fasta_streamer import FastaStreamer
from geneweaver.baseline.cpu_aligner import CpuAligner
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def run_benchmark(length_bp: int = 1_000_000, chunk_size: int = 250_000, guide: str = "GAGTCCGAGCAGAAGAAGAA"):
    console.print(Panel.fit(
        "[bold cyan]GeneWeaver Phase 1: CPU Baseline Benchmark[/bold cyan]\n"
        "[dim]Pure Python & BioPython Sliding-Window Sequence Alignment[/dim]",
        border_style="cyan"
    ))

    # Step 1: Generate Mock Synthetic Chromosome with Injected Spikes
    console.print(f"[bold yellow]1. Generating synthetic benchmark genome ({length_bp:,} bp)...[/bold yellow]")
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    fasta_path = os.path.join(data_dir, f"benchmark_{length_bp}_bp.fa")

    # Injected ground-truth spikes with predetermined mismatches
    known_spikes = [
        (150_000, guide + "AGG"),                             # 0 mismatches (On-target)
        (350_000, guide[:19] + "T" + "AGG"),                   # 1 mismatch (Seed pos 20)
        (550_000, "A" + guide[1:] + "AGG"),                   # 1 mismatch (Distal pos 1)
        (750_000, "A" + guide[1:9] + "T" + guide[10:] + "AGG") # 2 mismatches (Pos 1 & 10)
    ]
    # Filter spikes that fit within requested length
    valid_spikes = [(pos, seq) for pos, seq in known_spikes if pos + len(seq) <= length_bp]

    raw_seq = generate_mock_chromosome(length_bp, spikes=valid_spikes, seed=42)
    write_fasta(raw_seq, fasta_path, header="chr_benchmark_phase1")
    console.print(f"   [green]Saved FASTA to: {fasta_path}[/green] with {len(valid_spikes)} ground-truth spikes.")

    # Step 2: Initialize FastaStreamer and CpuAligner
    console.print(f"[bold yellow]2. Initializing BioPython FastaStreamer (Chunk Size: {chunk_size:,} bp, Overlap: 22 bp)...[/bold yellow]")
    streamer = FastaStreamer(chunk_size=chunk_size, overlap_bp=22)
    aligner = CpuAligner(target_guide=guide, pam_motif="NGG", max_mismatches=4)

    # Step 3: Execute Alignment Scan
    console.print(f"[bold yellow]3. Executing CPU linear scan over {length_bp:,} bp...[/bold yellow]")
    hits, metrics = aligner.benchmark_chunk_stream(streamer, fasta_path)

    # Step 4: Display Benchmark Results Table
    console.print("\n[bold green]✓ CPU Baseline Scan Complete![/bold green]\n")

    metrics_table = Table(title="[bold white]Phase 1: CPU Baseline Performance Metrics[/bold white]", border_style="blue")
    metrics_table.add_column("Metric", style="cyan", justify="left")
    metrics_table.add_column("Measured Value", style="bold yellow", justify="right")
    metrics_table.add_column("Benchmark Target Notes", style="dim white", justify="left")

    metrics_table.add_row("Total Base Pairs", f"{metrics.total_base_pairs:,} bp", "Mock chromosome size")
    metrics_table.add_row("Elapsed Time", f"{metrics.elapsed_seconds:.3f} seconds", "Pure Python CPU execution")
    metrics_table.add_row("Throughput (bp/s)", f"{metrics.throughput_bp_per_sec:,.0f} bp/s", "Base pairs processed / sec")
    metrics_table.add_row("Throughput (Mbp/s)", f"{metrics.throughput_mbp_per_sec:.3f} Mbp/s", "Industry comparison standard")
    metrics_table.add_row("Candidates Found", f"{metrics.total_candidates_found:,}", f"Hits with <= 4 mismatches")
    metrics_table.add_row("High-Risk Hits (S >= 0.5)", f"{metrics.high_risk_count:,}", "Severe off-target mutations")

    console.print(metrics_table)

    # Display Top Hits Table
    if hits:
        hits_table = Table(title="[bold white]Sample Detected Off-Target Candidates[/bold white]", border_style="magenta")
        hits_table.add_column("Locus Coordinate", style="cyan")
        hits_table.add_column("Strand", style="dim")
        hits_table.add_column("Mismatches", style="yellow", justify="center")
        hits_table.add_column("Hsu-Zhang Score", style="bold green", justify="right")
        hits_table.add_column("Risk Tier", style="bold")
        hits_table.add_column("Aligned 23-bp Sequence", style="white")

        for h in hits[:10]:
            risk_style = "bold red" if h.risk_tier == "HIGH_RISK" else ("yellow" if h.risk_tier == "MODERATE_RISK" else "green")
            hits_table.add_row(
                f"{h.chrom_name}:{h.global_pos:,}",
                h.strand,
                str(h.mismatches),
                f"{h.hsu_score:.4f}",
                f"[{risk_style}]{h.risk_tier}[/{risk_style}]",
                h.sequence
            )
        console.print(hits_table)

    return metrics

def main():
    parser = argparse.ArgumentParser(description="GeneWeaver Phase 1: CPU Baseline Benchmark")
    parser.add_argument("--length", type=int, default=1_000_000, help="Genome sequence length in bp (default: 1,000,000)")
    parser.add_argument("--chunk-size", type=int, default=250_000, help="Streaming chunk size (default: 250,000)")
    parser.add_argument("--guide", type=str, default="GAGTCCGAGCAGAAGAAGAA", help="Target guide 20nt")
    args = parser.parse_args()

    run_benchmark(length_bp=args.length, chunk_size=args.chunk_size, guide=args.guide)

if __name__ == "__main__":
    main()
