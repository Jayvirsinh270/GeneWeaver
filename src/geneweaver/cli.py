"""
GeneWeaver Unified Command-Line Interface (CLI).
Provides entry points for CPU benchmarks, GPU alignment scans, TUI dashboard, and Web GUI.
"""

import sys
import os

# Ensure project root and src/ are in sys.path
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("src"))

import argparse
import subprocess

def cmd_benchmark_cpu(args):
    """Executes Phase 1 CPU baseline benchmark."""
    from benchmarks.run_cpu_baseline import run_benchmark
    run_benchmark(length_bp=args.length, chunk_size=args.chunk_size, guide=args.guide)

def cmd_scan(args):
    """Executes CRISPR off-target alignment scan."""
    if args.web:
        print("[GeneWeaver] Launching Modern Web GUI Dashboard...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "src/geneweaver/web/app.py"])
    elif args.tui:
        print("[GeneWeaver] Launching Textual Terminal TUI Dashboard...")
        from geneweaver.tui.app import GeneWeaverApp
        app = GeneWeaverApp()
        app.run()
    else:
        # Default headless CLI scan
        print(f"[GeneWeaver] Scanning FASTA: {args.fasta} with Guide: {args.guide} (Max Mismatches: {args.mismatches})")
        from geneweaver.baseline.cpu_aligner import CpuAligner
        from geneweaver.parser.fasta_streamer import FastaStreamer
        streamer = FastaStreamer(chunk_size=args.chunk_size, overlap_bp=22)
        aligner = CpuAligner(target_guide=args.guide, pam_motif=args.pam, max_mismatches=args.mismatches)
        hits, metrics = aligner.benchmark_chunk_stream(streamer, args.fasta)
        print(f"[GeneWeaver] Scan complete! Found {metrics.total_candidates_found} candidates in {metrics.elapsed_seconds:.3f}s ({metrics.throughput_mbp_per_sec:.2f} Mbp/s)")

def main():
    parser = argparse.ArgumentParser(
        prog="geneweaver",
        description="GeneWeaver: GPU-Accelerated CRISPR Off-Target Alignment Engine"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Subcommand: benchmark-cpu
    p_bench = subparsers.add_parser("benchmark-cpu", help="Run Phase 1 CPU Baseline Benchmark")
    p_bench.add_argument("--length", type=int, default=1_000_000, help="Benchmark sequence length in bp")
    p_bench.add_argument("--chunk-size", type=int, default=250_000, help="Streaming chunk size")
    p_bench.add_argument("--guide", type=str, default="GAGTCCGAGCAGAAGAAGAA", help="Target guide 20nt")

    # Subcommand: scan
    p_scan = subparsers.add_parser("scan", help="Run alignment scan")
    p_scan.add_argument("--fasta", type=str, default="data/benchmark_1000000_bp.fa", help="Path to FASTA file")
    p_scan.add_argument("--guide", type=str, default="GAGTCCGAGCAGAAGAAGAA", help="Target guide 20nt")
    p_scan.add_argument("--pam", type=str, default="NGG", help="PAM motif (default: NGG)")
    p_scan.add_argument("--mismatches", type=int, default=4, help="Max mismatches (1-6)")
    p_scan.add_argument("--chunk-size", type=int, default=5_000_000, help="Genomic chunk size")
    p_scan.add_argument("--tui", action="store_true", help="Launch interactive Terminal TUI")
    p_scan.add_argument("--web", action="store_true", help="Launch interactive Web GUI")

    args = parser.parse_args()
    if args.command == "benchmark-cpu":
        cmd_benchmark_cpu(args)
    elif args.command == "scan":
        cmd_scan(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
