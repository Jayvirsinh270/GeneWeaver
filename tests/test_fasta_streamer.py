"""
Unit Tests for FastaStreamer and Overlapped Chunking.
"""

import os
import pytest
from geneweaver.parser.fasta_streamer import FastaStreamer, GenomicChunk
from geneweaver.parser.mock_genome import generate_mock_chromosome, write_fasta

@pytest.fixture
def sample_fasta(tmp_path):
    """Creates a temporary FASTA file with 10,000 base pairs."""
    fasta_file = tmp_path / "test_chr.fa"
    seq = generate_mock_chromosome(10_000, seed=99)
    write_fasta(seq, str(fasta_file), header="chrTest")
    return str(fasta_file)

def test_fasta_streamer_chunking(sample_fasta):
    chunk_size = 3000
    overlap_bp = 22
    streamer = FastaStreamer(chunk_size=chunk_size, overlap_bp=overlap_bp)
    
    chunks = list(streamer.stream_chunks(sample_fasta))
    assert len(chunks) > 1

    # Verify first chunk
    assert chunks[0].chunk_index == 0
    assert chunks[0].global_start == 0
    assert chunks[0].global_end == chunk_size + overlap_bp
    assert chunks[0].overlap_len == overlap_bp
    assert not chunks[0].is_last

    # Verify last chunk
    last = chunks[-1]
    assert last.is_last
    assert last.global_end == 10_000

def test_seam_preservation(tmp_path):
    """
    Critical bioinformatics test:
    Verify that an off-target motif placed exactly across the chunk boundary seam
    (e.g. from index 2990 to 3013 when chunk size is 3000) is preserved in the first chunk's overlap!
    """
    chunk_size = 3000
    overlap_bp = 22
    streamer = FastaStreamer(chunk_size=chunk_size, overlap_bp=overlap_bp)

    # Spike spanning from 2990 to 3013 (23bp)
    spike_motif = "GAGTCCGAGCAGAAGAAGAAAGG"
    spikes = [(2990, spike_motif)]
    seq = generate_mock_chromosome(6000, spikes=spikes, seed=42)

    fasta_file = tmp_path / "seam_test.fa"
    write_fasta(seq, str(fasta_file), header="chrSeam")

    chunks = list(streamer.stream_chunks(str(fasta_file)))
    # Chunk 0 covers 0 to 3022
    # The spike at 2990..3013 must be entirely contained in Chunk 0!
    assert spike_motif in chunks[0].sequence
