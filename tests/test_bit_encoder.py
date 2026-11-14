"""
Unit Tests for 2-Bit Binary DNA Compression & Encoding.
"""

import pytest
import numpy as np
from geneweaver.parser.bit_encoder import (
    encode_sequence_2bit_bytes,
    decode_2bit_bytes,
    encode_guide_to_words,
    get_reverse_complement
)

def test_roundtrip_2bit_encoding():
    seq = "ACGTACGTACGTACGT"
    packed, valid = encode_sequence_2bit_bytes(seq)
    
    # 16 bases = 4 bytes
    assert len(packed) == 4
    assert np.all(valid == 0x0F)  # All 4 bases in each byte are valid

    # Decode back
    decoded = decode_2bit_bytes(packed, length=len(seq))
    assert decoded == seq

def test_encode_guide_words():
    guide = "GAGTCCGAGCAGAAGAAGAA"
    w0, w1 = encode_guide_to_words(guide)
    
    assert isinstance(w0, int)
    assert isinstance(w1, int)
    assert w0 > 0
    assert w1 > 0

def test_reverse_complement():
    assert get_reverse_complement("ATGC") == "GCAT"
    assert get_reverse_complement("GAGTCCGAGCAGAAGAAGAA") == "TTCTTCTTCTGCTCGGACTC"

def test_ambiguous_base_masking():
    seq = "ACGN"
    packed, valid = encode_sequence_2bit_bytes(seq)
    assert len(packed) == 1
    # Base 3 is 'N' (invalid), so bit 0 of valid should be 0 (0b1110 == 14)
    assert valid[0] == 0b1110
