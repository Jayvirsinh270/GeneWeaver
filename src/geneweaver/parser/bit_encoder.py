"""
2-Bit Binary DNA Compressor & Vectorizer.
Compresses DNA sequences by 75% (4 bases per byte, 16 bases per uint32).
Enables hardware-level bitwise XOR and population count (POPC) alignment.
"""

import numpy as np
from typing import Tuple, List

# 2-bit mapping: A=00, C=01, G=10, T=11
BASE_TO_2BIT = {
    'A': 0, 'a': 0,
    'C': 1, 'c': 1,
    'G': 2, 'g': 2,
    'T': 3, 't': 3, 'U': 3, 'u': 3
}

TWO_BIT_TO_BASE = ['A', 'C', 'G', 'T']

# Fast lookup array for ASCII characters (0-255)
ASCII_LOOKUP = np.zeros(256, dtype=np.uint8)
VALID_LOOKUP = np.zeros(256, dtype=np.uint8)

for base, val in BASE_TO_2BIT.items():
    ascii_code = ord(base)
    ASCII_LOOKUP[ascii_code] = val
    VALID_LOOKUP[ascii_code] = 1


def encode_sequence_2bit_bytes(seq_str: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Packs an ASCII DNA string into a 2-bit uint8 array (4 bases per byte).
    
    Returns:
        packed_bytes: np.ndarray of uint8, length = ceil(len(seq) / 4)
        validity_mask: np.ndarray of uint8, where 1 indicates valid canonical base (non-N)
    """
    n = len(seq_str)
    # Convert string to ASCII byte array
    ascii_arr = np.frombuffer(seq_str.encode('ascii'), dtype=np.uint8)
    
    # Map to 2-bit values (0-3) and validity flags
    two_bit_vals = ASCII_LOOKUP[ascii_arr]
    valid_flags = VALID_LOOKUP[ascii_arr]
    
    # Pad to multiple of 4 if necessary
    remainder = n % 4
    if remainder != 0:
        pad_len = 4 - remainder
        two_bit_vals = np.pad(two_bit_vals, (0, pad_len), constant_values=0)
        valid_flags = np.pad(valid_flags, (0, pad_len), constant_values=0)
        
    # Reshape and pack 4 bases per byte: (B0 << 6) | (B1 << 4) | (B2 << 2) | B3
    reshaped = two_bit_vals.reshape(-1, 4)
    packed_bytes = (
        (reshaped[:, 0] << 6) |
        (reshaped[:, 1] << 4) |
        (reshaped[:, 2] << 2) |
        reshaped[:, 3]
    ).astype(np.uint8)
    
    # Reshape validity flags (all 4 must be valid for byte to be valid)
    valid_reshaped = valid_flags.reshape(-1, 4)
    validity_mask = (
        (valid_reshaped[:, 0] << 3) |
        (valid_reshaped[:, 1] << 2) |
        (valid_reshaped[:, 2] << 1) |
        valid_reshaped[:, 3]
    ).astype(np.uint8)
    
    return packed_bytes, validity_mask


def encode_guide_to_words(guide_20nt: str) -> Tuple[int, int]:
    """
    Packs a 20-nt guide sequence into two 32-bit integers:
    - word0: holds first 16 bases (32 bits, 2 bits per base)
    - word1: holds remaining 4 bases (top 8 bits)
    """
    if len(guide_20nt) != 20:
        raise ValueError(f"Guide must be exactly 20nt, got {len(guide_20nt)}")
        
    # First 16 bases -> word0
    word0 = 0
    for i in range(16):
        val = BASE_TO_2BIT.get(guide_20nt[i], 0)
        word0 = (word0 << 2) | val
        
    # Last 4 bases -> word1
    word1 = 0
    for i in range(16, 20):
        val = BASE_TO_2BIT.get(guide_20nt[i], 0)
        word1 = (word1 << 2) | val
        
    return word0, word1


def decode_2bit_bytes(packed_bytes: np.ndarray, length: int) -> str:
    """Decompresses a packed 2-bit byte array back into an ASCII DNA sequence."""
    chars = []
    total_unpacked = 0
    for b in packed_bytes:
        # Extract 4 2-bit values from most significant to least
        b0 = (b >> 6) & 0x03
        b1 = (b >> 4) & 0x03
        b2 = (b >> 2) & 0x03
        b3 = b & 0x03
        
        for val in (b0, b1, b2, b3):
            if total_unpacked < length:
                chars.append(TWO_BIT_TO_BASE[val])
                total_unpacked += 1
            else:
                break
    return "".join(chars)


def get_reverse_complement(seq_str: str) -> str:
    """Computes Watson-Crick reverse complement of a DNA string."""
    comp_map = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N',
                'a': 't', 't': 'a', 'c': 'g', 'g': 'c', 'n': 'n'}
    return "".join(comp_map.get(base, 'N') for base in reversed(seq_str))
