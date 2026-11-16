"""
Input Validation and Defensive Cleansing Engine.
Ensures genomic inputs, guide RNAs, PAM patterns, and FASTA files conform to biological specifications
before executing compute-intensive GPU kernels.
"""

import os
import re
from typing import Tuple
from geneweaver.exceptions import InvalidGuideError, InvalidPamError, FastaParseError

VALID_DNA_BASES = set("ACGTU")
VALID_PAM_CHARS = set("ACGTUNRYKMSWBDHV")  # IUPAC nucleotide ambiguity codes


def validate_and_clean_guide(guide: str) -> str:
    """
    Cleanses and validates a 20-nt CRISPR guide RNA sequence.
    
    Returns:
        Cleaned, uppercase 20-nt guide string (converts 'U' to 'T').
    Raises:
        InvalidGuideError if length is not 20 or contains non-nucleotide characters.
    """
    if not guide or not isinstance(guide, str):
        raise InvalidGuideError(str(guide), "Guide sequence cannot be empty or null.")

    # Strip whitespaces, linebreaks, and tabs
    clean = re.sub(r"\s+", "", guide).upper()
    # Normalize RNA Uracil (U) to DNA Thymine (T)
    clean = clean.replace("U", "T")

    if len(clean) != 20:
        raise InvalidGuideError(clean, f"Expected exactly 20 nucleotides, but got {len(clean)}.")

    invalid_chars = set(clean) - set("ACGT")
    if invalid_chars:
        bad_str = ", ".join(sorted(invalid_chars))
        raise InvalidGuideError(clean, f"Contains invalid non-nucleotide characters: {bad_str}.")

    return clean


def validate_and_clean_pam(pam: str) -> str:
    """
    Validates and standardizes a Protospacer Adjacent Motif (PAM).
    """
    if not pam or not isinstance(pam, str):
        raise InvalidPamError(str(pam), "PAM motif cannot be empty.")

    clean = re.sub(r"\s+", "", pam).upper().replace("U", "T")
    if len(clean) < 2 or len(clean) > 6:
        raise InvalidPamError(clean, f"PAM motif length must be between 2 and 6 nt (got {len(clean)}).")

    invalid = set(clean) - VALID_PAM_CHARS
    if invalid:
        bad_str = ", ".join(sorted(invalid))
        raise InvalidPamError(clean, f"Contains unrecognized IUPAC base codes: {bad_str}.")

    return clean


def validate_fasta_file(file_path: str) -> Tuple[bool, str]:
    """
    Inspects a FASTA file on disk to ensure it is readable and properly formatted.
    
    Returns:
        (is_valid, chromosome_header)
    Raises:
        FastaParseError if file is missing, empty, or lacks standard '>' header.
    """
    if not os.path.exists(file_path):
        raise FastaParseError(file_path, "File does not exist on disk.")

    file_size = os.path.getsize(file_path)
    if file_size == 0:
        raise FastaParseError(file_path, "File is completely empty (0 bytes).")

    # Read the first few lines to inspect FASTA header
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            first_line = f.readline().strip()
            while first_line == "" and f.tell() < file_size:
                first_line = f.readline().strip()

            if not first_line.startswith(">"):
                raise FastaParseError(
                    file_path,
                    f"First non-empty line does not start with '>' header (found: '{first_line[:40]}...')."
                )
            
            header_name = first_line[1:].split()[0]
            # Ensure at least one line of sequence exists
            second_line = f.readline().strip()
            if not second_line:
                raise FastaParseError(file_path, "Header is present, but no sequence data was found.")

            return True, header_name
    except UnicodeDecodeError as e:
        raise FastaParseError(file_path, f"Binary or non-UTF-8 encoding detected: {str(e)}")
    except Exception as e:
        if isinstance(e, FastaParseError):
            raise e
        raise FastaParseError(file_path, f"Unexpected I/O read failure: {str(e)}")
