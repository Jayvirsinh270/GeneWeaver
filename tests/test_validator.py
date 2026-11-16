"""
Unit Tests for Defensive Input Validation and Cleansing.
"""

import pytest
import os
from geneweaver.exceptions import InvalidGuideError, InvalidPamError, FastaParseError
from geneweaver.parser.validator import (
    validate_and_clean_guide,
    validate_and_clean_pam,
    validate_fasta_file
)

def test_valid_guide():
    clean = validate_and_clean_guide("gagtccgagcagaagaagaa")
    assert clean == "GAGTCCGAGCAGAAGAAGAA"
    assert len(clean) == 20

def test_guide_with_uracil():
    # RNA Uracil should be normalized to DNA Thymine
    clean = validate_and_clean_guide("GAGUCCGAGCAGAAGAAGAA")
    assert clean == "GAGTCCGAGCAGAAGAAGAA"

def test_invalid_guide_length():
    with pytest.raises(InvalidGuideError) as exc_info:
        validate_and_clean_guide("GAGTCCGAGC")
    assert "Expected exactly 20 nucleotides" in str(exc_info.value)

def test_invalid_guide_characters():
    with pytest.raises(InvalidGuideError) as exc_info:
        validate_and_clean_guide("GAGTCCGAGCAGAAGA1234")
    assert "invalid non-nucleotide characters" in str(exc_info.value)

def test_valid_pam():
    assert validate_and_clean_pam("ngg") == "NGG"
    assert validate_and_clean_pam("tttv") == "TTTV"

def test_invalid_pam():
    with pytest.raises(InvalidPamError):
        validate_and_clean_pam("")
    with pytest.raises(InvalidPamError):
        validate_and_clean_pam("123")

def test_fasta_validation_missing_file():
    with pytest.raises(FastaParseError):
        validate_fasta_file("data/non_existent_file.fa")

def test_fasta_validation_corrupted(tmp_path):
    bad_file = tmp_path / "corrupted.fa"
    bad_file.write_text("THIS IS NOT A VALID FASTA FILE HEADER\nATGCATGC")
    with pytest.raises(FastaParseError) as exc_info:
        validate_fasta_file(str(bad_file))
    assert "does not start with '>'" in str(exc_info.value)
