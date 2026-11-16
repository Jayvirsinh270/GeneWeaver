"""
Custom Exception Hierarchy for GeneWeaver.
Provides structured, informative, and actionable error messages for clinical researchers.
"""

class GeneWeaverError(Exception):
    """Base exception for all GeneWeaver errors."""
    def __init__(self, message: str, remediation: str = ""):
        self.message = message
        self.remediation = remediation
        full_msg = f"{message}\n[Suggested Action]: {remediation}" if remediation else message
        super().__init__(full_msg)


class InvalidGuideError(GeneWeaverError):
    """Raised when the target guide RNA sequence violates biological specifications."""
    def __init__(self, guide: str, reason: str):
        super().__init__(
            message=f"Invalid sgRNA Guide Sequence: '{guide}'. Reason: {reason}",
            remediation="A canonical CRISPR guide must be exactly 20 nucleotides containing only A, C, G, T, or U."
        )


class InvalidPamError(GeneWeaverError):
    """Raised when the specified PAM motif is malformed."""
    def __init__(self, pam: str, reason: str):
        super().__init__(
            message=f"Invalid Protospacer Adjacent Motif (PAM): '{pam}'. Reason: {reason}",
            remediation="Provide a valid 3-nt or 4-nt PAM pattern such as 'NGG', 'NAG', 'NGA', or 'TTTV'."
        )


class FastaParseError(GeneWeaverError):
    """Raised when a FASTA file is corrupted, empty, or unreadable."""
    def __init__(self, file_path: str, reason: str):
        super().__init__(
            message=f"FASTA Parsing Failure for file '{file_path}'. Reason: {reason}",
            remediation="Ensure the file begins with a header line starting with '>' and contains valid genomic base text."
        )


class HardwareMemoryError(GeneWeaverError):
    """Raised when requested chunk allocation exceeds safe GPU VRAM or host RAM limits."""
    def __init__(self, requested_bytes: int, available_bytes: int):
        req_mb = requested_bytes / (1024 * 1024)
        avail_mb = available_bytes / (1024 * 1024)
        super().__init__(
            message=f"Memory Safety Limit Exceeded: Requested {req_mb:.1f} MB, but only {avail_mb:.1f} MB free VRAM is available.",
            remediation="Reduce 'chunk_size_mb' in configuration or enable CPU fallback mode to avoid out-of-memory driver crashes."
        )
