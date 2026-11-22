"""
Base Editing (CBE / ABE) & Prime Editing (pegRNA) Biophysical Simulation Engine.
Implements deamination activity windows, bystander mutation quantification,
and automated pegRNA architecture (PBS + RT template) for DSB-free precision genome editing.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from geneweaver.biology.presets import calculate_gc_content, estimate_melting_temp


@dataclass
class BaseEditorProfile:
    """Biophysical definition of a base editing fusion enzyme."""
    id: str
    name: str
    editor_type: str            # 'CBE' (Cytosine) or 'ABE' (Adenine)
    target_base: str            # 'C' or 'A'
    product_base: str           # 'T' or 'G'
    deamination_window: Tuple[int, int]  # (start, end) 1-indexed protospacer positions
    peak_positions: List[int]   # Positions with highest catalytic rate
    nickase_architecture: str   # e.g., 'Cas9n (D10A)'
    description: str
    clinical_use_case: str


@dataclass
class BystanderEdit:
    """Information on an unintended bystander deamination inside the editing window."""
    position: int               # 1-indexed position in protospacer
    original_base: str
    mutated_base: str
    distance_from_target: int
    predicted_edit_rate_pct: float
    risk_level: str             # 'HIGH_BYSTANDER', 'MODERATE', 'LOW'


@dataclass
class BaseEditingOutcome:
    """Complete simulation result for base editing of a protospacer."""
    editor_id: str
    editor_name: str
    editor_type: str
    original_protospacer: str
    target_position: int
    target_base: str
    converted_base: str
    edited_protospacer: str
    is_target_in_window: bool
    window_range: Tuple[int, int]
    window_sequence: str
    bystanders_found: List[BystanderEdit]
    bystander_count: int
    bystander_risk_tier: str    # 'CLEAN_EDIT_NO_BYSTANDERS', 'MODERATE_BYSTANDER_RISK', 'HIGH_BYSTANDER_RISK'
    predicted_on_target_efficiency_pct: float
    product_purity_score_pct: float
    all_possible_alleles: List[str]
    biophysical_verdict: str
    remediation_guidance: str


@dataclass
class PegRNADesign:
    """Biophysical specification for a Prime Editing guide RNA (pegRNA)."""
    spacer_seq_20nt: str
    scaffold_seq: str
    rtt_seq: str                # Reverse Transcription Template
    pbs_seq: str                # Primer Binding Site
    composite_pegrna_seq: str   # Spacer + Scaffold + RTT + PBS
    intended_mutation: str      # e.g., 'C->T at position +5'
    pbs_length_nt: int
    pbs_tm_celsius: float
    pbs_gc_pct: float
    rtt_length_nt: int
    rtt_gc_pct: float
    nick_to_edit_distance_bp: int
    feasibility_score: float    # 0.0 to 100.0
    feasibility_tier: str       # 'OPTIMAL', 'MODERATE', 'CHALLENGING'
    recommendations: List[str]


# Clinical & Bioinformatic Base Editor Registry
BASE_EDITOR_REGISTRY: Dict[str, BaseEditorProfile] = {
    "be4max": BaseEditorProfile(
        id="be4max",
        name="BE4max (Cytosine Base Editor)",
        editor_type="CBE",
        target_base="C",
        product_base="T",
        deamination_window=(4, 8),
        peak_positions=[5, 6],
        nickase_architecture="SpCas9n (D10A) + rAPOBEC1 + 2x UGI",
        description="High-efficiency cytosine base editor optimized with mammalian codons and dual uracil glycosylase inhibitors.",
        clinical_use_case="Targeted transition of C·G to T·A to introduce premature stop codons (iSTOP) or correct pathogenic mutations without DSBs."
    ),
    "ancbe4max": BaseEditorProfile(
        id="ancbe4max",
        name="AncBE4max (Ancestral CBE)",
        editor_type="CBE",
        target_base="C",
        product_base="T",
        deamination_window=(4, 7),
        peak_positions=[5],
        nickase_architecture="SpCas9n (D10A) + Anc687 Deaminase",
        description="Narrower deamination window designed to minimize bystander cytosine mutations.",
        clinical_use_case="Precision correction of single cytosine disease variants located near bystander cytosines."
    ),
    "abe8e": BaseEditorProfile(
        id="abe8e",
        name="ABE8e (Adenine Base Editor - Ultra Fast)",
        editor_type="ABE",
        target_base="A",
        product_base="G",
        deamination_window=(3, 9),
        peak_positions=[5, 6, 7],
        nickase_architecture="SpCas9n (D10A) + TadA-8e Deaminase",
        description="Ultra-high-activity adenine base editor with accelerated deamination kinetics (>1000-fold faster than ABE7.10).",
        clinical_use_case="High-potency in-vivo correction of pathogenic G·C to A·T mutations (e.g. Verve PCSK9 clinical trial)."
    ),
    "abe7_10": BaseEditorProfile(
        id="abe7_10",
        name="ABE7.10 (Standard ABE)",
        editor_type="ABE",
        target_base="A",
        product_base="G",
        deamination_window=(4, 7),
        peak_positions=[5, 6],
        nickase_architecture="SpCas9n (D10A) + TadA*7.10",
        description="Canonical high-fidelity adenine base editor with tight editing window and negligible off-target RNA deamination.",
        clinical_use_case="Ex-vivo hematopoietic cell therapy and selective point mutation repair."
    )
}

STANDARD_SGRNA_SCAFFOLD = (
    "GTTTTAGAGCTAGAAATAGCAAGTTAAAATAAGGCTAGTCCGTTATCAACTTGAAAAAGTGGCACCGAGTCGGTGC"
)


def get_supported_base_editors() -> Dict[str, BaseEditorProfile]:
    """Returns the dictionary of supported base editing enzyme systems."""
    return BASE_EDITOR_REGISTRY


def simulate_base_editing(
    protospacer_seq: str,
    target_pos: int,
    editor_id: str = "be4max"
) -> BaseEditingOutcome:
    """
    Simulates base editing, deamination window activity, and bystander mutations.
    
    Args:
        protospacer_seq: 20-nt guide spacer sequence.
        target_pos: 1-indexed position of intended nucleotide to mutate (1 to 20).
        editor_id: Base editor system identifier (e.g., 'be4max', 'abe8e').
    
    Returns:
        BaseEditingOutcome with comprehensive bystander quantification.
    """
    proto = protospacer_seq.upper().strip()
    editor = BASE_EDITOR_REGISTRY.get(editor_id, BASE_EDITOR_REGISTRY["be4max"])
    w_start, w_end = editor.deamination_window

    if target_pos < 1 or target_pos > len(proto):
        raise ValueError(f"Target position {target_pos} is outside protospacer bounds (1–{len(proto)}).")

    intended_base = proto[target_pos - 1]
    is_in_window = (w_start <= target_pos <= w_end)

    # Window subsequence
    w_seq = proto[w_start - 1:w_end]

    # Predict intended base edit rate
    if is_in_window:
        if target_pos in editor.peak_positions:
            on_target_rate = 88.0 if editor.id == "abe8e" else 78.0
        else:
            on_target_rate = 62.0
    else:
        on_target_rate = 8.0  # severely reduced outside activity window

    # Find bystander bases in window
    bystanders: List[BystanderEdit] = []
    for pos_idx in range(w_start, w_end + 1):
        if pos_idx == target_pos:
            continue
        base_at_pos = proto[pos_idx - 1]
        if base_at_pos == editor.target_base:
            dist = abs(pos_idx - target_pos)
            if pos_idx in editor.peak_positions:
                bystander_rate = 75.0 if editor.id == "abe8e" else 65.0
                r_level = "HIGH_BYSTANDER"
            else:
                bystander_rate = 38.0
                r_level = "MODERATE"
            
            bystanders.append(BystanderEdit(
                position=pos_idx,
                original_base=base_at_pos,
                mutated_base=editor.product_base,
                distance_from_target=dist,
                predicted_edit_rate_pct=bystander_rate,
                risk_level=r_level
            ))

    b_count = len(bystanders)
    if b_count == 0:
        risk_tier = "CLEAN_EDIT_NO_BYSTANDERS"
        purity = 96.0
        verdict = f"Pure single-base conversion: Only the target {editor.target_base} at position {target_pos} is present in the deamination window."
        guidance = "Excellent candidate for clinical translation. Zero bystander amino acid or splicing alterations expected."
    elif b_count == 1:
        risk_tier = "MODERATE_BYSTANDER_RISK"
        purity = round(max(35.0, 100.0 - (bystanders[0].predicted_edit_rate_pct * 0.7)), 1)
        verdict = f"1 bystander {editor.target_base} detected at position {bystanders[0].position} ({bystanders[0].predicted_edit_rate_pct}% estimated co-deamination rate)."
        guidance = "Consider using AncBE4max or narrow-window engineering to suppress the bystander base."
    else:
        risk_tier = "HIGH_BYSTANDER_RISK"
        purity = round(max(15.0, 100.0 - sum(b.predicted_edit_rate_pct for b in bystanders) * 0.45), 1)
        verdict = f"High bystander heterogeneity: {b_count} bystander {editor.target_base} bases detected inside window {w_start}–{w_end}."
        guidance = "High risk of mixed codon missense alleles. Recommend shifting guide position or switching to Prime Editing (pegRNA)."

    # Construct edited sequence (target base only)
    edited_list = list(proto)
    if intended_base == editor.target_base:
        edited_list[target_pos - 1] = editor.product_base
    clean_edited_seq = "".join(edited_list)

    # Generate all possible combinatorial bystander alleles
    all_alleles = [clean_edited_seq]
    if b_count > 0:
        # Full co-deamination allele
        full_mut = list(edited_list)
        for b in bystanders:
            full_mut[b.position - 1] = editor.product_base
        all_alleles.append("".join(full_mut))

    return BaseEditingOutcome(
        editor_id=editor.id,
        editor_name=editor.name,
        editor_type=editor.editor_type,
        original_protospacer=proto,
        target_position=target_pos,
        target_base=intended_base,
        converted_base=editor.product_base if intended_base == editor.target_base else intended_base,
        edited_protospacer=clean_edited_seq,
        is_target_in_window=is_in_window,
        window_range=editor.deamination_window,
        window_sequence=w_seq,
        bystanders_found=bystanders,
        bystander_count=b_count,
        bystander_risk_tier=risk_tier,
        predicted_on_target_efficiency_pct=on_target_rate,
        product_purity_score_pct=purity,
        all_possible_alleles=all_alleles,
        biophysical_verdict=verdict,
        remediation_guidance=guidance
    )


def design_pegrna(
    target_guide: str,
    edit_type: str,
    edit_offset_from_nick: int,
    desired_sequence: str,
    flanking_downstream_dna: str = "GACGTACTGACGTACGTACT",
    pbs_length: int = 13
) -> PegRNADesign:
    """
    Designs a Prime Editing guide RNA (pegRNA) comprising:
    Spacer + sgRNA Scaffold + RT Template + PBS.
    
    Args:
        target_guide: 20-nt spacer sequence.
        edit_type: e.g., 'INSERTION', 'DELETION', 'SUBSTITUTION'.
        edit_offset_from_nick: Distance (bp) from Cas9 nick site (between nt 17 and 18).
        desired_sequence: New sequence to synthesize into genomic target.
        flanking_downstream_dna: Downstream genomic reference sequence for homology arm.
        pbs_length: Length of Primer Binding Site (typically 11–15 nt).
    """
    spacer = target_guide.upper().strip()
    
    # 1. Design Primer Binding Site (PBS)
    # The PBS hybridizes to the non-target strand upstream of the Cas9 nick (between 17 and 18)
    nick_idx = 17  # 0-indexed position 17
    start_pbs = max(0, nick_idx - pbs_length)
    pbs_candidate = spacer[start_pbs:nick_idx]
    
    # Reverse complement for hybridization to liberated single-stranded flap
    tr = str.maketrans("ACGT", "TGCA")
    pbs_seq = pbs_candidate.translate(tr)[::-1]
    pbs_tm = estimate_melting_temp(pbs_seq)
    pbs_gc = calculate_gc_content(pbs_seq)

    # 2. Design Reverse Transcription Template (RTT)
    # RTT carries the edit plus downstream homology arm (10–16 bp)
    homology_arm = flanking_downstream_dna[:12].upper()
    rtt_seq = f"{desired_sequence.upper()}{homology_arm}"
    rtt_gc = calculate_gc_content(rtt_seq)

    # 3. Construct Composite pegRNA
    composite = f"{spacer}{STANDARD_SGRNA_SCAFFOLD}{rtt_seq}{pbs_seq}"

    # 4. Biophysical Feasibility Scoring
    # Ideal PBS Tm is 30°C to 38°C; nick-to-edit distance optimal <= 10 bp
    recs: List[str] = []
    score = 90.0

    if 28.0 <= pbs_tm <= 38.0:
        recs.append(f"PBS melting temperature ({pbs_tm}°C) is in optimal flap hybridization window (30–38°C).")
    else:
        score -= 15.0
        recs.append(f"PBS melting temperature ({pbs_tm}°C) is outside ideal range; adjust PBS length to 12–14 nt.")

    if abs(edit_offset_from_nick) <= 10:
        recs.append(f"Nick-to-edit distance (+{edit_offset_from_nick} bp) yields high prime editing efficiency.")
    else:
        score -= 15.0
        recs.append(f"Large distance from nick (+{edit_offset_from_nick} bp) may attenuate RT extension rate.")

    if 40.0 <= rtt_gc <= 65.0:
        recs.append(f"RTT GC content ({rtt_gc}%) prevents secondary structure interference.")
    else:
        score -= 10.0
        recs.append(f"RTT GC content ({rtt_gc}%) may cause flap secondary structure.")

    score = max(20.0, min(100.0, score))
    tier = "OPTIMAL" if score >= 80 else ("MODERATE" if score >= 60 else "CHALLENGING")

    return PegRNADesign(
        spacer_seq_20nt=spacer,
        scaffold_seq=STANDARD_SGRNA_SCAFFOLD,
        rtt_seq=rtt_seq,
        pbs_seq=pbs_seq,
        composite_pegrna_seq=composite,
        intended_mutation=f"{edit_type}: '{desired_sequence}' at offset +{edit_offset_from_nick} bp",
        pbs_length_nt=len(pbs_seq),
        pbs_tm_celsius=pbs_tm,
        pbs_gc_pct=pbs_gc,
        rtt_length_nt=len(rtt_seq),
        rtt_gc_pct=rtt_gc,
        nick_to_edit_distance_bp=edit_offset_from_nick,
        feasibility_score=round(score, 1),
        feasibility_tier=tier,
        recommendations=recs
    )
