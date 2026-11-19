"""
In-Vivo CRISPR Delivery Vehicle & Cell-Type Epigenetic Simulator.
Simulates delivery kinetics, cell-type specific chromatin accessibility,
and Cas9 intracellular exposure duration to calculate FDA IND Preclinical Feasibility.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class DeliverySimulationResult:
    """Preclinical in-vivo simulation outcome for a CRISPR therapeutic candidate."""
    cell_type_name: str
    delivery_modality: str
    transduction_efficiency_pct: float     # e.g. 88.5%
    chromatin_accessibility_pct: float     # Open vs closed DNA
    cas9_exposure_duration: str            # e.g. '12 - 24 hours (Transient)' vs 'Constitutive (Months)'
    effective_in_vivo_cutting_pct: float   # Net predicted editing rate inside target tissue
    fda_ind_feasibility_score: float       # 0 - 100 composite score
    feasibility_tier: str                  # 'HIGH_FEASIBILITY', 'MODERATE_FEASIBILITY', 'CHALLENGING'
    preclinical_advantages: List[str]
    preclinical_warnings: List[str]
    regulatory_recommendation: str


# Tissue and Cell Type Profiles
CELL_TYPE_PROFILES: Dict[str, Dict[str, float]] = {
    "Hepatocytes (Liver Parenchyma)": {
        "chromatin_accessibility": 85.0,
        "membrane_permeability": 90.0,
        "primary_organ": "Liver",
        "description": "Ideal target for intravenous LNP infusion. Highly fenestrated sinusoidal endothelium allows rapid hepatic uptake."
    },
    "Hematopoietic Stem Cells (CD34+ HSCs)": {
        "chromatin_accessibility": 92.0,
        "membrane_permeability": 95.0,
        "primary_organ": "Bone Marrow (Ex-Vivo)",
        "description": "Standard for Sickle Cell and Thalassemia. Ex-vivo mobilization followed by electroporation achieves >90% editing rates."
    },
    "Primary T-Lymphocytes (CAR-T Oncology)": {
        "chromatin_accessibility": 88.0,
        "membrane_permeability": 92.0,
        "primary_organ": "Peripheral Blood (Ex-Vivo)",
        "description": "Target for immune checkpoint disruption (PD-1, CTLA-4). Ex-vivo RNP editing avoids systemic vector toxicity."
    },
    "Cortical & Striatal Neurons (CNS)": {
        "chromatin_accessibility": 72.0,
        "membrane_permeability": 65.0,
        "primary_organ": "Central Nervous System (Brain)",
        "description": "Protected by Blood-Brain Barrier (BBB). Requires direct stereotaxic delivery or engineered BBB-crossing AAV capsids."
    },
    "Cardiomyocytes (Heart Muscle)": {
        "chromatin_accessibility": 78.0,
        "membrane_permeability": 75.0,
        "primary_organ": "Heart (Myocardium)",
        "description": "Post-mitotic cells requiring AAV9 or heart-tropic LNPs. Intracellular delivery must avoid provoking cardiac arrhythmias."
    },
    "Airway Epithelial Cells (Lung Bronchi)": {
        "chromatin_accessibility": 80.0,
        "membrane_permeability": 70.0,
        "primary_organ": "Respiratory Tract",
        "description": "Relevant for Cystic Fibrosis (CFTR). Thick mucosal barrier requires specialized nebulized lipid nanoparticles or viral capsids."
    }
}

# Delivery Modality Kinetics
DELIVERY_MODALITIES: Dict[str, Dict[str, any]] = {
    "Lipid Nanoparticles (LNP - Cas9 mRNA + sgRNA)": {
        "base_efficiency": 88.0,
        "persistence_hours": "18 - 24 Hours",
        "persistence_tier": "Transient Pulse",
        "off_target_risk_factor": 0.20,  # Low cumulative off-target accumulation
        "advantages": [
            "Transient Cas9 expression (18-24h) minimizes cumulative off-target mutations.",
            "Non-viral: Zero risk of genomic viral vector integration.",
            "Fully synthetic manufacturing, scalable to clinical GMP quantities."
        ],
        "warnings": [
            "Strong hepatic tropism; requires specialized targeting ligands for extrahepatic tissues.",
            "Potential transient immune activation via Toll-like receptors (TLR3/7)."
        ]
    },
    "Ex-Vivo Electroporation (Cas9 RNP Protein Complex)": {
        "base_efficiency": 94.0,
        "persistence_hours": "6 - 12 Hours",
        "persistence_tier": "Ultra-Short Transient",
        "off_target_risk_factor": 0.10,  # Extremely safe off-target window
        "advantages": [
            "Gold standard for hematologic cell therapy (used in FDA-approved Casgevy).",
            "Cas9 protein degrades within 12 hours, almost completely eliminating delayed off-target cleavage.",
            "Unmatched editing efficiency exceeding 90% in CD34+ and T-cells."
        ],
        "warnings": [
            "Restricted to ex-vivo editable cells (cannot be delivered directly into systemic circulation).",
            "Requires cell harvesting, conditioning chemo-ablation, and autologous re-infusion."
        ]
    },
    "Adeno-Associated Virus (AAV8 / AAV9 Capsid)": {
        "base_efficiency": 82.0,
        "persistence_hours": "Constitutive (Weeks to Months)",
        "persistence_tier": "Prolonged Expression",
        "off_target_risk_factor": 0.85,  # High risk of continuous off-target cleavage over time
        "advantages": [
            "High in-vivo tissue tropism for muscle, retina, and central nervous system.",
            "Established clinical safety record across FDA-approved gene therapies (e.g. Zolgensma)."
        ],
        "warnings": [
            "Persistent episomal Cas9 expression substantially increases long-term off-target double-stranded breaks.",
            "Strict packaging capacity ceiling (<4.7 kb); difficult to fit SpCas9 and guide in one capsid.",
            "Pre-existing neutralizing antibodies (NAbs) in 40-60% of human populations prevent re-dosing."
        ]
    },
    "Engineered Virus-Like Particles (VLP / Nanovesicles)": {
        "base_efficiency": 78.0,
        "persistence_hours": "8 - 16 Hours",
        "persistence_tier": "Transient RNP",
        "off_target_risk_factor": 0.15,
        "advantages": [
            "Combines viral cell entry efficiency with the transient safety of protein delivery.",
            "Package pre-assembled Cas9:sgRNA complexes without transferring viral DNA genomes.",
            "Flexible surface engineering with tissue-targeting antibodies or receptor ligands."
        ],
        "warnings": [
            "Emerging delivery platform; manufacturing scalability currently in early preclinical optimization."
        ]
    }
}


def simulate_in_vivo_delivery(
    cell_type_choice: str,
    delivery_choice: str,
    on_target_score: float,
    high_risk_off_target_count: int
) -> DeliverySimulationResult:
    """
    Computes in-vivo clinical feasibility, net tissue editing efficiency,
    and preclinical FDA IND readiness score.
    """
    cell_info = CELL_TYPE_PROFILES.get(cell_type_choice, list(CELL_TYPE_PROFILES.values())[0])
    delivery_info = DELIVERY_MODALITIES.get(delivery_choice, list(DELIVERY_MODALITIES.values())[0])

    c_access = cell_info["chromatin_accessibility"]
    d_eff = delivery_info["base_efficiency"]
    ot_factor = delivery_info["off_target_risk_factor"]

    # In-vivo effective cutting rate is a product of:
    # (Delivery Transduction Rate) * (Chromatin Accessibility) * (Guide Cleavage Efficacy)
    net_cutting_rate = round((d_eff / 100.0) * (c_access / 100.0) * (on_target_score / 100.0) * 100.0, 1)

    # Preclinical FDA Feasibility Score (0 - 100)
    # Rewards high on-target + delivery; penalizes high-risk off-target mutations multiplied by exposure duration
    base_feasibility = (net_cutting_rate * 0.70) + (d_eff * 0.30)
    # Off-target penalty proportional to delivery persistence
    hazard_penalty = (high_risk_off_target_count * 15.0) * (1.0 + ot_factor)
    final_score = round(max(10.0, min(99.0, base_feasibility - hazard_penalty)), 1)

    if final_score >= 75.0:
        tier = "HIGH_FEASIBILITY"
        rec = "Optimal clinical profile. Candidate meets preclinical clearance criteria for Investigational New Drug (IND) toxicity testing."
    elif final_score >= 50.0:
        tier = "MODERATE_FEASIBILITY"
        rec = "Feasible therapeutic window. Recommend formulating with transient RNP delivery to suppress off-target accumulation."
    else:
        tier = "CHALLENGING"
        rec = "High clinical toxicity risk. Recommend auto-repairing guide sequence or switching from prolonged AAV to transient LNP."

    return DeliverySimulationResult(
        cell_type_name=cell_type_choice,
        delivery_modality=delivery_choice,
        transduction_efficiency_pct=d_eff,
        chromatin_accessibility_pct=c_access,
        cas9_exposure_duration=f"{delivery_info['persistence_hours']} ({delivery_info['persistence_tier']})",
        effective_in_vivo_cutting_pct=net_cutting_rate,
        fda_ind_feasibility_score=final_score,
        feasibility_tier=tier,
        preclinical_advantages=delivery_info["advantages"],
        preclinical_warnings=delivery_info["warnings"],
        regulatory_recommendation=rec
    )
