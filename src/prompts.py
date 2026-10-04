"""
Prompt engineering, conversation builders, and generative output parsers
for SAGE Qwen2.5-VL agricultural diagnosis.
"""

import re
import string
from typing import Dict, Any, Optional, List, Tuple


def is_valid_field(value: Any) -> bool:
    """Checks if a metadata value is non-null, non-empty, and informative."""
    if value is None:
        return False
    val_str = str(value).strip()
    if not val_str:
        return False
    if val_str.lower() in ("nan", "none", "null", "undefined", "n/a", "unknown", ""):
        return False
    return True


def clean_text(text: Any) -> str:
    """Cleans up whitespace and linebreaks in metadata strings."""
    if not is_valid_field(text):
        return ""
    # Replace multiple whitespaces/newlines with single space
    return " ".join(str(text).split())


def format_user_prompt(row: Dict[str, Any]) -> str:
    """
    Builds the user instruction prompt incorporating available context (crop, plant organ).
    Does NOT hallucinate absent fields.
    """
    lines = ["Analyze this agricultural image and identify the most likely crop disease."]
    
    crop = row.get("crop")
    if is_valid_field(crop):
        lines.append(f"Crop: {clean_text(crop)}")
        
    organ = row.get("plant_organ")
    if is_valid_field(organ):
        lines.append(f"Plant organ: {clean_text(organ)}")
        
    return "\n".join(lines)


def format_assistant_target(row: Dict[str, Any]) -> str:
    """
    Constructs the ground truth assistant response.
    Order:
      Crop: <crop>
      Diagnosis: <canonical_disease>
      Disease type: <disease_type>
      Plant organ: <plant_organ>
      Visual symptoms: <visual_symptoms>
      Pathogen: <pathogen>
    
    Strictly omits absent fields to prevent hallucination.
    Never includes provenance fields (symptom_source, symptom_quote, filename, raw_label).
    """
    lines = []
    
    # Crop
    if is_valid_field(row.get("crop")):
        lines.append(f"Crop: {clean_text(row['crop'])}")
        
    # Primary Diagnosis: canonical_disease
    canonical_disease = row.get("canonical_disease")
    if not is_valid_field(canonical_disease):
        # Fallback to disease if canonical_disease is missing
        canonical_disease = row.get("disease", "Unknown Disease")
    lines.append(f"Diagnosis: {clean_text(canonical_disease)}")
    
    # Disease Type
    if is_valid_field(row.get("disease_type")):
        lines.append(f"Disease type: {clean_text(row['disease_type'])}")
        
    # Plant Organ
    if is_valid_field(row.get("plant_organ")):
        lines.append(f"Plant organ: {clean_text(row['plant_organ'])}")
        
    # Visual Symptoms
    if is_valid_field(row.get("visual_symptoms")):
        lines.append(f"Visual symptoms: {clean_text(row['visual_symptoms'])}")
        
    # Pathogen
    if is_valid_field(row.get("pathogen")):
        lines.append(f"Pathogen: {clean_text(row['pathogen'])}")
        
    return "\n".join(lines)


def build_conversation(
    row: Dict[str, Any],
    image_obj: Any,
    include_target: bool = True
) -> List[Dict[str, Any]]:
    """
    Formats an example into official Qwen2.5-VL multi-turn conversation format.
    """
    user_text = format_user_prompt(row)
    
    user_content = [
        {"type": "image", "image": image_obj},
        {"type": "text", "text": user_text}
    ]
    
    messages = [
        {"role": "user", "content": user_content}
    ]
    
    if include_target:
        target_text = format_assistant_target(row)
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": target_text}]
        })
        
    return messages


def normalize_disease_name(text: Optional[str]) -> str:
    """
    Normalizes disease names for canonical evaluation:
    - strips whitespace
    - lowercase
    - removes leading/trailing punctuation
    - normalizes internal spaces and hyphens
    """
    if not text:
        return ""
    s = str(text).strip().lower()
    # Remove surrounding quotes, brackets, punctuation
    s = s.strip(string.punctuation + " ")
    # Replace multiple hyphens/underscores/spaces with a single space
    s = re.sub(r"[\s\-_]+", " ", s)
    return s.strip()


def parse_generated_response(response_text: str) -> Dict[str, str]:
    """
    Robustly parses key diagnostic fields from Qwen's generated text output.
    Extracts:
      - crop
      - diagnosis (canonical disease)
      - disease_type
      - plant_organ
      - visual_symptoms
      - pathogen
      - raw_text
    """
    extracted: Dict[str, str] = {
        "crop": "",
        "diagnosis": "",
        "disease_type": "",
        "plant_organ": "",
        "visual_symptoms": "",
        "pathogen": "",
        "raw_text": response_text.strip()
    }
    
    # Patterns for key fields (case-insensitive)
    patterns = {
        "diagnosis": r"(?:Diagnosis|Disease|Identified Disease|Canonical Disease)\s*:\s*([^\n]+)",
        "crop": r"(?:Crop|Host Crop|Plant)\s*:\s*([^\n]+)",
        "disease_type": r"(?:Disease\s*type|Type)\s*:\s*([^\n]+)",
        "plant_organ": r"(?:Plant\s*organ|Organ|Part)\s*:\s*([^\n]+)",
        "visual_symptoms": r"(?:Visual\s*symptoms|Symptoms|Visual\s*signs)\s*:\s*([^\n]+)",
        "pathogen": r"(?:Pathogen|Causal\s*Agent)\s*:\s*([^\n]+)",
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            val = match.group(1).strip()
            # Clean up trailing punctuation or prefixes
            val = val.rstrip(".,;")
            extracted[key] = val
            
    # If diagnosis was not found via regex prefix, fallback to first non-empty line
    if not extracted["diagnosis"]:
        lines = [line.strip() for line in response_text.splitlines() if line.strip()]
        if lines:
            first_line = lines[0]
            # If line has colon, take right side, else full line
            if ":" in first_line:
                extracted["diagnosis"] = first_line.split(":", 1)[1].strip()
            else:
                extracted["diagnosis"] = first_line
                
    return extracted
