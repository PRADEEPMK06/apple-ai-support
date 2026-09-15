from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class DatasetReport:
    """
    Data model to pass evaluation metrics between the various analysis agents.
    """
    requested_brand: str = ""
    extraction_success: bool = False
    extraction_error: str = ""
    
    total_conversations: int = 0
    
    data_quality_score: int = 0
    missing_data_level: str = "Unknown"
    duplicates_level: str = "Unknown"
    noise_level: str = "Unknown"
    
    intent_diversity_score: int = 0
    major_categories: List[str] = field(default_factory=list)
    
    retrieval_suitability_score: int = 0
    retrieval_recommendation: str = ""
    
    response_quality_score: int = 0
    response_quality_summary: str = ""
    
    overall_score: int = 0
    final_recommendation: str = ""
    recommendation_reason: str = ""
