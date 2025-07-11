import re
import string
from typing import List, Dict, Any
import numpy as np
from datetime import datetime

def preprocess_text(text: str) -> str:
    """Preprocess text by removing extra whitespace, converting to lowercase."""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text.strip())
    text = text.lower()
    text = re.sub(r'[^\w\s.,?!]', '', text)
    return text

def calculate_trust_score(similarity_scores: List[float], fact_check_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate trust score based on similarity scores and fact-check results."""
    if not similarity_scores or not fact_check_results:
        return {
            "trust_score": 50,
            "confidence": 0,
            "explanation": "No similar claims found in database"
        }
    
    verdict_scores = {
        "true": 100, "mostly true": 85, "half true": 50,
        "mostly false": 25, "false": 0, "pants on fire": 0
    }
    
    total_weight = 0
    weighted_score = 0
    
    for i, (score, result) in enumerate(zip(similarity_scores, fact_check_results)):
        weight = score * (1.0 / (i + 1))
        verdict = result.get('verdict', '').lower()
        verdict_score = verdict_scores.get(verdict, 50)
        confidence_multiplier = result.get('confidence_score', 80) / 100
        adjusted_score = verdict_score * confidence_multiplier
        weighted_score += adjusted_score * weight
        total_weight += weight
    
    trust_score = weighted_score / total_weight if total_weight > 0 else 50
    max_similarity = max(similarity_scores)
    avg_similarity = sum(similarity_scores) / len(similarity_scores)
    confidence = min(100, (max_similarity * 70) + (avg_similarity * 20) + (len(similarity_scores) * 3))
    
    if trust_score >= 80:
        credibility = "highly credible"
    elif trust_score >= 60:
        credibility = "mostly credible"
    elif trust_score >= 40:
        credibility = "questionable"
    elif trust_score >= 20:
        credibility = "likely false"
    else:
        credibility = "highly unreliable"
    
    explanation = f"This claim appears to be {credibility}. Found {len(fact_check_results)} similar fact-checked claims."
    
    return {
        "trust_score": round(trust_score, 1),
        "confidence": round(confidence, 1),
        "explanation": explanation
    }

def validate_claim_input(claim: str) -> Dict[str, Any]:
    """Validate user input claim."""
    if not claim or not claim.strip():
        return {"valid": False, "error": "Claim cannot be empty"}
    if len(claim.strip()) < 10:
        return {"valid": False, "error": "Claim must be at least 10 characters long"}
    if len(claim.strip()) > 1000:
        return {"valid": False, "error": "Claim cannot exceed 1000 characters"}
    return {"valid": True, "error": None}
