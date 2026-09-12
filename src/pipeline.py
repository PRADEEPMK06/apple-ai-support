import sys
from pathlib import Path

# Add the project root to sys.path so we can import from src
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.intent_classifier import classify_intent
from src.retriever import get_retriever
from src.reply_generator import generate_reply
from src.escalation import determine_escalation

"""
pipeline.py

This module orchestrates the complete end-to-end AI support agent. It connects
intent classification, semantic retrieval, reply generation, and escalation 
decision-making into a single pipeline.
"""

def run_pipeline(customer_message: str) -> dict:
    """
    Executes the full AI support agent pipeline.
    
    Args:
        customer_message (str): The raw tweet from the customer.
        
    Returns:
        dict: A comprehensive dictionary containing outputs from all pipeline stages.
    """
    # Step 1: Classify Intent
    intent_result = classify_intent(customer_message)
    intent = intent_result.get("intent")
    intent_confidence = intent_result.get("confidence", 0.0)
    
    # Step 2: Retrieve Historical Cases
    retriever = get_retriever()
    retrieved_cases, low_retrieval_confidence = retriever.get_similar_cases(customer_message)
    
    # Calculate retrieval confidence (highest similarity score)
    retrieval_confidence = 0.0
    if retrieved_cases:
        retrieval_confidence = max(case["similarity_score"] for case in retrieved_cases)
        
    # Step 3: Generate Grounded Reply
    reply_result = generate_reply(
        customer_message=customer_message,
        intent=intent,
        retrieved_cases=retrieved_cases,
        low_confidence=low_retrieval_confidence
    )
    
    # Step 4: Escalation Decision
    escalation_result = determine_escalation(
        customer_message=customer_message,
        intent=intent,
        intent_confidence=intent_confidence,
        retrieval_confidence=retrieval_confidence
    )
    
    # Step 5: Assemble Final Output
    return {
        "customer_message": customer_message,
        "intent": intent,
        "intent_confidence": intent_confidence,
        "retrieval_confidence": retrieval_confidence,
        "retrieved_cases": retrieved_cases,
        "reply": reply_result.get("reply"),
        "grounding_summary": reply_result.get("grounding_summary"),
        "decision": escalation_result.get("decision"),
        "escalation_reason": escalation_result.get("reason")
    }
