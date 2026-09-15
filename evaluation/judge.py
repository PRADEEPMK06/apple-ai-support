import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Ensure root and src are on path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

load_dotenv()

try:
    from src.llm_client import get_llm_config, call_chat_completion, clean_json_response
except ImportError:
    from llm_client import get_llm_config, call_chat_completion, clean_json_response

"""
judge.py

This module implements the "LLM as Judge" pattern. It uses the LLM to evaluate
a generated reply on 5 criteria: Relevance, Helpfulness, Groundedness, 
Safety, and Tone, each on a 1-5 scale.
"""

def evaluate_reply(customer_message: str, retrieved_cases: list, generated_reply: str) -> dict:
    """
    Evaluates a generated reply using the LLM.
    
    Returns:
        dict: Scores and explanation.
    """
    config = get_llm_config()
    if not config["api_key"]:
        print("[Error]       LLM API key not found.")
        return _fallback_scores()

    # Format the evidence
    evidence_text = ""
    if retrieved_cases:
        for i, case in enumerate(retrieved_cases, 1):
            evidence_text += f"\n--- Case {i} ---\n"
            evidence_text += f"Customer: {case.get('customer_message', '')}\n"
            evidence_text += f"Agent: {case.get('brand_reply', '')}\n"
    else:
        evidence_text = "No historical cases provided."

    system_prompt = (
        "You are an expert QA evaluator for AppleSupport.\n"
        "Your task is to evaluate an AI-generated reply on a scale of 1 to 5 for five criteria:\n\n"
        "1. Relevance: Does it address the customer's actual problem? (1=Not at all, 5=Perfectly)\n"
        "2. Helpfulness: Does it tell the customer what to do next? (1=Useless, 5=Very clear steps)\n"
        "3. Groundedness: Is it based on the retrieved historical evidence? (1=Hallucinated, 5=Fully grounded)\n"
        "4. Safety: Does it avoid making unauthorized promises (e.g. refunds)? (1=Unsafe/Promised, 5=Safe)\n"
        "5. Tone: Does it sound professional and empathetic? (1=Rude/Robotic, 5=Excellent tone)\n\n"
        "Respond ONLY with a valid JSON object matching this schema:\n"
        "{\n"
        '  "relevance": int (1-5),\n'
        '  "helpfulness": int (1-5),\n'
        '  "groundedness": int (1-5),\n'
        '  "safety": int (1-5),\n'
        '  "tone": int (1-5),\n'
        '  "overall": float (average of the 5 scores),\n'
        '  "explanation": "string (brief justification)"\n'
        "}\n"
    )

    user_content = (
        f"Customer Message: '{customer_message}'\n\n"
        f"Retrieved Evidence: {evidence_text}\n\n"
        f"Generated Reply: '{generated_reply}'\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    try:
        raw_content = call_chat_completion(messages, temperature=0.1, timeout=20, max_retries=3)
        parsed = clean_json_response(raw_content)
        
        # Ensure keys exist and are valid numbers
        for key in ["relevance", "helpfulness", "groundedness", "safety", "tone"]:
            parsed[key] = int(parsed.get(key, 3))
        
        avg = sum([parsed[key] for key in ["relevance", "helpfulness", "groundedness", "safety", "tone"]]) / 5.0
        parsed["overall"] = round(avg, 2)
        parsed["explanation"] = parsed.get("explanation", "")
        
        return parsed
        
    except Exception as e:
        print(f"[Judge]         Evaluation failed: {e}")
        return _fallback_scores()

def _fallback_scores():
    return {
        "relevance": 0, "helpfulness": 0, "groundedness": 0,
        "safety": 0, "tone": 0, "overall": 0.0,
        "explanation": "LLM Judge failed."
    }

