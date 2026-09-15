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

def generate_reply(customer_message: str, intent: str, retrieved_cases: list, low_confidence: bool) -> dict:
    """
    Generates a grounded response using the configured LLM.
    
    Args:
        customer_message (str): The user's query.
        intent (str): The previously classified intent.
        retrieved_cases (list): List of dictionaries containing historical cases.
        low_confidence (bool): Flag indicating if the retriever score was below threshold.
        
    Returns:
        dict: Containing 'reply', 'grounding_summary', and 'used_case_ids'.
    """
    config = get_llm_config()
    if not config["api_key"]:
        print("[Error]       LLM API key not found in environment.")
        return _fallback_reply()

    used_ids = [str(case.get("conversation_id")) for case in retrieved_cases]
    
    # Format the evidence for the prompt
    evidence_text = ""
    for i, case in enumerate(retrieved_cases, 1):
        evidence_text += f"\n--- Historical Case {i} ---\n"
        evidence_text += f"Customer: {case['customer_message']}\n"
        evidence_text += f"AppleSupport: {case['brand_reply']}\n"

    # Base instructions
    system_prompt = (
        "You are an expert customer support agent for AppleSupport replying on Twitter.\n"
        f"The user's message has been classified as: {intent}.\n\n"
    )

    if low_confidence:
        system_prompt += (
            "WARNING: We have LOW confidence in the retrieved historical cases. "
            "They might not perfectly match the user's issue.\n"
            "INSTRUCTIONS:\n"
            "1. Generate a very safe, generic reply.\n"
            "2. Apologize for the issue.\n"
            "3. Ask the user for more details or direct them to DM (Direct Message) us.\n"
            "4. Do NOT provide specific troubleshooting steps or invent solutions.\n"
            "5. Do NOT promise replacements, refunds, or account unlocks.\n"
        )
    else:
        system_prompt += (
            "INSTRUCTIONS:\n"
            "1. Write a reply to the customer using ONLY the historical evidence provided below.\n"
            "2. The reply must be concise and sound like a Twitter support response (friendly, professional, < 280 chars if possible).\n"
            "3. If the historical data directs them to a link or to DM, you should do the same.\n"
            "4. Do NOT promise refunds, replacements, or account actions unless it explicitly happened in the historical evidence.\n"
            "5. Do NOT invent URLs or troubleshooting steps outside the evidence.\n"
        )
        
    system_prompt += (
        "\nProvide your response ONLY as a JSON object matching this exact schema:\n"
        "{\n"
        '  "reply": "string (the actual message to send to the customer)",\n'
        '  "grounding_summary": "string (brief explanation of how you used the historical evidence to form this reply)"\n'
        "}\n\n"
        "HISTORICAL EVIDENCE:"
        f"{evidence_text}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Customer Message: '{customer_message}'"}
    ]

    print("[Generator]     Asking LLM to draft reply...")
    
    try:
        raw_content = call_chat_completion(messages, temperature=0.2, timeout=20, max_retries=3)
        parsed = clean_json_response(raw_content)
        
        reply = parsed.get("reply", "")
        summary = parsed.get("grounding_summary", "")
        
        if not reply:
            raise ValueError("Generated reply is empty.")
            
        return {
            "reply": reply,
            "grounding_summary": summary,
            "used_case_ids": used_ids
        }
        
    except Exception as e:
        print(f"[Generator]     Reply generation failed: {e}. Using generic fallback.")
        return _fallback_reply(used_ids)

def _fallback_reply(used_ids=None) -> dict:
    if used_ids is None:
        used_ids = []
    return {
        "reply": "We'd like to help look into this with you. Please send us a DM so we can discuss further.",
        "grounding_summary": "System defaulted to a safe generic reply due to API failure or missing keys.",
        "used_case_ids": used_ids
    }

# Small test snippet if run directly
if __name__ == "__main__":
    test_msg = "My battery drains so fast after iOS update"
    fake_cases = [
        {
            "conversation_id": "12345",
            "customer_message": "Battery is dead fast after updating iOS 15.",
            "brand_reply": "We'd like to help. Battery life can fluctuate after an update. Check out this article: apple.co/battery and DM us if it continues."
        }
    ]
    res = generate_reply(test_msg, "battery_issue", fake_cases, low_confidence=False)
    print("Result:", json.dumps(res, indent=2))
