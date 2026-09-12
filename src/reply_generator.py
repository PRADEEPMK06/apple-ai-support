import os
import json
import time
import requests
from dotenv import load_dotenv

"""
reply_generator.py

This module takes the customer message, its classified intent, and retrieved 
historical cases from our semantic index. It uses Grok LLM to generate a reply
that is strictly grounded in the historical evidence. It ensures no false promises
are made and handles low-confidence scenarios safely.
"""

load_dotenv()

def generate_reply(customer_message: str, intent: str, retrieved_cases: list, low_confidence: bool) -> dict:
    """
    Generates a grounded response using Grok LLM.
    
    Args:
        customer_message (str): The user's query.
        intent (str): The previously classified intent.
        retrieved_cases (list): List of dictionaries containing historical cases.
        low_confidence (bool): Flag indicating if the retriever score was below threshold.
        
    Returns:
        dict: Containing 'reply', 'grounding_summary', and 'used_case_ids'.
    """
    api_key = os.getenv("GROK_API_KEY")
    if not api_key:
        print("[Error]       GROK_API_KEY not found in environment.")
        return _fallback_reply()

    url = "https://api.x.ai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

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

    payload = {
        "model": "grok-3-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Customer Message: '{customer_message}'"}
        ],
        "temperature": 0.2, # Keep hallucination risk low
    }

    print("[Generator]     Asking Grok to draft reply...")
    
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=20)
            response.raise_for_status()
            
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            
            if content.startswith("```json"):
                content = content[7:-3].strip()
            elif content.startswith("```"):
                content = content[3:-3].strip()
                
            parsed = json.loads(content)
            reply = parsed.get("reply", "")
            summary = parsed.get("grounding_summary", "")
            
            if not reply:
                raise ValueError("Generated reply is empty.")
                
            return {
                "reply": reply,
                "grounding_summary": summary,
                "used_case_ids": used_ids
            }
            
        except (requests.RequestException, json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"[Generator]     Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                time.sleep(2)
            else:
                print("[Generator]     All retries failed. Using generic fallback.")
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
