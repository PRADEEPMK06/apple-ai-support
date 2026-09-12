import os
import json
import requests
import time
from dotenv import load_dotenv

"""
escalation.py

This module determines whether a customer message can be safely handled by the AI (AUTO)
or if it requires a human agent (ESCALATE). It applies a strict set of rules in order.
If a message does not trigger any hardcoded rules, it falls back to the Grok LLM to
make a borderline judgment.
"""

load_dotenv()

def determine_escalation(customer_message: str, intent: str, intent_confidence: float, retrieval_confidence: float) -> dict:
    """
    Decides whether to AUTO handle or ESCALATE a case based on predefined rules, 
    falling back to an LLM for borderline cases.
    
    Args:
        customer_message (str): The raw customer tweet.
        intent (str): The classified intent.
        intent_confidence (float): Confidence of the intent classification.
        retrieval_confidence (float): Highest similarity score from retrieval (if < 0.3, it means low confidence).
        
    Returns:
        dict: {"decision": "AUTO" or "ESCALATE", "reason": "human readable reason"}
    """
    
    msg_lower = customer_message.lower()
    
    # ---------------------------------------------------------
    # RULE SET 1: ALWAYS ESCALATE (High risk scenarios)
    # ---------------------------------------------------------
    
    # 1. Account security issues
    if intent == "account_and_services" and any(word in msg_lower for word in ["hacked", "compromised", "unauthorized", "stolen"]):
        return {"decision": "ESCALATE", "reason": "High risk account security issue mentioned."}
        
    # 2. Legal threats
    if any(word in msg_lower for word in ["lawsuit", "legal action", "lawyer"]):
        return {"decision": "ESCALATE", "reason": "Customer mentioned legal action or lawsuit."}
        
    # 3. Data loss
    if any(phrase in msg_lower for phrase in ["data lost", "everything deleted", "factory reset accidentally"]):
        return {"decision": "ESCALATE", "reason": "Customer reported severe data loss."}
        
    # 4. Physical damage asking for replacement
    has_damage = any(word in msg_lower for word in ["physical damage", "cracked", "broken screen", "water damage"])
    has_replacement_request = "replacement" in msg_lower
    if has_damage and has_replacement_request:
        return {"decision": "ESCALATE", "reason": "Customer requesting replacement for physical/water damage."}
        
    # 5. Low intent confidence
    if intent_confidence < 0.4:
        return {"decision": "ESCALATE", "reason": f"Intent classification confidence too low ({intent_confidence:.2f} < 0.4)."}
        
    # 6. Low retrieval confidence (unless it's just a general question)
    if retrieval_confidence < 0.3 and intent != "general_question":
        return {"decision": "ESCALATE", "reason": f"Retrieval confidence too low ({retrieval_confidence:.2f} < 0.3) for a technical issue."}

    # ---------------------------------------------------------
    # RULE SET 2: AUTO HANDLE (Safe scenarios)
    # ---------------------------------------------------------
    
    if intent in ["general_question", "complaint_feedback"]:
        return {"decision": "AUTO", "reason": f"Safe intent category: {intent}."}
        
    if intent in ["software_bug", "device_performance", "battery_issue", "connectivity_issue"]:
        if retrieval_confidence >= 0.5:
            return {"decision": "AUTO", "reason": f"Technical intent ({intent}) with high retrieval confidence ({retrieval_confidence:.2f} >= 0.5)."}

    # ---------------------------------------------------------
    # RULE SET 3: BORDERLINE (Use Grok to decide)
    # ---------------------------------------------------------
    
    print("[Escalation]    Case is borderline. Consulting Grok LLM...")
    return _llm_escalation_decision(customer_message, intent)

def _llm_escalation_decision(customer_message: str, intent: str) -> dict:
    """Uses Grok to decide escalation for borderline cases."""
    api_key = os.getenv("GROK_API_KEY")
    if not api_key:
        print("[Error]       GROK_API_KEY not found. Defaulting to ESCALATE for safety.")
        return {"decision": "ESCALATE", "reason": "API key missing for borderline evaluation."}

    url = "https://api.x.ai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    system_prompt = (
        "You are an escalation manager for AppleSupport.\n"
        "Review the customer's message and their predicted intent.\n"
        "Decide if an AI agent can safely reply (AUTO) or if a human must take over (ESCALATE).\n"
        "Rules for ESCALATE: complex troubleshooting needed, very angry customer, unique unresolvable hardware issues.\n"
        "Rules for AUTO: standard troubleshooting, mild frustration, clear questions.\n"
        "Respond ONLY with a valid JSON object matching this schema:\n"
        "{\n"
        '  "decision": "AUTO" or "ESCALATE",\n'
        '  "reason": "string (short explanation of why)"\n'
        "}\n"
    )

    payload = {
        "model": "grok-3-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Intent: {intent}\nCustomer Message: '{customer_message}'"}
        ],
        "temperature": 0.1,
    }

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            
            if content.startswith("```json"):
                content = content[7:-3].strip()
            elif content.startswith("```"):
                content = content[3:-3].strip()
                
            parsed = json.loads(content)
            decision = parsed.get("decision", "ESCALATE")
            reason = parsed.get("reason", "No reason provided by LLM")
            
            if decision not in ["AUTO", "ESCALATE"]:
                decision = "ESCALATE"
                reason = "LLM returned invalid decision, defaulting to ESCALATE."
                
            return {"decision": decision, "reason": reason}
            
        except (requests.RequestException, json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"[Escalation]    Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                time.sleep(2)
            else:
                print("[Escalation]    All retries failed. Defaulting to ESCALATE.")
                return {"decision": "ESCALATE", "reason": "LLM failure, safe fallback applied."}

# Small test snippet if run directly
if __name__ == "__main__":
    test_msg = "My battery drains so fast after iOS update"
    # Should AUTO because battery_issue and high retrieval confidence
    res1 = determine_escalation(test_msg, "battery_issue", 0.9, 0.8)
    print("Test 1 (High Confidence Battery):", res1)
    
    # Should ESCALATE because of legal threat
    test_msg2 = "I am filing a lawsuit if you don't fix my broken screen"
    res2 = determine_escalation(test_msg2, "hardware_issue", 0.9, 0.8)
    print("Test 2 (Legal Threat):", res2)
    
    # Should go to Borderline LLM
    test_msg3 = "I don't know what is going on but my phone is acting very weird today"
    res3 = determine_escalation(test_msg3, "device_performance", 0.8, 0.4) # retrieval < 0.5 but > 0.3
    print("Test 3 (Borderline):", res3)
