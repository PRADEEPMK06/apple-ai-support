import os
import json
import time
import requests
from dotenv import load_dotenv

"""
intent_classifier.py

This module uses the Grok LLM to read a customer message and classify its intent
into one of 8 predefined categories. It uses a structured prompt, handles retries 
on API failures, and validates the output. If all retries fail, it falls back
safely to a default intent.
"""

# Load environment variables (e.g., GROK_API_KEY)
load_dotenv()

VALID_INTENTS = {
    "software_bug",
    "device_performance",
    "battery_issue",
    "connectivity_issue",
    "account_and_services",
    "hardware_issue",
    "general_question",
    "complaint_feedback"
}

def classify_intent(customer_message: str) -> dict:
    """
    Classifies the customer message into one of 8 intents using Grok LLM.
    
    Args:
        customer_message (str): The text from the customer.
        
    Returns:
        dict: A dictionary containing 'intent', 'confidence', and 'reason'.
    """
    api_key = os.getenv("GROK_API_KEY")
    if not api_key:
        print("[Error]       GROK_API_KEY not found in environment.")
        return _fallback_response("API key missing")

    url = "https://api.x.ai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Define the prompt
    system_prompt = (
        "You are an expert customer support intent classifier for AppleSupport.\n"
        "Your task is to classify the user's message into exactly ONE of the following 8 intents:\n"
        "- software_bug: Issues with iOS, apps crashing, glitches.\n"
        "- device_performance: Phone freezing, lagging, running slow, overheating.\n"
        "- battery_issue: Battery draining fast, not charging, battery health.\n"
        "- connectivity_issue: Wi-Fi, Bluetooth, Cellular, dropping calls.\n"
        "- account_and_services: Apple ID, iCloud, billing, subscriptions, locked out.\n"
        "- hardware_issue: Broken screen, physical damage, buttons not working, water damage.\n"
        "- general_question: How-to questions, feature inquiries, generic queries.\n"
        "- complaint_feedback: General dissatisfaction without a specific technical request.\n\n"
        "Respond ONLY with a valid JSON object matching this schema:\n"
        "{\n"
        '  "intent": "string (must be one of the 8 intents)",\n'
        '  "confidence": "float (0.0 to 1.0)",\n'
        '  "reason": "string (short explanation)"\n'
        "}\n"
    )

    payload = {
        "model": "grok-3-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Customer Message: '{customer_message}'"}
        ],
        "temperature": 0.1,  # Low temperature for more deterministic JSON output
    }

    print("[Classifier]    Sending message to Grok...")
    
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            
            # Clean up potential markdown formatting if the model wraps JSON in code blocks
            if content.startswith("```json"):
                content = content[7:-3].strip()
            elif content.startswith("```"):
                content = content[3:-3].strip()
                
            parsed = json.loads(content)
            
            # Validate output
            intent = parsed.get("intent", "")
            confidence = parsed.get("confidence", 0.0)
            reason = parsed.get("reason", "No reason provided")
            
            if intent not in VALID_INTENTS:
                raise ValueError(f"Invalid intent returned: {intent}")
                
            # Convert confidence to float just in case
            confidence = float(confidence)
            
            print(f"[Classifier]    Intent detected: {intent} (confidence: {confidence:.2f})")
            return {
                "intent": intent,
                "confidence": confidence,
                "reason": reason
            }
            
        except (requests.RequestException, json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"[Classifier]    Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                time.sleep(2)  # short backoff before retry
            else:
                print("[Classifier]    All retries failed. Using fallback.")
                return _fallback_response(str(e))

def _fallback_response(error_message: str) -> dict:
    """Provides a safe default response if classification completely fails."""
    return {
        "intent": "complaint_feedback",
        "confidence": 0.0,
        "reason": f"Fallback due to error: {error_message}"
    }

# Small test snippet if run directly
if __name__ == "__main__":
    test_msg = "My battery is dying so fast since I updated to the new iOS!"
    result = classify_intent(test_msg)
    print("Result:", json.dumps(result, indent=2))
