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


VALID_INTENTS = {
    "software_bug",
    "device_performance",
    "battery_issue",
    "connectivity_issue",
    "account_and_services",
    "hardware_issue",
    "general_question",
    "complaint_feedback",
}


def classify_intent(customer_message: str) -> dict:
    """
    Classifies the customer message into one of 8 intents using the configured LLM (Groq / Grok).

    Args:
        customer_message (str): The text from the customer.

    Returns:
        dict: A dictionary containing 'intent', 'confidence', and 'reason'.
    """

    config = get_llm_config()
    if not config["api_key"]:
        print("[Error]       LLM API key not found in environment.")
        return _fallback_response("API key missing")

    print(f"[Classifier]    Using model: {config['model']}")

    system_prompt = (
        "You are an expert customer support intent classifier for AppleSupport.\n\n"
        "Your task is to classify the user's message into exactly ONE of the following 8 intents:\n\n"
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

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Customer Message: '{customer_message}'"}
    ]

    print("[Classifier]    Sending message to LLM...")

    try:
        raw_content = call_chat_completion(messages, temperature=0.1, timeout=20, max_retries=3)
        parsed = clean_json_response(raw_content)

        intent = parsed.get("intent", "")
        confidence = parsed.get("confidence", 0.0)
        reason = parsed.get("reason", "No reason provided")

        if intent not in VALID_INTENTS:
            # Check if intent contains any of the valid intents as a substring
            found_intent = None
            for vi in VALID_INTENTS:
                if vi in intent:
                    found_intent = vi
                    break
            if found_intent:
                intent = found_intent
            else:
                raise ValueError(f"Invalid intent returned: {intent}")

        # Convert confidence to float
        confidence = float(confidence)
        confidence = max(0.0, min(1.0, confidence))

        print(f"[Classifier]    Intent detected: {intent} (confidence: {confidence:.2f})")

        return {
            "intent": intent,
            "confidence": confidence,
            "reason": reason,
        }

    except Exception as e:
        print(f"[Classifier]    Classification failed: {e}")
        return _fallback_response(str(e))


def _fallback_response(error_message: str) -> dict:
    """
    Provides a safe default response if classification completely fails.
    """

    return {
        "intent": "complaint_feedback",
        "confidence": 0.0,
        "reason": f"Fallback due to error: {error_message}",
    }


# -------------------------------------------------------------
# Test when this file is executed directly
# -------------------------------------------------------------
if __name__ == "__main__":

    test_msg = (
        "My battery is dying so fast "
        "since I updated to the new iOS!"
    )

    result = classify_intent(test_msg)

    print(
        "Result:",
        json.dumps(result, indent=2)
    )