import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY was not found.\n"
        "Create a .env file containing:\n"
        "GROQ_API_KEY=your-groq-key"
    )

if not API_KEY.startswith("gsk_"):
    raise RuntimeError(
        "GROQ_API_KEY does not look like a Groq API key.\n"
        "Check your .env file."
    )

# Your Groq account has access to this model.
MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)

INPUT_FILE = "data/evaluation/golden_set.csv"
OUTPUT_FILE = "data/evaluation/golden_set.csv"

# ============================================================
# GROQ CLIENT
# ============================================================

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.groq.com/openai/v1",
    timeout=60.0,
    max_retries=0,
)

# ============================================================
# VALID LABELS
# ============================================================

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

VALID_DECISIONS = {
    "AUTO",
    "ESCALATE",
}


# ============================================================
# PROMPT
# ============================================================

def build_prompt(customer_message, brand_reply):

    return f"""
You are an expert customer-support dataset annotator.

You are labeling customer-support conversations involving AppleSupport.

CUSTOMER MESSAGE:
{customer_message}

BRAND REPLY:
{brand_reply}

Choose EXACTLY ONE gold_intent.

Allowed intents:

1. software_bug
   - iOS/macOS update caused a glitch
   - broken software feature
   - application crash
   - keyboard/text/UI/software malfunction

2. device_performance
   - device is slow
   - device freezes
   - device is lagging
   - poor performance

3. battery_issue
   - battery drains quickly
   - charging problem
   - battery percentage problem
   - battery health problem

4. connectivity_issue
   - Wi-Fi problem
   - Bluetooth problem
   - cellular/network connection problem
   - device cannot connect to a network

5. account_and_services
   - Apple ID
   - iCloud
   - App Store
   - account access
   - activation
   - subscription/service issue

6. hardware_issue
   - physical hardware problem
   - broken screen
   - camera problem
   - speaker problem
   - microphone problem
   - buttons/physical component failure

7. general_question
   - asking how to perform something
   - asking for instructions
   - asking about a feature
   - asking for status/information
   - no clear malfunction

8. complaint_feedback
   - pure frustration
   - criticism
   - general dissatisfaction
   - venting
   - complaint without a specific technical issue

IMPORTANT:
If the message contains a SPECIFIC technical problem, do NOT automatically
choose complaint_feedback just because the customer is angry.

Examples:

"Why won't my iPhone connect to WiFi?"
=> connectivity_issue

"My iPhone keeps freezing after the update."
=> software_bug or device_performance depending on the primary issue

"Apple sucks, terrible service."
=> complaint_feedback

Now choose gold_decision.

AUTO:
- common support problem
- normal troubleshooting can solve it
- clear/general answer exists
- no obvious security, data-loss, or physical-safety concern

ESCALATE:
- security issue
- serious data loss
- account/security compromise
- physical damage requiring repair
- highly unusual situation
- issue cannot reasonably be resolved through normal support guidance

Important:
- Choose the intent based primarily on the customer's actual problem.
- Use the brand reply as supporting context, not as a replacement for the
  customer's problem.
- Do not invent facts that are not present.
- If there is a clear technical issue, prefer the specific technical intent
  over complaint_feedback.
- If the customer is only expressing dissatisfaction without a specific issue,
  use complaint_feedback.
- If the customer is asking for instructions without reporting a malfunction,
  use general_question.

Return ONLY valid JSON.

Do not use markdown.
Do not add explanations outside JSON.

Required format:

{{
  "gold_intent": "one_allowed_intent",
  "gold_decision": "AUTO_or_ESCALATE",
  "gold_escalation_reason": "short reason or empty string"
}}
"""


# ============================================================
# JSON EXTRACTION
# ============================================================

def parse_json_response(text):

    if text is None:
        raise ValueError("Model returned None instead of text")

    text = str(text).strip()

    if not text:
        raise ValueError(
            "Model returned an empty response"
        )

    # --------------------------------------------------------
    # Remove markdown fences if model returns them.
    # --------------------------------------------------------

    if text.startswith("```"):

        lines = text.splitlines()

        cleaned = []

        for line in lines:

            if line.strip().startswith("```"):
                continue

            cleaned.append(line)

        text = "\n".join(cleaned).strip()

    # --------------------------------------------------------
    # First attempt: parse complete response.
    # --------------------------------------------------------

    try:
        return json.loads(text)

    except json.JSONDecodeError:
        pass

    # --------------------------------------------------------
    # Second attempt: extract JSON object from surrounding text.
    # --------------------------------------------------------

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:

        candidate = text[start:end + 1]

        try:
            return json.loads(candidate)

        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"Could not parse JSON response:\n{text}"
    )


# ============================================================
# VALIDATE RESULT
# ============================================================

def validate_result(result):

    if not isinstance(result, dict):

        raise ValueError(
            "Model response is not a JSON object"
        )

    intent = result.get("gold_intent")
    decision = result.get("gold_decision")
    reason = result.get(
        "gold_escalation_reason",
        ""
    )

    if intent not in VALID_INTENTS:

        raise ValueError(
            f"Invalid gold_intent: {intent}"
        )

    if decision not in VALID_DECISIONS:

        raise ValueError(
            f"Invalid gold_decision: {decision}"
        )

    # AUTO should always have an empty escalation reason.
    if decision == "AUTO":
        reason = ""

    return {
        "gold_intent": intent,
        "gold_decision": decision,
        "gold_escalation_reason": str(
            reason or ""
        ),
    }


# ============================================================
# TEST API CONNECTION
# ============================================================

def test_api():

    print("Testing Groq API connection...")
    print(f"Model: {MODEL}")

    try:

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Reply with exactly this JSON and nothing else: "
                        '{"status":"OK"}'
                    ),
                }
            ],
            temperature=0,
            max_tokens=200,
        )

        message = response.choices[0].message

        text = message.content

        print("Groq API connection successful.")

        if text:
            print(
                f"Test response: {repr(text.strip())}"
            )

        else:
            print(
                "Warning: model returned empty text."
            )

            # GPT-OSS models can spend tokens on reasoning.
            if getattr(message, "reasoning", None):

                print(
                    "Reasoning was returned, but no final text "
                    "was produced."
                )

                print(
                    "Reasoning preview:",
                    repr(
                        str(message.reasoning)[:200]
                    )
                )

        print()

        return True

    except Exception as e:

        print()
        print("=" * 60)
        print("GROQ API CONNECTION FAILED")
        print("=" * 60)
        print(str(e))
        print()
        print("Check:")
        print("1. GROQ_API_KEY is correct")
        print("2. The key belongs to Groq")
        print("3. The key has API access")
        print("4. .env is in the project root")
        print("5. GROQ_API_KEY is spelled correctly")
        print("6. GROQ_MODEL is available to your account")
        print("=" * 60)

        return False


# ============================================================
# LABEL ONE MESSAGE
# ============================================================

def label_message(
    customer_message,
    brand_reply
):

    prompt = build_prompt(
        customer_message,
        brand_reply
    )

    last_error = None

    # --------------------------------------------------------
    # Retry temporary failures.
    # --------------------------------------------------------

    for attempt in range(3):

        try:

            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise customer-support "
                            "dataset annotation system. "
                            "Return valid JSON only. "
                            "Do not explain your answer."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0,
                max_tokens=1000,
            )

            message = response.choices[0].message

            text = message.content

            # ------------------------------------------------
            # GPT-OSS can use reasoning tokens.
            # If content is empty, give a useful error instead
            # of attempting to parse None.
            # ------------------------------------------------

            if not text or not text.strip():

                reasoning = getattr(
                    message,
                    "reasoning",
                    None
                )

                if reasoning:

                    raise ValueError(
                        "Model returned reasoning but no final "
                        "JSON content. Increase max_tokens."
                    )

                raise ValueError(
                    "Model returned empty content."
                )

            result = parse_json_response(text)

            return validate_result(result)

        except Exception as e:

            last_error = e

            error_text = str(e)

            # ------------------------------------------------
            # Invalid API key should not be retried.
            # ------------------------------------------------

            if (
                "Incorrect API key" in error_text
                or "invalid_api_key" in error_text
            ):

                raise RuntimeError(
                    "Groq rejected the API key. "
                    "Check GROQ_API_KEY in .env."
                )

            # ------------------------------------------------
            # Model not found should not be retried.
            # ------------------------------------------------

            if (
                "model_not_found" in error_text
                or "Model not found" in error_text
            ):

                raise RuntimeError(
                    f"Groq model '{MODEL}' was not found "
                    "or is not accessible."
                )

            # ------------------------------------------------
            # Wait before retrying temporary failures.
            # ------------------------------------------------

            if attempt < 2:

                wait_time = 2 ** attempt

                print(
                    f"  Retry {attempt + 2}/3 "
                    f"after {wait_time}s..."
                )

                time.sleep(wait_time)

    raise RuntimeError(
        f"Groq request failed after 3 attempts: "
        f"{last_error}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("Loading golden dataset...")

    # --------------------------------------------------------
    # Check dataset.
    # --------------------------------------------------------

    if not os.path.exists(INPUT_FILE):

        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"Found {len(df)} rows.")
    print()

    # --------------------------------------------------------
    # Required columns.
    # --------------------------------------------------------

    required_columns = {
        "customer_message",
        "brand_reply",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # --------------------------------------------------------
    # Convert label columns to object dtype.
    #
    # This prevents the pandas float64 assignment error.
    # --------------------------------------------------------

    for column in [
        "gold_intent",
        "gold_decision",
        "gold_escalation_reason",
    ]:

        if column not in df.columns:

            df[column] = ""

        df[column] = df[column].astype(
            "object"
        )

    # --------------------------------------------------------
    # TEST API BEFORE LABELING.
    # --------------------------------------------------------

    if not test_api():

        return

    print("=" * 60)
    print(
        f"Labeling {len(df)} rows using Groq..."
    )
    print("=" * 60)
    print()

    successful = 0
    failed = 0

    # --------------------------------------------------------
    # PROCESS DATASET.
    # --------------------------------------------------------

    for position, (index, row) in enumerate(
        df.iterrows(),
        start=1
    ):

        customer_message = str(
            row.get(
                "customer_message",
                ""
            )
        )

        brand_reply = str(
            row.get(
                "brand_reply",
                ""
            )
        )

        print(
            f"[{position}/{len(df)}] ",
            end="",
            flush=True
        )

        try:

            result = label_message(
                customer_message,
                brand_reply
            )

            df.at[
                index,
                "gold_intent"
            ] = result[
                "gold_intent"
            ]

            df.at[
                index,
                "gold_decision"
            ] = result[
                "gold_decision"
            ]

            df.at[
                index,
                "gold_escalation_reason"
            ] = result[
                "gold_escalation_reason"
            ]

            successful += 1

            print(
                f"{result['gold_intent']} | "
                f"{result['gold_decision']}"
            )

        except KeyboardInterrupt:

            print()
            print(
                "Interrupted by user."
            )

            print(
                "Saving progress..."
            )

            df.to_csv(
                OUTPUT_FILE,
                index=False
            )

            print(
                f"Progress saved to "
                f"{OUTPUT_FILE}"
            )

            return

        except Exception as e:

            failed += 1

            print()
            print(
                f"  ERROR: {e}"
            )

            # ------------------------------------------------
            # DO NOT create fake labels.
            #
            # Failed rows remain blank and can be retried.
            # ------------------------------------------------

            df.at[
                index,
                "gold_intent"
            ] = ""

            df.at[
                index,
                "gold_decision"
            ] = ""

            df.at[
                index,
                "gold_escalation_reason"
            ] = ""

        # ----------------------------------------------------
        # Save progress every 10 rows.
        # ----------------------------------------------------

        if position % 10 == 0:

            df.to_csv(
                OUTPUT_FILE,
                index=False
            )

            print()
            print(
                f"Progress saved: "
                f"{position}/{len(df)}"
            )
            print()

        # ----------------------------------------------------
        # Small delay between API requests.
        # ----------------------------------------------------

        time.sleep(0.2)

    # --------------------------------------------------------
    # FINAL SAVE.
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # SUMMARY.
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("LABELING COMPLETE")
    print("=" * 60)

    print(
        f"Total rows:      {len(df)}"
    )

    print(
        f"Successful:      {successful}"
    )

    print(
        f"Failed:          {failed}"
    )

    print(
        f"Output file:     {OUTPUT_FILE}"
    )

    # --------------------------------------------------------
    # INTENT DISTRIBUTION.
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("INTENT DISTRIBUTION")
    print("=" * 60)

    print(
        df[
            "gold_intent"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    # --------------------------------------------------------
    # DECISION DISTRIBUTION.
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("DECISION DISTRIBUTION")
    print("=" * 60)

    print(
        df[
            "gold_decision"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    # --------------------------------------------------------
    # SAMPLE RESULTS.
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("SAMPLE RESULTS")
    print("=" * 60)

    print(
        df[
            [
                "customer_message",
                "gold_intent",
                "gold_decision",
                "gold_escalation_reason",
            ]
        ]
        .head(10)
        .to_string()
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()