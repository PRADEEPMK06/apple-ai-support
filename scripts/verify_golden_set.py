import pandas as pd
import sys
from pathlib import Path

"""
verify_golden_set.py

This script verifies the integrity and correctness of the golden evaluation set
before it is used to measure the performance of our AI support agent. It checks
for completeness of required labels, validity of intent categories, and provides
a summary distribution of the dataset.

Expected dataset: data/evaluation/golden_set.csv
"""

# Define the valid intent categories as per docs/intent_taxonomy.md
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

VALID_DECISIONS = {"AUTO", "ESCALATE"}

def main():
    # Define file path
    project_root = Path(__file__).resolve().parent.parent
    golden_set_path = project_root / "data" / "evaluation" / "golden_set.csv"

    print(f"[Loading]     Loading {golden_set_path.name}...")
    
    try:
        df = pd.read_csv(golden_set_path)
    except FileNotFoundError:
        print(f"[Error]       Could not find {golden_set_path}")
        sys.exit(1)

    total_rows = len(df)
    print(f"[Checking]    Loaded {total_rows} rows. Checking intent labels...")
    
    errors = []

    # Check 1: All rows have gold_intent filled
    missing_intent = df[df["gold_intent"].isna() | (df["gold_intent"] == "")]
    if not missing_intent.empty:
        for idx, row in missing_intent.iterrows():
            errors.append(f"Row {idx} (conversation_id: {row.get('conversation_id', 'Unknown')}): Missing gold_intent")

    # Check 2: All rows have gold_decision filled
    print("[Checking]    Checking decision labels...")
    missing_decision = df[df["gold_decision"].isna() | (df["gold_decision"] == "")]
    if not missing_decision.empty:
        for idx, row in missing_decision.iterrows():
            errors.append(f"Row {idx} (conversation_id: {row.get('conversation_id', 'Unknown')}): Missing gold_decision")

    # Check 3: No invalid intents exist outside the 8 categories
    # Also dropna to avoid errors with invalid check if already missing
    invalid_intents = df[df["gold_intent"].notna() & ~df["gold_intent"].isin(VALID_INTENTS)]
    if not invalid_intents.empty:
        for idx, row in invalid_intents.iterrows():
            errors.append(f"Row {idx} (conversation_id: {row.get('conversation_id', 'Unknown')}): Invalid intent '{row['gold_intent']}'")

    # Check 4: Validate decision labels (optional but good practice)
    invalid_decisions = df[df["gold_decision"].notna() & ~df["gold_decision"].isin(VALID_DECISIONS)]
    if not invalid_decisions.empty:
        for idx, row in invalid_decisions.iterrows():
            errors.append(f"Row {idx} (conversation_id: {row.get('conversation_id', 'Unknown')}): Invalid decision '{row['gold_decision']}'")

    print("[Stats]       Showing distributions...")
    
    # Show intent distribution
    print("\n--- Intent Distribution ---")
    if "gold_intent" in df.columns:
        intent_counts = df["gold_intent"].value_counts(dropna=False)
        for intent, count in intent_counts.items():
            print(f"  {intent}: {count}")

    # Show decision distribution
    print("\n--- Decision Distribution ---")
    if "gold_decision" in df.columns:
        decision_counts = df["gold_decision"].value_counts(dropna=False)
        for decision, count in decision_counts.items():
            print(f"  {decision}: {count}")

    print("\n---------------------------")

    if total_rows != 200:
        errors.append(f"Expected 200 rows in golden set, found {total_rows}")

    # Result
    if errors:
        print("[Result]      FAIL — golden set has errors:")
        for err in errors:
            print(f"              - {err}")
        sys.exit(1)
    else:
        print("[Result]      PASS — golden set is valid")

if __name__ == "__main__":
    main()
