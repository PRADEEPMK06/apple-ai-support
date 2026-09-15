import sys
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.baselines import keyword_classify, TFIDFClassifier, KEYWORD_RULES
from src.pipeline import run_pipeline
from src.escalation import determine_escalation
from evaluation.metrics import evaluate_classification, evaluate_escalation

"""
run_evaluation.py

This script runs the keyword baseline, TF-IDF baseline, and full AI pipeline
on the golden set. It calculates performance metrics and outputs a comparison table.
"""

# Set UTF-8 encoding for standard output if supported
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def main():
    golden_path = project_root / "data" / "evaluation" / "golden_set.csv"
    results_json_path = project_root / "data" / "evaluation" / "results.json"
    results_csv_path = project_root / "data" / "evaluation" / "results.csv"

    print(f"[Loading]     Loading golden set from {golden_path.name}...")
    df = pd.read_csv(golden_path)

    # 1. Prepare TF-IDF Baseline
    # We will fit it on the keyword definitions since we don't have labeled training data
    print("[Setup]       Training TF-IDF Baseline on intent taxonomy keywords...")
    tfidf = TFIDFClassifier()
    texts = [" ".join(words) for words in KEYWORD_RULES.values()]
    labels = list(KEYWORD_RULES.keys())
    tfidf.fit(texts, labels)

    y_true_intent = df["gold_intent"].tolist()
    y_true_decision = df["gold_decision"].tolist()

    sys1_intent = []
    sys1_decision = []
    
    sys2_intent = []
    sys2_decision = []
    
    sys3_intent = []
    sys3_decision = []

    print("[Evaluating]  Running all 3 systems on the golden set (this will take a while)...")
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating Rows"):
        msg = row["customer_message"]
        
        # --- System 1: Keyword Baseline ---
        s1_int = keyword_classify(msg)
        sys1_intent.append(s1_int)
        # Use simple escalation rule with dummy confidences for baselines
        s1_dec = determine_escalation(msg, s1_int, 1.0, 1.0)["decision"]
        sys1_decision.append(s1_dec)
        
        # --- System 2: TF-IDF Baseline ---
        s2_int, s2_conf = tfidf.predict(msg)
        sys2_intent.append(s2_int)
        s2_dec = determine_escalation(msg, s2_int, s2_conf, 1.0)["decision"]
        sys2_decision.append(s2_dec)
        
        # --- System 3: AI Agent ---
        s3_res = run_pipeline(msg)
        sys3_intent.append(s3_res["intent"])
        sys3_decision.append(s3_res["decision"])

    print("[Metrics]     Calculating scores...")
    intent_labels = list(KEYWORD_RULES.keys()) + ["complaint_feedback"]
    intent_labels = list(set(intent_labels))

    res1_int = evaluate_classification(y_true_intent, sys1_intent, intent_labels)
    res1_dec = evaluate_escalation(y_true_decision, sys1_decision)
    
    res2_int = evaluate_classification(y_true_intent, sys2_intent, intent_labels)
    res2_dec = evaluate_escalation(y_true_decision, sys2_decision)
    
    res3_int = evaluate_classification(y_true_intent, sys3_intent, intent_labels)
    res3_dec = evaluate_escalation(y_true_decision, sys3_decision)

    # Compile results
    results = {
        "System 1 (Keyword)": {
            "intent_accuracy": res1_int["accuracy"],
            "intent_f1_macro": res1_int["f1_macro"],
            "escalation_accuracy": res1_dec["accuracy"],
            "escalation_recall": res1_dec["recall_escalate"]
        },
        "System 2 (TF-IDF)": {
            "intent_accuracy": res2_int["accuracy"],
            "intent_f1_macro": res2_int["f1_macro"],
            "escalation_accuracy": res2_dec["accuracy"],
            "escalation_recall": res2_dec["recall_escalate"]
        },
        "System 3 (AI Agent)": {
            "intent_accuracy": res3_int["accuracy"],
            "intent_f1_macro": res3_int["f1_macro"],
            "escalation_accuracy": res3_dec["accuracy"],
            "escalation_recall": res3_dec["recall_escalate"]
        }
    }

    print(f"[Saving]      Saving results to {results_json_path.name}...")
    with open(results_json_path, "w") as f:
        json.dump(results, f, indent=4)
        
    df_results = pd.DataFrame(results).T
    df_results.to_csv(results_csv_path)

    # Print clean comparison table
    print("\n" + "─"*60)
    print(f"{'System':<22} | {'Int. Acc':<8} | {'Int. F1':<8} | {'Esc. Acc':<8} | {'Esc. Rec':<8}")
    print("─"*60)
    for sys_name, mets in results.items():
        print(f"{sys_name:<22} | {mets['intent_accuracy']*100:>7.1f}% | {mets['intent_f1_macro']*100:>7.1f}% | {mets['escalation_accuracy']*100:>7.1f}% | {mets['escalation_recall']*100:>7.1f}%")
    print("─"*60)

if __name__ == "__main__":
    main()
