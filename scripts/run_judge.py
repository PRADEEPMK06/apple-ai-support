import sys
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.pipeline import run_pipeline
from evaluation.judge import evaluate_reply

"""
run_judge.py

This script samples 30 cases from the golden set, generates replies using
the pipeline, and evaluates them with the LLM Judge. It also creates a blank
human scoring template, or computes agreement metrics if human scores are present.
"""

def main():
    eval_dir = project_root / "data" / "evaluation"
    golden_path = eval_dir / "golden_set.csv"
    template_path = eval_dir / "human_scores_template.csv"
    results_path = eval_dir / "llm_judge_scores.csv"

    print(f"[Loading]       Loading golden set from {golden_path.name}...")
    df = pd.read_csv(golden_path)

    # Check if template exists and is partially/fully filled
    if template_path.exists():
        human_df = pd.read_csv(template_path)
        # Check if human scores are present (not NaN)
        if human_df['human_relevance'].notna().sum() > 0:
            print("[Metrics]       Found human scores! Calculating agreement metrics...")
            calculate_agreement(human_df, pd.read_csv(results_path))
            return

    # Otherwise, generate the template
    print("[Sampling]      Sampling 30 rows for LLM vs Human Judge comparison...")
    sample_df = df.sample(n=30, random_state=42).copy()

    records = []
    
    print("[Evaluating]    Running pipeline and LLM Judge on sample...")
    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Judging Replies"):
        msg = row["customer_message"]
        conv_id = row["conversation_id"]
        
        # 1. Run pipeline to get the AI Agent's reply
        pipeline_res = run_pipeline(msg)
        reply = pipeline_res.get("reply", "")
        cases = pipeline_res.get("retrieved_cases", [])
        
        # 2. Run LLM Judge
        judge_res = evaluate_reply(msg, cases, reply)
        
        record = {
            "conversation_id": conv_id,
            "customer_message": msg,
            "generated_reply": reply,
            "llm_relevance": judge_res["relevance"],
            "llm_helpfulness": judge_res["helpfulness"],
            "llm_groundedness": judge_res["groundedness"],
            "llm_safety": judge_res["safety"],
            "llm_tone": judge_res["tone"],
            "llm_overall": judge_res["overall"],
            "llm_explanation": judge_res["explanation"]
        }
        records.append(record)

    results_df = pd.DataFrame(records)
    
    print(f"[Saving]        Saving LLM scores to {results_path.name}...")
    results_df.to_csv(results_path, index=False)

    # Create the template for human scoring
    template_cols = [
        "conversation_id", "customer_message", "generated_reply",
        "human_relevance", "human_helpfulness", "human_groundedness",
        "human_safety", "human_tone"
    ]
    template_df = results_df[["conversation_id", "customer_message", "generated_reply"]].copy()
    
    for col in ["human_relevance", "human_helpfulness", "human_groundedness", "human_safety", "human_tone"]:
        template_df[col] = "" # Leave empty for human
        
    print(f"[Saving]        Saving human template to {template_path.name}...")
    template_df.to_csv(template_path, index=False)
    
    print("\n" + "="*60)
    print("✅ JUDGE GENERATION COMPLETE")
    print("="*60)
    print("To compare Human vs LLM scores:")
    print("1. Open data/evaluation/human_scores_template.csv")
    print("2. Fill in scores (1-5) for the human_* columns")
    print("3. Run this script again (python scripts/run_judge.py)")

def calculate_agreement(human_df, llm_df):
    """Calculates agreement metrics if human scores exist."""
    merged = pd.merge(human_df, llm_df, on="conversation_id", suffixes=('_human', '_llm'))
    merged = merged.dropna(subset=['human_relevance']) # drop rows human didn't score yet
    
    metrics = ["relevance", "helpfulness", "groundedness", "safety", "tone"]
    
    print("\n--- Agreement Metrics ---")
    
    for m in metrics:
        human_scores = pd.to_numeric(merged[f"human_{m}"])
        llm_scores = merged[f"llm_{m}"]
        
        mae = np.mean(np.abs(human_scores - llm_scores))
        
        # Pearson correlation (handle edge case if std is 0)
        if human_scores.std() == 0 or llm_scores.std() == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(human_scores, llm_scores)[0, 1]
            
        print(f"{m.capitalize():<12}: MAE = {mae:.2f} | Correlation = {corr:.2f}")

if __name__ == "__main__":
    main()
