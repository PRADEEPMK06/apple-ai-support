import pandas as pd
import numpy as np

"""
retrieval_agent.py

Evaluates whether the dataset is suitable for semantic embeddings and retrieval.
Checks text length variance and vocabulary richness.
"""

class RetrievalSuitabilityAgent:
    def analyze(self, df: pd.DataFrame, report) -> None:
        print("[Retrieval Agent] Evaluating semantic search suitability...")
        if df is None or df.empty:
            return
            
        msgs = df['customer_message'].astype(str)
        
        # Calculate lengths
        lengths = msgs.str.split().str.len()
        avg_len = lengths.mean()
        std_len = lengths.std()
        
        score = 80 # base score
        
        # Too short messages lack semantic density
        if avg_len < 5:
            score -= 30
            rec = "Poor for semantic search (messages too short)"
        elif avg_len < 10:
            score -= 10
            rec = "Acceptable, but messages are somewhat brief"
        else:
            score += 10
            rec = "Excellent for semantic search"
            
        # Variance indicates diverse phrasing
        if std_len > 10:
            score += 10
            
        # Cap at 100
        score = min(100, max(0, int(score)))
        
        report.retrieval_suitability_score = score
        report.retrieval_recommendation = rec
        print(f"[Retrieval Agent] Score: {score}/100 - {rec}")
