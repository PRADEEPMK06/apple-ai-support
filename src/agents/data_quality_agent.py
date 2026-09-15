import pandas as pd
import numpy as np

"""
data_quality_agent.py

Evaluates the raw extracted dataframe for missing values, duplicates, and noise.
Generates a deterministic Data Quality Score.
"""

class DataQualityAgent:
    def analyze(self, df: pd.DataFrame, report) -> None:
        print("[Quality Agent] Analyzing dataset quality...")
        
        if df is None or df.empty:
            report.data_quality_score = 0
            return
            
        total = len(df)
        
        # Missing data check (already dropped in extraction, but good to check if raw passed)
        missing = df['customer_message'].isna().sum() + df['brand_reply'].isna().sum()
        missing_rate = missing / (total * 2)
        if missing_rate < 0.01:
            report.missing_data_level = "Low"
        elif missing_rate < 0.05:
            report.missing_data_level = "Medium"
        else:
            report.missing_data_level = "High"
            
        # Duplicates check
        dup_rate = df.duplicated(subset=['customer_message']).sum() / total
        if dup_rate < 0.05:
            report.duplicates_level = "Low"
        elif dup_rate < 0.15:
            report.duplicates_level = "Medium"
        else:
            report.duplicates_level = "High"
            
        # Noise check (very short messages)
        short_msgs = df[df['customer_message'].str.len() < 10]
        noise_rate = len(short_msgs) / total
        if noise_rate < 0.05:
            report.noise_level = "Low"
        elif noise_rate < 0.15:
            report.noise_level = "Medium"
        else:
            report.noise_level = "High"
            
        # Calculate score (100 base, subtract penalties)
        score = 100
        if report.missing_data_level == "Medium": score -= 5
        if report.missing_data_level == "High": score -= 20
        if report.duplicates_level == "Medium": score -= 5
        if report.duplicates_level == "High": score -= 15
        if report.noise_level == "Medium": score -= 5
        if report.noise_level == "High": score -= 15
        
        # Bonus for size
        if total > 50000:
            score = min(100, score + 5)
            
        report.data_quality_score = max(0, score)
        print(f"[Quality Agent] Score: {report.data_quality_score}/100")
