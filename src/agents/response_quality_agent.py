import pandas as pd

"""
response_quality_agent.py

Evaluates the brand replies. If they are highly repetitive or very short,
they make poor grounding material for the LLM reply generator.
"""

class ResponseQualityAgent:
    def analyze(self, df: pd.DataFrame, report) -> None:
        print("[Response Quality Agent] Evaluating historical response quality...")
        if df is None or df.empty:
            return
            
        replies = df['brand_reply'].astype(str)
        
        # Check uniqueness (are they all canned responses?)
        total = len(replies)
        unique_count = replies.nunique()
        uniqueness_ratio = unique_count / total
        
        # Check links (good support often involves linking to docs)
        has_link = replies.str.contains(r'http[s]?://', regex=True).sum()
        link_ratio = has_link / total
        
        score = 60 # base
        
        # Uniqueness modifiers
        if uniqueness_ratio > 0.8:
            score += 25
        elif uniqueness_ratio > 0.4:
            score += 15
        else:
            score -= 20
            
        # Link modifiers
        if link_ratio > 0.1:
            score += 15
            
        score = min(100, max(0, int(score)))
        
        if score > 75:
            summary = "Historical responses are highly unique and suitable for grounded AI generation."
        elif score > 50:
            summary = "Historical responses are acceptable, but contain some repetitive templates."
        else:
            summary = "Historical responses are highly repetitive and may lead to robotic AI replies."
            
        report.response_quality_score = score
        report.response_quality_summary = summary
        print(f"[Response Quality Agent] Score: {score}/100 - {summary}")
