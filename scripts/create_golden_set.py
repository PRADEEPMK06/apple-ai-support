import pandas as pd

df = pd.read_csv("data/processed/apple_conversations.csv")

# Sample 200 examples with a fixed seed so it's reproducible
sampled = df.sample(200, random_state=99).reset_index(drop=True)

# Add empty columns for manual labeling
sampled['gold_intent'] = ''
sampled['gold_decision'] = ''       # AUTO or ESCALATE
sampled['gold_escalation_reason'] = ''
sampled['notes'] = ''

# Keep only what we need
golden = sampled[[
    'conversation_id',
    'customer_message',
    'brand_reply',
    'gold_intent',
    'gold_decision',
    'gold_escalation_reason',
    'notes'
]]

golden.to_csv("data/evaluation/golden_set.csv", index=False)
print(f"Saved {len(golden)} examples to data/evaluation/golden_set.csv")
print("\nFirst 5 messages to label:")
for i, row in golden.head(5).iterrows():
    print(f"\n{i+1}. {row['customer_message']}")