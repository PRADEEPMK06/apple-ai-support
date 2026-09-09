import pandas as pd

df = pd.read_csv("data/processed/apple_conversations.csv")

print(f"Total pairs: {len(df)}")
print("\n--- 50 RANDOM CUSTOMER MESSAGES ---\n")

# Show 50 random customer messages so we can read and group them
sample = df['customer_message'].sample(50, random_state=42).tolist()
for i, msg in enumerate(sample, 1):
    print(f"{i}. {msg}")
    print()