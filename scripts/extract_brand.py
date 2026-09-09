import pandas as pd

print("Loading dataset...")
df = pd.read_csv("data/raw/twcs.csv")

print(f"Total rows: {len(df)}")

# Get all tweet IDs involved with AppleSupport
apple_replies = df[df['author_id'] == 'AppleSupport'].copy()
print(f"AppleSupport reply tweets: {len(apple_replies)}")

# Get the customer tweets that AppleSupport replied to
customer_tweet_ids = apple_replies['in_response_to_tweet_id'].dropna().astype(int).tolist()
customer_tweets = df[df['tweet_id'].isin(customer_tweet_ids)].copy()
print(f"Matching customer tweets: {len(customer_tweets)}")

# Build conversation pairs
pairs = []
for _, brand_row in apple_replies.iterrows():
    cust_id = brand_row['in_response_to_tweet_id']
    if pd.isna(cust_id):
        continue
    cust_id = int(cust_id)
    cust_rows = customer_tweets[customer_tweets['tweet_id'] == cust_id]
    if cust_rows.empty:
        continue
    cust_row = cust_rows.iloc[0]
    pairs.append({
        'conversation_id': cust_id,
        'customer_message': cust_row['text'],
        'brand_reply': brand_row['text'],
        'customer_author': cust_row['author_id'],
        'created_at': brand_row['created_at']
    })

pairs_df = pd.DataFrame(pairs)
print(f"\nTotal conversation pairs extracted: {len(pairs_df)}")
print("\nSample pairs:")
print(pairs_df[['customer_message', 'brand_reply']].head(5).to_string())

# Save
pairs_df.to_csv("data/processed/apple_conversations.csv", index=False)
print("\nSaved to data/processed/apple_conversations.csv")