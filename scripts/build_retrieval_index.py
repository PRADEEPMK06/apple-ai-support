import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import sys

"""
build_retrieval_index.py

This script processes the historical AppleSupport conversations and builds a semantic
retrieval index. It strictly filters out any conversations that are part of the golden
evaluation set to prevent data leakage. Embeddings are generated using the 
sentence-transformers/all-MiniLM-L6-v2 model.
"""

def main():
    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / "data" / "processed"
    eval_dir = project_root / "data" / "evaluation"
    
    corpus_path = processed_dir / "apple_conversations.csv"
    golden_path = eval_dir / "golden_set.csv"
    
    # Check if files exist
    if not corpus_path.exists():
        print(f"[Error]       Could not find {corpus_path}")
        sys.exit(1)
    if not golden_path.exists():
        print(f"[Error]       Could not find {golden_path}")
        sys.exit(1)

    print(f"[Loading]       Loading corpus from {corpus_path.name}...")
    corpus_df = pd.read_csv(corpus_path)
    initial_len = len(corpus_df)
    print(f"[Loading]       Loading corpus: {initial_len} rows")

    print("[Filtering]     Removing golden set rows...")
    golden_df = pd.read_csv(golden_path)
    golden_ids = set(golden_df["conversation_id"].astype(str))
    
    # Filter out golden ids
    corpus_df["conversation_id_str"] = corpus_df["conversation_id"].astype(str)
    corpus_df = corpus_df[~corpus_df["conversation_id_str"].isin(golden_ids)].copy()
    corpus_df.drop(columns=["conversation_id_str"], inplace=True)
    
    final_len = len(corpus_df)
    print(f"[Corpus]        Final corpus size: {final_len} rows (Removed {initial_len - final_len} rows)")

    # Reset index so it aligns perfectly with embeddings array
    corpus_df.reset_index(drop=True, inplace=True)

    print("[Embedding]     Loading sentence-transformers model (all-MiniLM-L6-v2)...")
    # Using all-MiniLM-L6-v2 as requested
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    print("[Embedding]     Generating embeddings...")
    # Extract the customer messages to encode
    texts = corpus_df["customer_message"].fillna("").tolist()
    
    # We will generate in batches to show progress
    batch_size = 1000
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = model.encode(batch_texts, show_progress_bar=False)
        all_embeddings.append(batch_embeddings)
        
        # Show progress every 1000 rows
        progress = min(i + batch_size, len(texts))
        print(f"[Embedding]     Progress: {progress} / {final_len} rows...")

    # Concatenate all batches
    embeddings_matrix = np.vstack(all_embeddings)

    out_embeddings_path = processed_dir / "embeddings.npy"
    out_corpus_path = processed_dir / "corpus.csv"

    print(f"[Saving]        Saving {out_embeddings_path.name}...")
    np.save(out_embeddings_path, embeddings_matrix)

    print(f"[Saving]        Saving {out_corpus_path.name}...")
    corpus_df.to_csv(out_corpus_path, index=False)

    print("[Complete]      Retrieval index ready.")

if __name__ == "__main__":
    main()
