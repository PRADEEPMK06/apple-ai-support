import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# ── helpers ──────────────────────────────────────────────────────────────────

def clean_text(text):
    """Remove @mentions, URLs, and extra spaces."""
    text = re.sub(r'@\w+', '', str(text))
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

# ── Baseline 1: Keyword Rules ─────────────────────────────────────────────────

KEYWORD_RULES = {
    'battery_issue': [
        'battery', 'charging', 'charge', 'drain', 'percentage', 'power'
    ],
    'software_bug': [
        'bug', 'glitch', 'update', 'ios', 'macos', 'fix', 'autocorrect',
        'crash', 'broken', 'issue with update'
    ],
    'device_performance': [
        'slow', 'freezing', 'freeze', 'lag', 'lagging', 'unresponsive',
        'stuck', 'hanging'
    ],
    'connectivity_issue': [
        'wifi', 'wi-fi', 'bluetooth', 'network', 'connection', 'connect',
        'internet', 'signal'
    ],
    'account_and_services': [
        'apple id', 'icloud', 'app store', 'itunes', 'activate', 'activation',
        'login', 'sign in', 'password', 'account'
    ],
    'hardware_issue': [
        'speaker', 'camera', 'screen', 'microphone', 'button', 'hardware',
        'physical', 'broken screen'
    ],
    'general_question': [
        'how do i', 'how to', 'what is', 'when will', 'where is',
        'can i', 'is there a way', 'eta', 'expected'
    ],
}

def keyword_classify(text):
    """Baseline 1: simple keyword matching."""
    cleaned = clean_text(text)
    scores = {}
    for intent, keywords in KEYWORD_RULES.items():
        scores[intent] = sum(1 for kw in keywords if kw in cleaned)
    best_intent = max(scores, key=scores.get)
    if scores[best_intent] == 0:
        return 'complaint_feedback'
    return best_intent

# ── Baseline 2: TF-IDF Nearest Neighbor ──────────────────────────────────────

class TFIDFClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.train_texts = []
        self.train_labels = []
        self.matrix = None

    def fit(self, texts, labels):
        self.train_texts = [clean_text(t) for t in texts]
        self.train_labels = list(labels)
        self.matrix = self.vectorizer.fit_transform(self.train_texts)

    def predict(self, text):
        cleaned = clean_text(text)
        vec = self.vectorizer.transform([cleaned])
        sims = cosine_similarity(vec, self.matrix)[0]
        best_idx = np.argmax(sims)
        return self.train_labels[best_idx], float(sims[best_idx])


# ── Quick demo ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_messages = [
        "My battery is draining so fast after the update",
        "WiFi keeps disconnecting every few minutes",
        "Phone is freezing and very slow",
        "Can't sign into my Apple ID",
        "Speaker stopped working on my iPhone 8",
        "How do I check my battery health?",
        "This update is absolutely terrible, worst Apple has ever done",
        "iOS 11 has a bug where autocorrect breaks completely",
    ]

    print("=" * 60)
    print("BASELINE 1: Keyword Classifier")
    print("=" * 60)
    for msg in test_messages:
        intent = keyword_classify(msg)
        print(f"  [{intent}]\n  {msg}\n")