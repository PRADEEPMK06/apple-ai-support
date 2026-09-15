# AppleSupport AI Support Agent

This project is a take-home assignment for the SDE Intern role at Hiver. It implements a complete, end-to-end AI Support Agent designed to ingest incoming customer tweets for AppleSupport, classify their intent, retrieve semantically similar historical conversations, generate a grounded reply, and intelligently decide whether the case should be handled automatically or escalated to a human agent.

---

## 1. Problem Statement

Real customer support teams receive thousands of Twitter messages every day. Reading every message, figuring out what the customer wants, writing a reply, and deciding whether a human agent needs to step in takes enormous time and effort. 

The goal of this project is to build an AI agent that can reliably automate standard support responses without hallucinating or making false promises. It must anchor its knowledge in actual historical brand replies and know its own limitations by escalating borderline or critical cases to human experts.

## 2. Solution Overview

The system processes incoming customer messages through a multi-stage pipeline:

```text
  CUSTOMER TWEET
       │
       ▼
  ┌─────────────────────┐
  │  Intent Classifier   │  ← Grok LLM classifies into 1 of 8 intents
  └─────────────────────┘
       │ (Intent + Confidence)
       ▼
  ┌─────────────────────┐
  │  Retrieval System    │  ← Uses sentence-transformers (all-MiniLM-L6-v2)
  │  (Semantic Search)   │    to find top 3 similar historical cases
  └─────────────────────┘
       │ (Historical Evidence + Confidence Score)
       ▼
  ┌─────────────────────┐
  │  Reply Generator     │  ← Grok LLM drafts a concise Twitter reply 
  │  (Grounded by data)  │    anchored strictly in the retrieved evidence
  └─────────────────────┘
       │ (Draft Reply)
       ▼
  ┌─────────────────────┐
  │  Escalation Engine   │  ← Rules + LLM judge if safe to AUTO-reply or
  └─────────────────────┘    if it requires ESCALATE to a human
       │
       ▼
  FINAL OUTPUT JSON
```

## 3. Setup Instructions

You can run this entire pipeline locally in under 15 minutes.

**Prerequisites:**
- Python 3.12+
- A valid Grok (xAI) API Key

**Steps:**
1. Clone this repository and navigate to the project root.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the root directory and add your API key:
   ```env
   GROK_API_KEY=your_api_key_here
   ```
5. Test the pipeline via the demo script:
   ```bash
   python scripts/demo.py "My battery is dying so fast after the update!"
   ```

## 4. Dataset and Brand Selection Rationale

This project utilizes the publicly available **Customer Support on Twitter** dataset (2.8M tweets).
I selected **AppleSupport** as the target brand, extracting ~106,648 conversation pairs.
**Rationale:** AppleSupport handles a massive volume of highly technical inquiries spanning both hardware and software. The data is rich in standardized troubleshooting links and distinct issue categories, making it a perfect candidate for semantic retrieval and intent classification.

## 5. Intent Taxonomy Summary

Based on data exploration, incoming messages are classified into 8 core intents:
1. `software_bug`: iOS/app glitches.
2. `device_performance`: Phone lagging/freezing.
3. `battery_issue`: Draining fast, won't charge.
4. `connectivity_issue`: Wi-Fi, cellular drops.
5. `account_and_services`: Apple ID, iCloud lockouts.
6. `hardware_issue`: Broken screens, physical damage.
7. `general_question`: How-to's, release dates.
8. `complaint_feedback`: General dissatisfaction.

## 6. How the Golden Set Was Created

To properly evaluate the system without data leakage, I generated a "Golden Set":
- Sampled 200 random rows from the processed AppleSupport conversations.
- Removed these 200 rows from the main corpus before generating retrieval embeddings.
- Used an automated Grok LLM script (`autolabel_golden.py`) to label the `gold_intent` and `gold_decision`.
- Manually reviewed a subset to ensure label quality.

## 7. Results Comparison Table

We evaluated the system against two baselines (Keyword Matching and TF-IDF Nearest Neighbor).

| System                 | Int. Acc | Int. F1  | Esc. Acc | Esc. Rec |
|------------------------|----------|----------|----------|----------|
| System 1 (Keyword)     | 34.5%    | 32.1%    | 40.0%    | 12.0%    |
| System 2 (TF-IDF)      | 42.0%    | 39.8%    | 45.5%    | 18.5%    |
| System 3 (AI Agent)    | 89.0%    | 88.5%    | 85.0%    | 95.0%    |

*Note: The AI Agent's Escalation Recall (95.0%) is critical. Missing an escalation (false negative) means an angry customer or complex issue is wrongly given an automated reply.*

## 8. Reply Quality Evaluation (LLM as Judge)

We evaluated the actual generated text using Grok as an LLM Judge across 30 samples:
- **Relevance:** 4.8 / 5.0
- **Helpfulness:** 4.5 / 5.0
- **Groundedness:** 4.9 / 5.0 (Safely avoided hallucinations)
- **Safety:** 5.0 / 5.0 (No unauthorized promises)
- **Tone:** 4.7 / 5.0

## 9. What is Misleading About the Headline Number

The 89.0% Intent Accuracy looks fantastic on paper, but **it is artificially inflated**. 
- The golden set was auto-labeled by Grok, and our classifier also uses Grok. We are essentially asking the model if it agrees with itself, which naturally inflates the score compared to purely human-annotated ground truth.
- Furthermore, Twitter data is noisy. Many tweets contain multiple intents (e.g., "Phone is slow AND my screen is cracked"). Measuring "accuracy" against a single correct label fails to capture the nuance of multi-intent real-world queries.

## 10. Top 5 Failure Modes Summary

1. **Sarcastic Complaints Misclassified:** The AI often reads sarcasm ("Great update, phone is a brick") as a literal performance issue rather than a pure complaint.
2. **Multi-Intent Blindspots:** If a user mentions a billing issue alongside a critical hardware failure, the system might classify it as billing and fail to escalate the hardware issue.
3. **Battery vs Performance Overlap:** Ambiguity causes flip-flopping between these two intents, though retrieval generally smooths this over.
4. **Low Retrieval on Vague Queries:** Extremely short complaints ("Fix it now") yield low retrieval confidence, forcing generic fallback replies.
5. **Regex False Alarms:** The escalation engine over-indexes on words like "lawyer", escalating even when the context isn't a legal threat.

## 11. What I Would Do With One More Week

1. **Implement RAG Reranking:** Right now, I just take the top 3 cosine similarities. I'd add a Cross-Encoder to rerank the top 10 results for better contextual relevance.
2. **Multi-Label Classification:** Allow the intent classifier to return a list of intents to solve the multi-intent failure mode.
3. **Human-in-the-loop UI:** Build a simple Streamlit dashboard where human agents can review the AI's drafted reply and click "Approve" or "Edit".
4. **Few-Shot Prompting Tuning:** Add specific examples of sarcasm to the intent classifier prompt to improve robustness.

## 12. Decision Log

1. **Used `sentence-transformers/all-MiniLM-L6-v2` instead of OpenAI embeddings:** It runs locally, is free, fast, and sufficient for Twitter-length text retrieval.
2. **Excluded golden set from corpus:** Strict requirement to prevent the model from retrieving the exact answer it's being tested on (data leakage).
3. **JSON enforcement via prompting:** Grok was instructed with strict JSON schemas and low temperatures (0.1) to ensure robust pipeline data parsing.
4. **Multi-retry logic on API calls:** Added 3 retries with backoff because LLM APIs are notorious for random timeout or 503 errors.
5. **Fallback values instead of crashing:** If classification fails entirely, it defaults to `complaint_feedback` rather than raising a fatal exception.
6. **Hardcoded Escalation Rules first:** Instead of purely trusting the LLM to decide escalation, I put regex-based safety nets (e.g., "hacked", "lawsuit") *before* the LLM.
7. **Used `numpy` and `pandas` instead of vector DB:** For 100k rows, `numpy` cosine similarity executes in milliseconds. Setting up Pinecone or Milvus would add unnecessary architectural complexity for an intern project.
8. **Removed URLs and @mentions in baselines:** Twitter handles and links skew TF-IDF distributions without adding semantic value.
9. **Prioritized Escalation Recall over Precision:** It's much safer for the business to have a human review a false alarm than to have a bot wrongly handle a critical hardware meltdown.
10. **Passed `low_confidence` flag to Generator:** Instead of letting the LLM hallucinate when retrieval fails, I explicitly tell it when the evidence is weak so it can generate a safe "Please DM us" fallback.
11. **Did not use LangChain:** LangChain obscures prompts and adds heavy overhead. Writing native `requests` blocks makes the pipeline more transparent and easier to debug.
12. **Kept replies under 280 characters:** Enforced via prompt to match Twitter's native constraints.
13. **Used `macro` F1 for evaluation:** Because the intent classes might be imbalanced, `macro` treats all classes equally, giving a truer picture of performance on rare edge cases.
14. **Groundedness as a primary LLM Judge metric:** We care more that the agent didn't invent a fake Apple policy than whether its grammar was perfectly poetic.
15. **Separated Pipeline from Demo:** Placed the core logic in `src/` and the entry points in `scripts/` to maintain a clean, production-like directory structure.
