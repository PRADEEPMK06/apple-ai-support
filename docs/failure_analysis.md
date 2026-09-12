# Failure Analysis: AI Support Agent

This document analyzes the top 5 failure modes observed during the evaluation of the AI Agent pipeline on the golden dataset. Understanding these edge cases is critical for improving the system before production deployment.

---

## 1. Sarcastic Complaints Misclassified as Technical Bugs

- **Real example from golden set:** *"Wow, great job Apple! My phone gets so hot I can use it to fry an egg. Best feature of iOS 15 so far! 🍳🙄"*
- **What the system predicted:** `device_performance` (Decision: AUTO)
- **What the correct answer was:** `complaint_feedback` (Decision: ESCALATE or AUTO depending on human nuance)
- **Why this failure likely happened:** The LLM intent classifier anchored heavily on the keywords "hot", "phone", and "iOS 15" which are strong indicators of device performance. It failed to grasp the sarcastic tone which makes this a pure complaint rather than a standard troubleshooting request.
- **How severe this failure is:** **Medium**. The customer is likely very frustrated, and sending them an automated link about "How to keep your iPhone at acceptable operating temperatures" will only make them angrier.
- **What could fix it:** Add a few-shot examples of sarcasm in the intent classifier's system prompt to help Grok identify sarcastic tone and route it to `complaint_feedback` or explicitly lower the confidence so the Escalation Engine passes it to a human.

---

## 2. Multi-Intent Messages Ignoring the Core Request

- **Real example from golden set:** *"I accidentally dropped my phone in the pool yesterday and now it won't turn on. Also can you tell me how to cancel my Apple Arcade subscription?"*
- **What the system predicted:** `account_and_services` (Decision: AUTO)
- **What the correct answer was:** `hardware_issue` (Decision: ESCALATE)
- **Why this failure likely happened:** The customer presented two issues: a critical hardware failure (water damage) and a simple billing question. The LLM classifier is forced to pick exactly ONE intent. It happened to latch onto the clearly phrased subscription question and ignored the water damage.
- **How severe this failure is:** **High**. The system bypassed the strict Escalation Engine rule for "water damage" because the intent was classified as `account_and_services`. It auto-handled the billing question while completely ignoring a dead, water-damaged device.
- **What could fix it:** Modify the intent classifier to output a list of intents if multiple exist, or instruct it to prioritize hardware/critical failures over generic questions. The Escalation Engine should also scan for critical keywords (like "water damage") *regardless* of the classified intent.

---

## 3. Ambiguous Boundary Between Performance and Battery Issues

- **Real example from golden set:** *"Ever since I updated, my phone is dying by 2 PM and everything lags when I try to open the camera."*
- **What the system predicted:** `device_performance`
- **What the correct answer was:** `battery_issue`
- **Why this failure likely happened:** Both intents are present in the text ("dying by 2 PM" = battery, "everything lags" = performance). In the training taxonomy, these are separate categories, causing confusion for both the LLM and the retrieval system.
- **How severe this failure is:** **Low**. The retrieval system generally finds historical cases that cover both battery and performance optimization (which share overlapping troubleshooting steps like "check for rogue apps"). The generated reply remains helpful.
- **What could fix it:** Merge `battery_issue` and `device_performance` into a single `device_health` intent, or allow multi-label classification.

---

## 4. Retrieval Confidence Drops on Vague or Short Queries

- **Real example from golden set:** *"Fix it now. I'm sick of this."*
- **What the system predicted:** `complaint_feedback` with `retrieval_confidence` = 0.22
- **What the correct answer was:** `complaint_feedback`
- **Why this failure likely happened:** The message is so short and devoid of technical nouns that its semantic embedding does not strongly match any specific historical case. As a result, the retrieval confidence falls below our 0.3 threshold.
- **How severe this failure is:** **Low to Medium**. While the classification is correct, the low retrieval confidence forces the Reply Generator to output a very generic fallback ("Please DM us so we can look into this").
- **What could fix it:** This is actually behaving safely by design (falling back to a human/DM). However, we could improve it by having the AI politely ask clarifying questions instead of an immediate generic DM prompt: *"I'd like to help! Could you provide a bit more detail about what needs fixing?"*

---

## 5. False Alarms on the "Legal Action" Keyword Rule

- **Real example from golden set:** *"I work as a lawyer and I use my iPad for everything, but the wifi keeps dropping in court!"*
- **What the system predicted:** `connectivity_issue` (Decision: ESCALATE)
- **What the correct answer was:** `connectivity_issue` (Decision: AUTO)
- **Why this failure likely happened:** The Escalation Engine uses simple substring matching (`if "lawyer" in msg_lower`). It saw the word "lawyer" and immediately triggered the "Legal Threat" escalation rule, assuming the customer was threatening to sue Apple.
- **How severe this failure is:** **Low**. It results in a false positive for escalation. A human agent will step in and easily handle the wifi issue. Missing an escalation is dangerous, but over-escalating just wastes a bit of agent time.
- **What could fix it:** Replace the crude regex/keyword substring matching in `escalation.py` with an LLM-based boolean check, or refine the keyword list to look for stronger phrases like `"speaking to my lawyer"` or `"class action lawsuit"`.
