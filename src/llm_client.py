import os
import json
import re
import time
import requests
from dotenv import load_dotenv

"""
llm_client.py

Unified LLM interface supporting Groq and xAI Grok.
Handles API key discovery, model mapping, base URL resolution,
and robust JSON response parsing.
"""

load_dotenv()

def get_llm_config() -> dict:
    """
    Resolves the LLM API key, endpoint URL, and model name.
    Automatically detects Groq (gsk_...) vs xAI Grok (xai-...) keys
    and applies sensible model fallbacks.
    """
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY")
    
    # Model selection from env
    model = os.getenv("GROQ_MODEL") or os.getenv("GROK_MODEL") or "openai/gpt-oss-120b"
    
    # Normalize model names if user supplied short name
    if model in ["gpt-oss-120b", "gpt-oss-20b"]:
        model = f"openai/{model}"
    elif model in ["qwen3.8-27b"]:
        model = f"qwen/{model}"
        
    # Endpoint URL selection
    if api_key and api_key.startswith("xai-"):
        url = "https://api.x.ai/v1/chat/completions"
        if model.startswith("openai/") or model.startswith("qwen/"):
            model = "grok-3-mini"
    else:
        url = "https://api.groq.com/openai/v1/chat/completions"
        # If model is an xAI grok model name but key is Groq, use Groq default
        if "grok" in model.lower():
            model = "openai/gpt-oss-120b"

    return {
        "api_key": api_key,
        "url": url,
        "model": model
    }

def clean_json_response(content: str) -> dict:
    """
    Extracts and parses JSON from LLM response text,
    handling markdown blocks and surrounding text safely.
    """
    if not content:
        raise ValueError("Empty response content from LLM.")
        
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()
    
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Search for JSON object structure { ... }
        match = re.search(r"(\{[\s\S]*\})", content)
        if match:
            return json.loads(match.group(1))
        raise

def call_chat_completion(messages: list, temperature: float = 0.1, timeout: int = 20, max_retries: int = 3) -> str:
    """
    Calls the chat completions API with retries and exponential backoff.
    """
    config = get_llm_config()
    api_key = config["api_key"]
    if not api_key:
        raise ValueError("No LLM API key found. Please set GROQ_API_KEY or GROK_API_KEY in your .env file.")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": config["model"],
        "messages": messages,
        "temperature": temperature
    }

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(config["url"], headers=headers, json=payload, timeout=timeout)
            if not response.ok:
                try:
                    err_json = response.json()
                    err_msg = err_json.get("error", err_json)
                except Exception:
                    err_msg = response.text
                raise requests.HTTPError(f"HTTP {response.status_code}: {err_msg}")
            
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(2)
            else:
                raise last_err
