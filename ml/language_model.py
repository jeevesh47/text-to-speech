"""
Language model integration for ISL Sign-to-Speech.

Reconstructs natural English sentences from recognized ISL sign sequences.
Supports Google Gemini, OpenAI, and a reliable offline fallback.
"""

from __future__ import annotations

import os
import json
import urllib.request
import urllib.error
from typing import Sequence


def get_active_provider() -> str:
    """Return the active LLM provider name: 'google', 'openai', or 'fallback'."""
    explicit = os.getenv("LLM_PROVIDER", "").strip().lower()
    if explicit in {"google", "gemini"}:
        return "google"
    if explicit in {"openai"}:
        return "openai"
    if explicit in {"fallback", "none"}:
        return "fallback"

    # Auto-detection from environment variables
    if os.getenv("GEMINI_API_KEY"):
        return "google"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"

    return "fallback"


def _fallback_sentence(words: Sequence[str]) -> str:
    """Create a readable English sentence when no LLM API is available."""
    if not words:
        return ""

    cleaned_words = [str(w).strip() for w in words if str(w).strip()]
    if not cleaned_words:
        return ""

    # Simple dictionary substitutions for smoother fallback speech
    normalized = []
    for i, w in enumerate(cleaned_words):
        w_lower = w.lower()
        if w_lower == "i":
            normalized.append("I")
        elif i == 0:
            normalized.append(w_lower.capitalize())
        else:
            normalized.append(w_lower)

    sentence = " ".join(normalized)
    if not sentence.endswith((".", "!", "?")):
        sentence += "."

    return sentence


def _generate_with_gemini(words: Sequence[str], api_key: str) -> str:
    """Generate natural English sentence using Google Gemini REST API."""
    prompt = (
        "You are an Indian Sign Language (ISL) to English translator. "
        "Convert the following sequence of recognized ISL sign glosses into a single, "
        "grammatically correct, natural English sentence. "
        "Return ONLY the plain English sentence without explanations, markdown, or quotation marks.\n\n"
        f"ISL Signs: {' '.join(words)}"
    )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 60,
        },
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=10) as resp:
        res_data = json.loads(resp.read().decode("utf-8"))
        candidate = res_data.get("candidates", [{}])[0]
        content = candidate.get("content", {}).get("parts", [{}])[0].get("text", "")
        return content.strip().strip('"\'')


def _generate_with_openai(words: Sequence[str], api_key: str) -> str:
    """Generate natural English sentence using OpenAI Chat Completions REST API."""
    prompt = (
        "Convert the following recognized Indian Sign Language glosses into a single natural English sentence. "
        "Return ONLY the English sentence.\n\n"
        f"Signs: {' '.join(words)}"
    )

    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are an ISL to English sign language translator."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 60,
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=10) as resp:
        res_data = json.loads(resp.read().decode("utf-8"))
        msg = res_data["choices"][0]["message"]["content"]
        return msg.strip().strip('"\'')


def generate_sentence(words: Sequence[str]) -> str:
    """Reconstruct an ordered list of ISL signs into a natural English sentence.

    Falls back cleanly to local rule-based sentence construction if the API
    is unreachable or no key is provided.
    """
    if not words:
        return ""

    provider = get_active_provider()

    if provider == "google":
        key = os.getenv("GEMINI_API_KEY", "")
        if key:
            try:
                return _generate_with_gemini(words, key)
            except Exception as exc:
                print(f"[LanguageModel] Gemini API error ({exc}). Using fallback.")

    elif provider == "openai":
        key = os.getenv("OPENAI_API_KEY", "")
        if key:
            try:
                return _generate_with_openai(words, key)
            except Exception as exc:
                print(f"[LanguageModel] OpenAI API error ({exc}). Using fallback.")

    return _fallback_sentence(words)
