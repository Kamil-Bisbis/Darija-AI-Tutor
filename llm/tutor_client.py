# llm/tutor_client.py
import os
import time
import requests
from typing import List, Optional, Tuple

from utils.arabizi import arabic_to_arabizi, has_arabic_chars

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
OPENAI_MODEL     = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
TUTOR_API_URL    = os.environ.get("TUTOR_API_URL") 
TUTOR_API_KEY    = os.environ.get("TUTOR_API_KEY")
TIMEOUT_SEC      = float(os.environ.get("TUTOR_TIMEOUT", "12.0"))
MAX_RETRIES      = int(os.environ.get("LLM_MAX_RETRIES", "3"))
BACKOFF_BASE     = float(os.environ.get("LLM_BACKOFF_BASE", "0.8"))
MAX_TOKENS       = int(os.environ.get("LLM_MAX_TOKENS", "160"))

def _post_with_retries(url: str, *, headers: dict, json: dict) -> requests.Response:
    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = requests.post(url, headers=headers, json=json, timeout=TIMEOUT_SEC)
            if r.status_code in (429, 503):
                ra = r.headers.get("Retry-After")
                delay = float(ra) if ra else BACKOFF_BASE * (2 ** attempt)
                time.sleep(min(delay, 8.0))
                last_err = requests.HTTPError(f"{r.status_code} {r.reason}")
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            last_err = e
            time.sleep(BACKOFF_BASE * (2 ** attempt))
    raise last_err if last_err else RuntimeError("LLM request failed")

def _openai_chat(messages: List[dict]) -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    url = f"{OPENAI_API_BASE}/chat/completions"
    body = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": MAX_TOKENS,
    }
    r = _post_with_retries(url, headers={"Authorization": f"Bearer {key}"}, json=body)
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()

def _custom_rest_tutor(prompt: str, lang: str) -> str:
    # Fine-tune hook: point this at your Darija-specific responder if/when you host it.
    if not TUTOR_API_URL:
        raise RuntimeError("TUTOR_API_URL not set")
    url = f"{TUTOR_API_URL.rstrip('/')}/reply"
    payload = {"prompt": prompt, "lang": lang}
    headers = {}
    if TUTOR_API_KEY:
        headers["Authorization"] = f"Bearer {TUTOR_API_KEY}"
    r = _post_with_retries(url, headers=headers, json=payload)
    data = r.json()
    return (data.get("text") or "").strip()

def ask_llm(
    transcript: str,
    lang_hint: Optional[str],             # 'en' or 'ar'
    mode: str = "normal",                 # "normal", "translate_en_to_ar", "translate_ar_to_en"
    output_script: Optional[str] = None,  # "arabizi", "arabic" when lang_hint =="ar"
    topics: Optional[Tuple[str, ...]] = None,
) -> str:
    """
    mode:
      - "normal"             -> reply in lang_hint
      - "translate_en_to_ar" -> English sentence containing the Darija phrase (in chosen script) in quotes
      - "translate_ar_to_en" -> English meaning
    output_script when lang_hint == "ar": "arabizi" | "arabic" | None
    topics: last few learner interests/themes to bias examples (tuple for safety)
    """
    lang = (lang_hint or "en").lower()
    out_script = (output_script or "arabizi").lower() if lang == "ar" else None
    topics_line = ", ".join(list(topics or ())[-10:]) if topics else ""

    # If you deploy your own fine-tuned responder, it plugs in here:
    if TUTOR_API_URL:
        return _custom_rest_tutor(transcript, lang)

    # Build a compact Darija-first prompt
    if mode == "translate_en_to_ar":
        if out_script == "arabic":
            system = (
                "You are a Moroccan Arabic tutor. The user wrote English and wants the Darija translation. "
                "Answer with ONE short English sentence that contains the Darija phrase in Arabic script in quotes. "
                f"Prefer examples about: {topics_line}."
            )
        else:
            system = (
                "You are a Moroccan Arabic tutor. The user wrote English and wants the Darija translation. "
                "Answer with ONE short English sentence that contains the Darija phrase in Arabizi (ASCII) in quotes. "
                "Use: ch, kh, gh, 3, 7, 9, 2, j. "
                f"Prefer examples about: {topics_line}."
            )
    elif mode == "translate_ar_to_en":
        system = (
            "You are a Moroccan Arabic tutor. The user wrote Darija and wants the English translation. "
            "Answer with ONE short English sentence. "
            f"Prefer examples about: {topics_line}."
        )
    else:
        if lang == "ar":
            if out_script == "arabic":
                system = (
                    "You are a Moroccan Arabic (Darija) tutor. Reply in Arabic script only. "
                    "ONE short sentence. Keep total under 25 words. "
                    f"Prefer examples about: {topics_line}."
                )
            else:
                system = (
                    "You are a Moroccan Arabic (Darija) tutor. Reply in Arabizi (ASCII) only. "
                    "Use: ch, kh, gh, 3, 7, 9, 2, j. "
                    "ONE short sentence. Keep total under 25 words. "
                    f"Prefer examples about: {topics_line}."
                )
        else:
            system = (
                "You are a concise English-speaking tutor. Reply with ONE short sentence. "
                f"Prefer examples about: {topics_line}. "
                "Keep total under 25 words."
            )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": transcript.strip()},
    ]
    out = _openai_chat(messages)

    # Safety: if Arabizi requested but model emits Arabic letters, convert to Arabizi
    if lang == "ar" and out_script == "arabizi" and has_arabic_chars(out):
        out = arabic_to_arabizi(out)

    return out