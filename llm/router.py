# llm/router.py
from __future__ import annotations
import json, re, pathlib
from typing import Optional, Sequence, Tuple, Dict, Any

import utils.arabizi as ar_utils
from llm.tutor_client import ask_llm

# Safe access to helpers you already expose
_has_ar = getattr(ar_utils, "has_arabic_chars", lambda s: False)
_norm_mishears = getattr(ar_utils, "normalize_mishears", lambda s: s)

# Config path fits your current tree: data/router_lex.json
_THIS_DIR = pathlib.Path(__file__).resolve().parent
_LEX_PATH = (_THIS_DIR / ".." / "data" / "router_lex.json").resolve()

_DEFAULT_LEX: Dict[str, Any] = {
    "translate_en_to_ar": {
        "phrases": [r"\bhow (do|to) i say\b", r"\btranslate\b", r"\bdarija for\b"],
        "keywords": ["darija", "moroccan arabic", "arabic"]
    },
    "translate_ar_to_en": {
        "phrases": [r"\bin english\b", r"\btranslate to english\b"],
        "keywords": ["inglizi", "inglizya", "tarjama", "english"]
    },
    "force_darija": {
        "phrases": [r"\brespond in darija\b", r"\breply in darija\b", r"\bspeak darija\b"]
    },
    "force_english": {
        "phrases": [r"\brespond in english\b", r"\breply in english\b", r"\bspeak english\b"]
    },
    "arabizi_cues": ["salam","labas","kif","3lach","bghit","mazyan","wach","tarjama","inglizi","s7ab","3afak","chno"]
}

def _load_lex() -> Dict[str, Any]:
    try:
        with open(_LEX_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        out = _DEFAULT_LEX.copy()
        out.update(data)
        return out
    except Exception:
        return _DEFAULT_LEX

LEX = _load_lex()
ARABIZI_CUES = set(LEX.get("arabizi_cues", []))
_ARABIZI_DIGITS = ("2", "3", "5", "6", "7", "9")

def _guess_script(s: str) -> str:
    if _has_ar(s): return "arabic"
    if any(d in s for d in _ARABIZI_DIGITS) and re.search(r"[a-z]", s, re.I):
        return "arabizi"
    return "latin"

def _score(intent: str, text_low: str) -> int:
    spec = LEX.get(intent, {})
    conf = 0
    for p in spec.get("phrases", []):
        try:
            if re.search(p, text_low): conf += 3
        except re.error:
            pass
    for kw in spec.get("keywords", []):
        if kw in text_low: conf += 1
    return conf

def _decide_mode(text_low: str, lang_in: str) -> Tuple[str, str]:
    force_dar = _score("force_darija", text_low)
    force_en  = _score("force_english", text_low)
    en2ar     = _score("translate_en_to_ar", text_low)
    ar2en     = _score("translate_ar_to_en", text_low)

    if force_dar >= 3: return "normal", "ar"
    if force_en  >= 3: return "normal", "en"

    if lang_in == "ar": ar2en += 1
    else:               en2ar += 1

    if any(w in text_low for w in ARABIZI_CUES): en2ar += 1

    if max(en2ar, ar2en) >= 3:
        if en2ar > ar2en: return "translate_en_to_ar", "ar"
        if ar2en > en2ar: return "translate_ar_to_en", "en"
        return (("translate_en_to_ar", "ar") if lang_in == "en"
                else ("translate_ar_to_en", "en"))

    return "normal", lang_in

def route(
    text: str,
    lang_in: str,
    want_script: str,                     # "arabizi" | "arabic"
    topics: Optional[Sequence[str]] = (), # list or tuple
) -> str:
    s_raw = text or ""
    s = _norm_mishears(s_raw).strip()
    s_low = s.lower()

    lang_in = (lang_in or "en").lower()
    want_script = (want_script or "arabizi").lower()
    topics_tuple: Tuple[str, ...] = tuple(topics or ())

    mode, out_lang = _decide_mode(s_low, lang_in)

    if out_lang == "ar":
        if want_script in ("arabic", "arabizi"):
            output_script = want_script
        else:
            output_script = "arabic" if _guess_script(s) == "arabic" else "arabizi"
    else:
        output_script = None

    return ask_llm(
        s,
        out_lang,
        mode=mode,
        output_script=output_script,
        topics=topics_tuple,
    )