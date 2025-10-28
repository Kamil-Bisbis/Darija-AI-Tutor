# utils/arabizi.py
import os, json, re, functools

# --- transliteration (Arabic letters -> Arabizi) ---
_DIAC = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED]")
_TATWEEL = "\u0640"

_MAP = {
    "لا": "la",
    "ا": "a", "أ": "2a", "إ": "2i", "آ": "a", "ى": "a",
    "ء": "2", "ؤ": "2u", "ئ": "2i", "ٔ": "2",
    "ب": "b", "ت": "t", "ث": "t", "ج": "j", "ح": "7", "خ": "kh",
    "د": "d", "ذ": "d", "ر": "r", "ز": "z", "س": "s", "ش": "ch",
    "ص": "s", "ض": "d", "ط": "t", "ظ": "d",
    "ع": "3", "غ": "gh", "ف": "f", "ق": "9", "ك": "k", "گ": "g", "ڭ": "g",
    "ل": "l", "م": "m", "ن": "n", "ه": "h", "ة": "a",
    "و": "u", "ي": "i",
}

def arabic_to_arabizi(text: str) -> str:
    if not text: return ""
    s = _DIAC.sub("", text.replace(_TATWEEL, ""))
    # two-char first
    s = s.replace("لا", "la")
    out = []
    i = 0
    while i < len(s):
        ch = s[i]
        out.append(_MAP.get(ch, ch))
        i += 1
    res = "".join(out)
    res = re.sub(r"\s+", " ", res).strip()
    return res

def has_arabic_chars(s: str) -> bool:
    if not s: return False
    return any('\u0600' <= ch <= '\u06FF' for ch in s)

# --- language hints for fallback ---
def detect_lang(text: str) -> str:
    if not text: return "en"
    if has_arabic_chars(text): return "ar"
    s = text.lower()
    toks = re.findall(r"[a-z0-9]+", s)
    darija_vocab = {
        "salam","slm","labas","kulchi","kulshi","safi","wach","bghit","bghina","kifash","kifach","kif",
        "fin","shno","chno","smahli","3afak","bslama","mzyan","bikhir","smiti","smitk","hanout",
        "kayen","daba","bzzaf","shukran","chokrane","hadi","hadak","hadik"
    }
    score = 0
    if any(p in s for p in ("kh","gh","ch","sh")): score += 1
    if any(d in s for d in ("2","3","7","9")): score += 1
    if any(t in darija_vocab for t in toks): score += 1
    return "ar" if score >= 2 else "en"

_DARIJA_WORD_RE = re.compile(r'd[a-z]*r[a-z]*i?z?h?i[a-z]*a', re.IGNORECASE)
def mentions_darija_word(text: str) -> bool:
    if not text: return False
    return bool(_DARIJA_WORD_RE.search(text))

# --- mishear normalization (e.g., "cool feel the bass" -> "kulchi labas") ---
@functools.lru_cache(maxsize=1)
def _load_mishears():
    here = os.path.dirname(__file__)
    path = os.path.abspath(os.path.join(here, "..", "data", "mishears.json"))
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def normalize_mishears(text: str) -> str:
    table = _load_mishears()
    if not text or not table: return text
    s = text
    for canonical, variants in table.items():
        for v in variants:
            # case-insensitive whole/partial phrase replace
            pattern = re.compile(re.escape(v), re.IGNORECASE)
            s = pattern.sub(canonical, s)
    return s
