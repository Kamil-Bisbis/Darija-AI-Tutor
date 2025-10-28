# llm/topics.py
from typing import List, Sequence

import re

_STOP = {
    "the","and","for","with","from","that","this","your","you","are","was","were",
    "have","has","had","about","into","over","under","after","before","some","very",
    "just","like","what","when","which","will","would","could","should","them","they",
    "their","there","here","then","than","also","only","more","most","many","much"
}

# lightweight hints we care about
_HINTS = (
    "pronunciation","spelling","conjugation","past tense","present","future",
    "negation","polite","formal","casual","greeting","travel","food","family","work","study",
    "slang","phrase","idiom","expression"
)

def _rule_topics(msg: str) -> List[str]:
    s = (msg or "").lower()
    words = re.findall(r"[a-z\u0600-\u06FF]{4,}", s)
    out: List[str] = []
    for w in words:
        if w in _STOP:
            continue
        out.append(w)
    for h in _HINTS:
        if h in s and h not in out:
            out.append(h)
    # keep short, unique, ordered
    seen = set()
    dedup = []
    for t in out:
        if t not in seen:
            seen.add(t)
            dedup.append(t)
    return dedup[:8]

def extract_topics(message: str, current_topics: Sequence[str] | None) -> List[str]:
    """
    Cheap, deterministic topic extraction to bias examples. No caching here,
    and we always return a plain list (safe for your GUI state).
    """
    base = list(current_topics or [])
    if not message or len(message.strip()) < 10:
        return base

    # rule-based (fast, offline)
    new = _rule_topics(message)

    for t in new:
        if t and t not in base:
            base.append(t)

    # trim to last 30
    if len(base) > 30:
        base = base[-30:]
    return base