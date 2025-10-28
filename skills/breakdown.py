import json, os

def breakdown(text, script="arabizi"):
    lexicon_path = os.path.join(os.path.dirname(__file__), "..", "data", "darija_lexicon.json")
    try:
        with open(lexicon_path, "r", encoding="utf-8") as f:
            lexicon = json.load(f)
    except Exception:
        return "[lexicon unavailable]"
    # very simple: just return glosses for each token
    tokens = text.strip().split()
    out = []
    for token in tokens:
        entry = next((e for e in lexicon if e.get(script) == token or e.get("arabic") == token or e.get("arabizi") == token), None)
        if entry:
            out.append(f"{token}: {entry.get('gloss','?')} ({entry.get('register','?')})")
        else:
            out.append(f"{token}: ?")
    return " | ".join(out)
