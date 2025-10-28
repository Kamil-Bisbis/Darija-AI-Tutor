import json, os

def conjugation_table(verb, script="arabizi"):
    verbs_path = os.path.join(os.path.dirname(__file__), "..", "data", "verbs.json")
    try:
        with open(verbs_path, "r", encoding="utf-8") as f:
            verbs = json.load(f)
    except Exception:
        return "[verbs unavailable]"
    v = verb.strip().split()[0].lower()
    entry = verbs.get(v)
    if not entry:
        return f"[no data for '{v}']"
    forms = entry.get("paradigm", {})
    lines = [f"{v} ({entry.get('arabic','?')}):"]
    for pron, form in forms.items():
        lines.append(f"  {pron}: {form}")
    return "\n".join(lines)
