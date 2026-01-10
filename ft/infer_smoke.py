# ft/infer_smoke.py
from __future__ import annotations

import os
import requests

def main() -> None:
    url = os.environ.get("TUTOR_API_URL", "http://127.0.0.1:8000").rstrip("/") + "/reply"

    tests = [
        {"prompt": "How do I say 'I am tired'?", "lang": "ar", "script": "arabizi"},
        {"prompt": "Tell me a short Darija greeting for a friend.", "lang": "ar", "script": "arabizi"},
        {"prompt": "How do I ask for the price in a market?", "lang": "ar", "script": "arabizi"},
    ]

    for t in tests:
        r = requests.post(url, json=t, timeout=15)
        print(">", t["prompt"])
        print(r.json().get("text"))
        print()

if __name__ == "__main__":
    main()