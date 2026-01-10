# ft/prepare_dataset.py
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from transformers import AutoTokenizer

from ft.config import (
    BASE_MODEL,
    DATASET_NAME,
    DATASET_SPLIT,
    DATASET_COL_MESSAGES_EN,
    DATASET_COL_MESSAGES_DAR,
    DATA_DIR,
    TRAIN_JSONL,
    MAX_TRAIN_SAMPLES,
    SEED,
    MAX_SEQ_LEN,
    SYSTEM_PROMPT_DARIJA_ARABIC,
    SYSTEM_PROMPT_DARIJA_ARABIZI,
    INCLUDE_ARABIZI_AUGMENT,
)

# Reuse your existing transliterator
try:
    from utils.arabizi import arabic_to_arabizi, has_arabic_chars
except Exception:
    arabic_to_arabizi = None
    has_arabic_chars = lambda s: False  # type: ignore


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _pick_turn(messages: List[Dict[str, Any]], role: str) -> Optional[str]:
    # Find the first turn of a role
    for m in messages or []:
        if (m.get("role") or "").lower() == role:
            c = (m.get("content") or "").strip()
            if c:
                return c
    return None


def _format_chat(tokenizer, system: str, user: str, assistant: str) -> str:
    # Use model-native chat template when available
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]
    try:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    except Exception:
        # Fallback: simple plain format
        return f"System: {system}\nUser: {user}\nAssistant: {assistant}\n"


def main() -> None:
    random.seed(SEED)
    _ensure_dir(DATA_DIR)

    print(f"[prepare_dataset] base_model={BASE_MODEL}")
    print(f"[prepare_dataset] dataset={DATASET_NAME}:{DATASET_SPLIT}")

    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)

    # Subsample for sanity / speed (still huge enough to be credible)
    n = min(len(ds), MAX_TRAIN_SAMPLES)
    idxs = list(range(len(ds)))
    random.shuffle(idxs)
    idxs = idxs[:n]
    ds = ds.select(idxs)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

    out_path = TRAIN_JSONL
    written = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in ds:
            en_msgs = ex.get(DATASET_COL_MESSAGES_EN) or []
            dar_msgs = ex.get(DATASET_COL_MESSAGES_DAR) or []

            user_en = _pick_turn(en_msgs, "user")
            asst_dar = _pick_turn(dar_msgs, "assistant")

            if not user_en or not asst_dar:
                continue

            # Primary training example: English user -> Darija assistant (Arabic script)
            text1 = _format_chat(tokenizer, SYSTEM_PROMPT_DARIJA_ARABIC, user_en, asst_dar)
            f.write(json.dumps({"text": text1}, ensure_ascii=False) + "\n")
            written += 1

            # Optional augmentation: also teach Arabizi output style
            if INCLUDE_ARABIZI_AUGMENT and arabic_to_arabizi is not None:
                if has_arabic_chars(asst_dar):
                    asst_az = arabic_to_arabizi(asst_dar)
                    text2 = _format_chat(tokenizer, SYSTEM_PROMPT_DARIJA_ARABIZI, user_en, asst_az)
                    f.write(json.dumps({"text": text2}, ensure_ascii=False) + "\n")
                    written += 1

    print(f"[prepare_dataset] wrote {written} samples to {out_path}")
    print("[prepare_dataset] done")


if __name__ == "__main__":
    main()