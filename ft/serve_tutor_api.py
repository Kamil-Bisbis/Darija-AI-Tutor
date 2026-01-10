# ft/serve_tutor_api.py
from __future__ import annotations

import os
from typing import Optional

import torch
from fastapi import FastAPI
from pydantic import BaseModel

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from ft.config import (
    BASE_MODEL,
    OUT_DIR,
    SYSTEM_PROMPT_DARIJA_ARABIC,
    SYSTEM_PROMPT_DARIJA_ARABIZI,
)

# Reuse your transliterator to enforce Arabizi on output if desired
try:
    from utils.arabizi import arabic_to_arabizi, has_arabic_chars
except Exception:
    arabic_to_arabizi = None
    has_arabic_chars = lambda s: False  # type: ignore


class ReplyReq(BaseModel):
    prompt: str
    lang: str = "ar"
    script: Optional[str] = None  # "arabic" or "arabizi"


app = FastAPI()

_tok = None
_model = None


def _load_model():
    global _tok, _model

    merged_dir = os.path.join(OUT_DIR, "merged_model")
    adapter_dir = os.path.join(OUT_DIR, "lora_adapter")

    if os.path.isdir(merged_dir):
        model_path = merged_dir
        _tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        _model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        return

    # Fallback: base + adapter (no merge)
    _tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    if os.path.isdir(adapter_dir):
        _model = PeftModel.from_pretrained(base, adapter_dir)
    else:
        _model = base


def _chat(system: str, user: str) -> str:
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        prompt = _tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = f"System: {system}\nUser: {user}\nAssistant:"

    inputs = _tok(prompt, return_tensors="pt").to(_model.device)

    with torch.no_grad():
        out_ids = _model.generate(
            **inputs,
            max_new_tokens=160,
            do_sample=False,
        )

    text = _tok.decode(out_ids[0], skip_special_tokens=True)

    # Try to extract just the assistant portion in a simple way
    if "Assistant:" in text:
        text = text.split("Assistant:", 1)[-1].strip()
    return text.strip()


@app.on_event("startup")
def _startup():
    _load_model()


@app.post("/reply")
def reply(req: ReplyReq):
    lang = (req.lang or "ar").lower().strip()
    script = (req.script or "arabizi").lower().strip()

    if lang == "ar":
        system = SYSTEM_PROMPT_DARIJA_ARABIC if script == "arabic" else SYSTEM_PROMPT_DARIJA_ARABIZI
    else:
        system = "You are a concise English-speaking tutor. Reply with ONE short sentence under 25 words."

    out = _chat(system, (req.prompt or "").strip())

    # Hard enforce Arabizi if requested
    if lang == "ar" and script == "arabizi" and arabic_to_arabizi is not None:
        if has_arabic_chars(out):
            out = arabic_to_arabizi(out)

    return {"text": out}