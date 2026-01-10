# ft/merge_lora.py
from __future__ import annotations

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from ft.config import BASE_MODEL, OUT_DIR


def main() -> None:
    adapter_dir = os.path.join(OUT_DIR, "lora_adapter")
    merged_dir = os.path.join(OUT_DIR, "merged_model")
    os.makedirs(merged_dir, exist_ok=True)

    print(f"[merge] base={BASE_MODEL}")
    print(f"[merge] adapter={adapter_dir}")
    print(f"[merge] out={merged_dir}")

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="cpu",
    )

    m = PeftModel.from_pretrained(base, adapter_dir)
    m = m.merge_and_unload()

    m.save_pretrained(merged_dir)
    tok.save_pretrained(merged_dir)

    print("[merge] done")


if __name__ == "__main__":
    main()