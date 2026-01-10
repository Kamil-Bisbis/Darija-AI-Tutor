# ft/train_sft_lora.py
from __future__ import annotations

import os
import torch

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

from ft.config import (
    BASE_MODEL,
    OUT_DIR,
    TRAIN_JSONL,
    MAX_SEQ_LEN,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_TARGET_MODULES,
    EPOCHS,
    LR,
    WARMUP_RATIO,
    WEIGHT_DECAY,
    PER_DEVICE_BATCH,
    GRAD_ACCUM,
    LOGGING_STEPS,
    SAVE_STEPS,
    SEED,
)


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA GPU required for training. "
            "If you only want publishable code, keep this as-is and document it in ft/README.md."
        )


def main() -> None:
    _require_cuda()

    os.makedirs(OUT_DIR, exist_ok=True)
    adapter_out = os.path.join(OUT_DIR, "lora_adapter")

    print(f"[train] base_model={BASE_MODEL}")
    print(f"[train] train_file={TRAIN_JSONL}")
    print(f"[train] out={adapter_out}")

    ds = load_dataset("json", data_files=TRAIN_JSONL, split="train")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # QLoRA style loading (4-bit)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        load_in_4bit=True,
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )

    model = get_peft_model(model, lora_cfg)

    args = TrainingArguments(
        output_dir=adapter_out,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        bf16=False,
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="none",
        seed=SEED,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        args=args,
    )

    trainer.train()
    trainer.model.save_pretrained(adapter_out)
    tokenizer.save_pretrained(adapter_out)

    print("[train] done")


if __name__ == "__main__":
    main()