# Fine-tuning the Darija Tutor (LLM, not Whisper)

This folder fine-tunes the *LLM responses* so the tutor stays consistently Moroccan Darija instead of drifting into MSA or other dialects.

We do SFT (supervised fine-tuning) using LoRA (QLoRA optional) on a Darija-aligned instruction dataset:
- Dataset: yousef-khoubrane/tulu-v2-sft-mixture-english-darija
- Base model: Qwen/Qwen2.5-7B-Instruct

Why this dataset:
It contains English instruction prompts aligned with Moroccan Darija translations, which is exactly what we need for a Darija-first tutor.

## 0) Install dependencies

From repo root:

pip install -r ft/requirements.txt

You need a CUDA GPU for training. These scripts are written for "publishable code" and will refuse to train on CPU.

## 1) Prepare dataset (downloads from Hugging Face)

python -m ft.prepare_dataset

This creates:
ft/data/train.jsonl

Each row is a "text" field containing a chat-formatted training sample:
- user prompt in English
- assistant response in Darija (Arabic script)
Optionally, it also adds an Arabizi version of the Darija response.

## 2) Train LoRA (SFT)

python -m ft.train_sft_lora

Outputs (by default):
ft/out/lora_adapter/

## 3) Merge LoRA into the base model (optional)

python -m ft.merge_lora

Outputs:
ft/out/merged_model/

If you do not merge, you can still serve using base + adapter.

## 4) Serve local tutor API (so your GUI uses it)

python -m ft.serve_tutor_api

This starts:
http://127.0.0.1:8000/reply

Then set your GUI to use it:

export TUTOR_API_URL="http://127.0.0.1:8000"
# optional:
export TUTOR_API_KEY=""

Your existing llm/tutor_client.py already supports this.
When TUTOR_API_URL is set, it will call /reply instead of OpenAI.

## Smoke test

python -m ft.infer_smoke