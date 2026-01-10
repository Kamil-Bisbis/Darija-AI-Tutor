# Fine Tuning the Darija Tutor

This folder contains the training pipeline that teaches the tutor to speak real Moroccan Darija instead of drifting into Modern Standard Arabic or other dialects.

Large language models do not naturally learn Darija well because it barely appears on the public web. When they are unsure, they fall back to MSA or mix in Levantine or Egyptian Arabic. This fine tuning step corrects that by retraining the model on a Darija focused dataset so that Darija becomes its default behavior.

The training method used here is supervised fine tuning with LoRA.

The base model is Qwen/Qwen2.5-7B-Instruct.

The dataset is yousef-khoubrane/tulu-v2-sft-mixture-english-darija from Hugging Face. It contains English instructions paired with Moroccan Darija responses written in Arabic script. That makes it ideal for a tutor that takes English input and replies in Darija.

---

## Install dependencies

From the repository root:

```
pip3 install -r ft/requirements.txt
```

These scripts are written for GPU training. They will not run on CPU.

---

## Prepare the dataset

This step downloads the dataset from Hugging Face and converts it into chat formatted training samples.

```
python -m ft.prepare_dataset
```

It creates:

```
ft/data/train.jsonl
```

Each line is a full chat example containing a user message in English and a Darija reply from the assistant.

---

## Train the LoRA adapter

This runs supervised fine tuning and learns Darija behavior while keeping the base model frozen.

```
python -m ft.train_sft_lora
```

The output is written to:

```
ft/out/lora_adapter/
```

This directory contains only the small LoRA weights, not a full model.

---

## Merge the adapter into the base model (optional)

If you want a single standalone model instead of base plus adapter, run:

```
python -m ft.merge_lora
```

This produces:

```
ft/out/merged_model/
```

If you skip this step, the API can still load the base model and the adapter together.

---

## Run the local tutor API

This starts a small HTTP server that your GUI can talk to instead of OpenAI.

```
python -m ft.serve_tutor_api
```

The API runs at:

```
http://127.0.0.1:8000/reply
```

To connect the GUI to it, set:

```
export TUTOR_API_URL="http://127.0.0.1:8000"
export TUTOR_API_KEY=""
```

Your existing llm/tutor_client.py already checks for this and will route requests to the local model when it is set.

---

## Smoke test

This sends a test prompt through the fine tuned model and prints the output.

```
python -m ft.infer_smoke
```

If this works, the Darija model is correctly installed and serving.