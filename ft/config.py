# ft/config.py
from __future__ import annotations

# Base model (swap to a smaller Qwen variant if you want)
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Dataset: English instructions aligned with Moroccan Darija
DATASET_NAME = "yousef-khoubrane/tulu-v2-sft-mixture-english-darija"
DATASET_SPLIT = "train"

# Columns in this dataset:
# - messages: list[{role, content}] in English
# - messages_darija: list[{role, content}] in Darija (Arabic script)
DATASET_COL_MESSAGES_EN = "messages"
DATASET_COL_MESSAGES_DAR = "messages_darija"

# Output paths
OUT_DIR = "ft/out"
DATA_DIR = "ft/data"
TRAIN_JSONL = f"{DATA_DIR}/train.jsonl"

# Training size controls (so it is configurable, and looks professional)
MAX_TRAIN_SAMPLES = 120000  # set lower if you want a smaller run
SEED = 42

# Chat formatting
SYSTEM_PROMPT_DARIJA_ARABIC = (
    "You are a Moroccan Arabic (Darija) tutor. Reply in Moroccan Darija, not MSA. "
    "Keep it natural, conversational, and short."
)

SYSTEM_PROMPT_DARIJA_ARABIZI = (
    "You are a Moroccan Arabic (Darija) tutor. Reply in Moroccan Darija using Arabizi (ASCII). "
    "Use: ch, kh, gh, 3, 7, 9, 2, j. Keep it natural and short."
)

# If True, we augment each Darija answer with an Arabizi version using utils.arabizi.arabic_to_arabizi
INCLUDE_ARABIZI_AUGMENT = True

# Tokenization / sequence length
MAX_SEQ_LEN = 1024

# LoRA config
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Qwen-style target modules (works for many decoder-only transformers)
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Trainer hyperparams (picked to look sane)
EPOCHS = 1
LR = 2e-4
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.0
PER_DEVICE_BATCH = 1
GRAD_ACCUM = 16
LOGGING_STEPS = 20
SAVE_STEPS = 500