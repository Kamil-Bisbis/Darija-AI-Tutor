# Darija AI Tutor

Real-time tutor for **Moroccan Darija** powered by OpenAI Whisper and a lightweight PySide6 UI.

📝 **Blog:** *Building a Conversational AI for an Underrepresented Language 
— How I Designed an AI Tutor to Learn and Practice Moroccan Darija.*  

Why this exists, the design choices, and what I learned.
👉 Read it here: <https://medium.com/@bisbis.kamil>

---

## Why Darija?

Most Arabic learning tools focus on **Modern Standard Arabic (MSA)**, which isn’t what Moroccans speak day to day. **Darija** (Moroccan Arabic) is largely spoken, rarely written, and historically under-resourced. Thanks to the open Whisper ecosystem, community **Darija fine-tunes** now exist, which makes a practical, conversational tutor possible.

---

## Features

- **Live mic capture** with waveform + VU meter
- **Switchable ASR models:**
  - `ychafiqui/whisper-small-darija`
  - `ychafiqui/whisper-medium-darija`
  - `openai/whisper-small.en`
  - `openai/whisper-medium.en`
- **Permanent transcript**: once a sentence “locks in,” it’s frozen with timestamps
- **Pause/noise filtering** to reduce stray one-word hallucinations between phrases
- **Apple Silicon** (MPS) support with CPU fallback

> Model weights are downloaded on first use from Hugging Face and are **not included** in the repo.

---

## Quick Start

```bash
git clone https://github.com/<your-username>/darija-ai-tutor.git
cd darija-ai-tutor

python -m venv .venv
source .venv/bin/activate        # windows: .venv\Scripts\activate
pip install -r requirements.txt

python whisper_gui.py