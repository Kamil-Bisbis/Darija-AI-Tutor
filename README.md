# Darija AI Tutor

Real-time tutor for **Moroccan Darija** powered by OpenAI Whisper (ASR) and an LLM (OpenAI GPT) for interactive replies, all in a lightweight PySide6 UI.

**Blog:** *Building a Conversational AI for an Underrepresented Language 
— How I Designed an AI Tutor to Learn and Practice Moroccan Darija.*  

Why this exists, the design choices, and what I learned.
Read it here: <[https://medium.com/@bisbis.kamil](https://medium.com/@bisbis.kamil/building-a-conversational-ai-to-teach-an-underrepresented-language-8dd118d17562)>

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
- **LLM replies:** After each utterance, the app sends your transcript to an LLM (OpenAI GPT) and displays a short, conversational reply in Arabizi (Darija) or English.
- **Arabizi mode:** Darija transcripts and replies are shown in Arabizi (Latin script) for easier reading and texting.

> Model weights are downloaded on first use from Hugging Face and are **not included** in the repo.
> LLM replies require an OpenAI API key (see below).

---

## Quick Start

```bash
git clone https://github.com/<your-username>/darija-ai-tutor.git
cd darija-ai-tutor

python -m venv .venv
source .venv/bin/activate        # windows: .venv\Scripts\activate
pip install -r requirements.txt

# Add your OpenAI API key for LLM replies:
echo "OPENAI_API_KEY=sk-..." > .env

python whisper_gui.py
```

**Note:**
- You need an OpenAI API key for LLM replies. Get one at https://platform.openai.com/account/api-keys
- The `.env` file is ignored by git for safety.

---

## How it works

1. **ASR (Speech-to-Text):** Your speech is transcribed using Whisper (Darija or English models).
2. **LLM (Language Model):** The transcript is sent to an LLM (OpenAI GPT) which generates a short reply, shown in Arabizi or English.
3. **UI:** Everything runs in a simple PySide6 desktop app with live mic, transcript, and tutor reply.

---

For more details, see the blog post or explore the code.
