# asr/decoder.py
import time
import numpy as np
import torch
import torch.nn.functional as F

# Tunables
WINDOW_SECS = 8.0            # context window for decoding
EMIT_EVERY_SECS = 0.5        # UI update cadence (~2 Hz)
FINALIZE_SILENCE_SECS = 0.8  # silence needed to "lock" a sentence
STABLE_MIN_SECS = 0.5        # text must remain unchanged for this long (after silence)
MIN_EMIT_CHARS_NOTALK = 3    # don't emit tiny deltas while silent
PUNCT_FINAL = (".", "?", "!", "…", "،", "؛")

def _is_multilingual(model):
    return bool(getattr(model.config, "is_multilingual", False))

def run_decode(window, fs, processor, model, inbuf, emit_text, finalize_segment, device):
    """
    Pull audio from `inbuf`, run Whisper, and emit incremental text.
    Also: lock lines on stable silence, suppress single-word junk during pauses,
    and provide attention masks to avoid warnings.
    """
    model.eval()
    # branch generation settings by checkpoint type
    is_multilingual = bool(getattr(model.config, "is_multilingual", False))
    gen_kwargs = dict(
        do_sample=False,
        temperature=0.0,
        num_beams=1,
        return_timestamps=False,
        max_new_tokens=96,
    )

    if is_multilingual:
        # darija: seed via language/task flags
        gen_kwargs.update(
            language=getattr(window, "lang_hint", None) or "ar",
            task=getattr(window, "task_mode", None) or "transcribe",
        )
    else:
        # english: clear baked prompt to avoid the deprecation note
        if getattr(model.generation_config, "forced_decoder_ids", None) is not None:
            model.generation_config.forced_decoder_ids = None

    # encoder features and mask from the rolling buffer
    feats = processor.feature_extractor(buf, sampling_rate=16000, return_tensors="pt").input_features.to(device)
    enc_mask = torch.ones((feats.shape[0], feats.shape[-1]), dtype=torch.long, device=feats.device)

    # decode
    ids = model.generate(input_features=feats, attention_mask=enc_mask, **gen_kwargs)
    text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

    def _resample_mono(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
        if sr_in == sr_out:
            return x
        t = torch.from_numpy(x).to(torch.float32)[None, None, :]
        y = F.interpolate(t, size=int(len(x) * sr_out / sr_in), mode="linear", align_corners=False)
        return y[0, 0].cpu().numpy()

    mic_sr = int(getattr(window, "mic_sr", fs))

    while getattr(window, "recording", False):
        # Collect a block (non-blocking)
        try:
            in_block = inbuf.get(timeout=0.05)
        except Exception:
            in_block = None
        if in_block is None:
            continue

        x = in_block.astype(np.float32).squeeze()
        if x.ndim != 1:
            x = x[:, 0]

        # Simple RMS VAD on this block
        block_rms = float(np.sqrt(np.mean(x * x) + 1e-12))
        now = time.time()
        if block_rms > vad_gate:
            last_speech_t = now

        # Resample to 16 kHz
        x16 = _resample_mono(x, mic_sr, target_sr)

        # Slide into ring buffer
        n = min(len(x16), len(buf))
        if n > 0:
            buf = np.roll(buf, -n)
            buf[-n:] = x16[:n]

        # Throttle decoding rate
        if now - last_emit < EMIT_EVERY_SECS:
            continue
        last_emit = now

        with torch.no_grad():
            # extract log-mel spectrogram features from the 8s rolling buffer
            feats = processor.feature_extractor(buf, sampling_rate=16000, return_tensors="pt").input_features.to(device)

            # create an encoder attention mask (all ones since we never pad streaming audio)
            enc_mask = torch.ones((feats.shape[0], feats.shape[-1]), dtype=torch.long, device=feats.device)

            # run whisper’s decoder with deterministic settings for stable output
            ids = model.generate(
                input_features=feats,
                attention_mask=enc_mask,
                do_sample=False,
                temperature=0.0,
                num_beams=1
            )

            # decode token ids back into text
            text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

        # Track stability
        if text != prev_text:
            prev_text = text
            last_change_t = now

        # Decide whether to emit this update
        speaking = (now - last_speech_t) < 0.25  # small hangover; treat as "speaking" shortly after voice
        delta_len = max(0, len(text) - len(last_emitted))

        should_emit = False
        if speaking:
            # While speaking we keep UI updated
            should_emit = True
        else:
            # While silent, suppress tiny jitter (random "I/you/so")
            if delta_len >= MIN_EMIT_CHARS_NOTALK or (text and last_emitted == ""):
                should_emit = True

        if should_emit and text:
            last_emitted = text
            emit_text(text)

        # Finalize (“lock in”) when:
        #  - we’ve been silent long enough, and
        #  - text hasn’t changed recently, or ends with terminal punctuation
        silence_for = now - last_speech_t
        unchanged_for = now - last_change_t
        ends_with_punct = any(text.endswith(p) for p in PUNCT_FINAL)

        if text and (
            (silence_for >= FINALIZE_SILENCE_SECS and unchanged_for >= STABLE_MIN_SECS)
            or (ends_with_punct and silence_for >= 0.3)
        ):
            # Push the final text one more time (in case a small ending changed),
            # then lock the segment and reset state for the next one.
            if text != last_emitted:
                emit_text(text)
            finalize_segment(now)
            prev_text = ""
            last_emitted = ""
            last_change_t = now  # reset stability window

    # Recording stopped; finalize any residual
    finalize_segment(time.time())