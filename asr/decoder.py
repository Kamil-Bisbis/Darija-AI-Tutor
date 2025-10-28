# asr/decoder.py
import time
import numpy as np
import torch
import torch.nn.functional as F

def run_decode(window, fs, processor, model, inbuf, emit_text, emit_finalize, device,
               emit_progress=None, forced_lang: str = "auto"):
    """
    Collect audio while held, then run ONE Whisper decode on release.
    Supports forced_lang in {"en","ar","auto"} to bias multilingual checkpoints.
    """
    model.eval()
    model_dtype = next(model.parameters()).dtype

    target_sr = 16000
    accum = []
    sr_in = getattr(window, "mic_sr", fs)

    # while holding, drain the queue and store audio
    while getattr(window, "recording", False):
        try:
            block = inbuf.get(timeout=0.05)
        except Exception:
            continue

        x = block.astype(np.float32).squeeze()
        if x.ndim != 1:
            x = x[:, 0]

        if sr_in != target_sr:
            t = torch.from_numpy(x)[None, None, :].to(torch.float32)
            t = F.interpolate(t, size=int(len(x) * target_sr / max(1, sr_in)),
                              mode="linear", align_corners=False)
            x16 = t[0, 0].cpu().numpy()
        else:
            x16 = x
        accum.append(x16.copy())

    if emit_progress:
        try: emit_progress(10)
        except Exception: pass

    if not accum:
        emit_finalize(time.time())
        return

    audio16 = np.concatenate(accum)

    # features
    feats = processor(
        audio16, sampling_rate=target_sr, return_tensors="pt"
    ).input_features.to(device=device, dtype=model_dtype)

    if emit_progress:
        try: emit_progress(35)
        except Exception: pass

    # generation args
    gen_kwargs = {
        "max_new_tokens": 128,
        "do_sample": False,
        "return_dict_in_generate": False
    }

    # If multilingual whisper (small/medium), we can supply language+task prompt IDs
    forced_lang = (forced_lang or "auto").lower()
    try:
        is_multilingual = getattr(model, "is_multilingual", True)  # small/medium are multilingual
    except Exception:
        is_multilingual = True

    task = "transcribe"
    forced_ids = None
    if is_multilingual and forced_lang in ("en", "ar"):
        try:
            forced_ids = processor.get_decoder_prompt_ids(language=forced_lang, task=task)
        except Exception:
            forced_ids = None

    # make sure English checkpoints are not forcing English
    try:
        if getattr(model.generation_config, "forced_decoder_ids", None) is not None:
            model.generation_config.forced_decoder_ids = None
    except Exception:
        pass

    with torch.no_grad():
        if forced_ids is not None:
            ids = model.generate(input_features=feats, forced_decoder_ids=forced_ids, **gen_kwargs)
        else:
            ids = model.generate(input_features=feats, **gen_kwargs)

    if emit_progress:
        try: emit_progress(92)
        except Exception: pass

    text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
    emit_text(text)

    if emit_progress:
        try: emit_progress(100)
        except Exception: pass
    emit_finalize(time.time())
