import json
import threading
import time
import PySimpleGUI as sg
from faster_whisper import WhisperModel

from utils.audio_io import record_until_silence
from utils.score import score_turn

## Load lesson
with open("lessons/greetings.json", "r", encoding="utf-8") as f:
    LESSON = json.load(f)

## Load Whisper model (first run downloads weights). "small" is a good compromise.
ASR = WhisperModel("small", compute_type="int8")

running_flag = False


def tutor_loop(window: sg.Window) -> None:
    global running_flag
    turns = LESSON.get("turns", [])
    for idx, turn in enumerate(turns):
        if not running_flag:
            break
        # show prompt text
        window["-STATUS-"].update(
            f"Prompt {idx+1}/{len(turns)}: {turn.get('prompt_text','')}"
        )
        window.refresh()

        # listen
        t0 = time.time()
        audio = record_until_silence()

        # transcribe
        segments, info = ASR.transcribe(audio, language="ar")
        text = " ".join(s.text for s in segments).strip()

        # score
        result = score_turn(text, turn)
        latency_ms = int((time.time() - t0) * 1000)

        # display
        out = [
            f"Transcript: {text}",
            f"Intent: {result['intent']}  OK: {result['intent_ok']}",
            f"Edit distance: {result['edit_distance']}  Best match: {result['best_match']}",
            f"Latency: {latency_ms} ms",
            f"Feedback: {result['feedback']}",
        ]
        window["-OUTPUT-"].update("\n".join(out))
        window.refresh()

    window["-STATUS-"].update("Done or stopped.")


def start_tutor(window: sg.Window) -> None:
    global running_flag
    if running_flag:
        return
    running_flag = True
    threading.Thread(target=tutor_loop, args=(window,), daemon=True).start()


def stop_tutor() -> None:
    global running_flag
    running_flag = False


def main() -> None:
    layout = [
        [sg.Text("Darija Tutor", font=("Arial", 16))],
        [sg.Text("", size=(80, 2), key="-STATUS-")],
        [
            sg.Multiline(
                size=(80, 12), key="-OUTPUT-", autoscroll=True, disabled=True
            )
        ],
        [sg.Button("Start"), sg.Button("Stop"), sg.Button("Exit")],
    ]
    window = sg.Window("Darija Tutor", layout)

    while True:
        event, _ = window.read()
        if event in (sg.WINDOW_CLOSED, "Exit"):
            stop_tutor()
            break
        if event == "Start":
            window["-OUTPUT-"].update("")
            start_tutor(window)
        if event == "Stop":
            stop_tutor()

    window.close()


if __name__ == "__main__":
    main()