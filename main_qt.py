# main_qt.py
import json
import threading
import time
import sys

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel,
    QPushButton, QTextEdit, QHBoxLayout, QMessageBox
)
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt

from faster_whisper import WhisperModel
from utils.audio_io import record_until_silence
from utils.score import score_turn

# Load lesson
with open("lessons/greetings.json", "r", encoding="utf-8") as f:
    LESSON = json.load(f)

# Load Whisper model
ASR = WhisperModel("small", compute_type="int8")

running_flag = False

class TutorWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Darija Tutor")
        self.setMinimumSize(760, 440)

        layout = QVBoxLayout(self)

        title = QLabel("Darija Tutor")
        title.setFont(QFont("Segoe UI", 16))
        layout.addWidget(title)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.out = QTextEdit()
        self.out.setReadOnly(True)
        self.out.setFont(QFont("Consolas", 11))
        layout.addWidget(self.out, 1)

        row = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.exit_btn = QPushButton("Exit")
        row.addWidget(self.start_btn)
        row.addWidget(self.stop_btn)
        row.addStretch(1)
        row.addWidget(self.exit_btn)
        layout.addLayout(row)

        self.start_btn.clicked.connect(self.start_tutor)
        self.stop_btn.clicked.connect(self.stop_tutor)
        self.exit_btn.clicked.connect(self.close)



    def append_lines(self, lines: list[str]):
        self.out.append("\n".join(lines))

    def start_tutor(self):
        global running_flag
        if running_flag:
            return
        self.out.clear()
        running_flag = True
        t = threading.Thread(target=self.tutor_loop, daemon=True)
        t.start()

    def stop_tutor(self):
        global running_flag
        running_flag = False

    def tutor_loop(self):
        global running_flag
        turns = LESSON.get("turns", [])
        for idx, turn in enumerate(turns):
            if not running_flag:
                break

            self.status.setText(
                f"Prompt {idx+1}/{len(turns)}: {turn.get('prompt_text','')}"
            )

            t0 = time.time()
            audio = record_until_silence()

            segments, info = ASR.transcribe(audio, language="ar")
            text = " ".join(s.text for s in segments).strip()

            result = score_turn(text, turn)
            latency_ms = int((time.time() - t0) * 1000)

            lines = [
                f"Transcript: {text}",
                f"Intent: {result['intent']}  OK: {result['intent_ok']}",
                f"Edit distance: {result['edit_distance']}  Best match: {result['best_match']}",
                f"Latency: {latency_ms} ms",
                f"Feedback: {result['feedback']}",
                ""
            ]
            self.append_lines(lines)

            # ...existing code...

        self.status.setText("Done or stopped.")

def main():
    app = QApplication(sys.argv)
    w = TutorWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()