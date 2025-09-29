# stdlib
import sys, queue, threading, time
from datetime import datetime

# third party
import numpy as np
import sounddevice as sd
import torch
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QTextEdit,
    QPushButton, QComboBox, QLabel, QHBoxLayout, QFrame
)
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

# local
from ui.levels import LevelBar
from ui.waves import WaveScope
from ui.lamps import TalkLamp
from asr.decoder import run_decode


class EchoGateWindow(QMainWindow):
    text_ready = Signal(str)
    break_line = Signal()
    vu_level = Signal(float)
    speech_state = Signal(bool)

    def __init__(self):
        super().__init__()
        # UI state
        self.live_text = ""
        self.seg_t0 = None
        self.live_anchor_pos = None
        self._autoscroll = True

        # audio/model state
        self.recording = False
        self.inbuf = queue.Queue()
        self.fs = 16000
        self.channels = 1
        self.frames_per_block = int(self.fs * 0.3)
        self.model = None
        self.processor = None
        self.worker = None
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.speak_gate = 0.009
        self.task_mode = "transcribe"
        self.lang_hint = None

        self._init_ui()
        self._connect_signals()
        self.load_backend()  # load default model

        # autoscroll hookup (now that text_display exists)
        sb = self.text_display.verticalScrollBar()
        def _on_scroll(_):
            self._autoscroll = (sb.value() >= sb.maximum() - 2)
        sb.valueChanged.connect(_on_scroll)

    def _init_ui(self):
        main_layout = QVBoxLayout()

        # controls row
        controls_frame = QFrame()
        controls_frame.setFrameStyle(QFrame.Panel | QFrame.Raised)
        controls_layout = QHBoxLayout()

        model_label = QLabel("Model:")
        self.model_combo = QComboBox()
        self.model_combo.clear()
        self.model_combo.addItems(["en-small", "en-medium", "darija-small", "darija-medium"])
        self.model_combo.setCurrentText("darija-small")
        self.model_combo.setEnabled(True)
        self.model_combo.currentTextChanged.connect(self.load_backend)

        self.btn_record = QPushButton("Listen")
        self.btn_record.clicked.connect(self.toggle_io)

        controls_layout.addWidget(model_label)
        controls_layout.addWidget(self.model_combo)
        controls_layout.addWidget(self.btn_record)
        controls_frame.setLayout(controls_layout)

        # meters / indicators
        self.waveform = WaveScope()
        top_row = QHBoxLayout()
        self.pill = TalkLamp()
        self.vu = LevelBar(gate=self.speak_gate)
        top_row.addWidget(self.pill, 0)
        top_row.addWidget(self.vu, 1)

        # transcript
        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        self.text_display.setStyleSheet("QTextEdit { background: #232629; color: #ECEFF4; }")
        self.text_display.setFont(QFont("Menlo", 11))

        # assemble
        main_layout.addWidget(controls_frame)
        main_layout.addLayout(top_row)
        main_layout.addWidget(self.waveform)
        main_layout.addWidget(self.text_display)

        central = QWidget()
        central.setLayout(main_layout)
        self.setCentralWidget(central)

        # window chrome
        self.statusBar().showMessage("idle")
        self.setWindowTitle("Darija AI Tutor")
        self.setGeometry(100, 100, 900, 600)

    def _connect_signals(self):
        self.text_ready.connect(self.paint_text)
        self.break_line.connect(self.insert_blank)
        self.vu_level.connect(lambda v: self.vu.set_level(v))
        self.speech_state.connect(lambda on: self.pill.set_state(on))

    def load_backend(self):
        was_recording = self.recording
        if was_recording:
            self.end_io()
        choice = self.model_combo.currentText()
        is_darija = choice.startswith("darija-")
        self.lang_hint = "ar" if is_darija else None
        NAME_MAP = {
            "en-small": "openai/whisper-small.en",
            "en-medium": "openai/whisper-medium.en",
            "darija-small": "ychafiqui/whisper-small-darija",
            "darija-medium": "ychafiqui/whisper-medium-darija",
        }
        repo = NAME_MAP[choice]
        self.statusBar().showMessage(f"Loading backend {repo}...")
        QApplication.processEvents()
        self.processor = WhisperProcessor.from_pretrained(repo)
        dtype = torch.float16 if self.device.type == "mps" else torch.float32
        self.model = WhisperForConditionalGeneration.from_pretrained(repo, dtype=dtype).to(self.device)
        self.statusBar().showMessage(f"Loaded {choice}")
        if was_recording:
            self.begin_io()

    def toggle_io(self):
        if not self.recording:
            self.begin_io()
        else:
            self.end_io()

    def begin_io(self):
        if self.model is None or self.processor is None:
            self.load_backend()
        self.recording = True
        self.btn_record.setText("Halt")
        self.waveform.start()
        self.live_text = ""
        self.seg_t0 = time.time()
        self.worker = threading.Thread(
            target=run_decode,
            args=(self, self.fs, self.processor, self.model, self.inbuf, self.text_ready.emit, self.finalize_segment, self.device),
            daemon=True
        )
        self.worker.start()

        # Mic settings (use device default rate; resample is handled in processor)
        sd.default.dtype = ("float32", "float32")
        sd.default.channels = 1
        self.stream = sd.InputStream(
            device=None,               # or set an explicit input index
            samplerate=44100,          # common default on macOS
            channels=1,
            callback=self.on_audio,
            blocksize=int(0.3 * 44100),
            latency="low"
        )
        self.stream.start()
        self.mic_sr = int(self.stream.samplerate)

    def end_io(self):
        self.recording = False
        self.btn_record.setText("Listen")
        self.waveform.stop()
        if hasattr(self, "stream"):
            self.stream.stop()
            self.stream.close()
        if self.worker and self.worker.is_alive():
            self.worker.join(timeout=0.75)
        self.worker = None
        self.statusBar().showMessage("idle")

    def on_audio(self, indata, frames, time_info, status):
        if status:
            print(status)
        self.inbuf.put(indata.copy())
        self.waveform.update_audio(indata.copy())
        x = indata.astype(np.float32).flatten()
        rms = float(np.sqrt(np.mean(x * x) + 1e-12))
        self.vu_level.emit(rms)
        self.speech_state.emit(rms > self.speak_gate)
        self.waveform.is_recording = rms > self.speak_gate

    def finalize_segment(self, now=None):
        now = now or time.time()
        self._finalize_live_segment(end_time=now)
        self.live_text = ""
        self.seg_t0 = now

    def insert_blank(self):
        pass  # permanent transcript style

    def _update_display(self, text: str):
        doc = self.text_display.document()
        cur = self.text_display.textCursor()

        if self.live_anchor_pos is None:
            now = time.time()
            start = datetime.fromtimestamp(self.seg_t0 or now)
            ts_header = f"\n[{start.strftime('%H:%M:%S')} - …] "
            cur.movePosition(QTextCursor.End)
            cur.insertText(ts_header)
            self.live_anchor_pos = cur.position()

        cur = QTextCursor(doc)
        cur.setPosition(self.live_anchor_pos)
        cur.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
        cur.removeSelectedText()
        cur.insertText(text)

        if self._autoscroll:
            self.text_display.moveCursor(QTextCursor.End)

    def _finalize_live_segment(self, end_time=None):
        if self.live_anchor_pos is None:
            return

        doc = self.text_display.document()
        cur = QTextCursor(doc)

        cur.setPosition(self.live_anchor_pos)
        cur.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
        live_text = cur.selectedText()

        header_cursor = QTextCursor(doc)
        header_cursor.setPosition(self.live_anchor_pos)
        header_cursor.movePosition(QTextCursor.StartOfBlock)
        header_cursor.movePosition(QTextCursor.Right, QTextCursor.MoveAnchor, 0)
        header_cursor.setPosition(self.live_anchor_pos, QTextCursor.KeepAnchor)
        header = header_cursor.selectedText()

        t1 = datetime.fromtimestamp(end_time or time.time()).strftime("%H:%M:%S")
        patched_header = header.replace("…]", f"{t1}]")

        header_cursor.removeSelectedText()
        header_cursor.insertText(patched_header)

        cur.setPosition(self.live_anchor_pos)
        cur.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
        cur.removeSelectedText()
        cur.insertText(live_text + "\n")

        self.live_anchor_pos = None

        if self._autoscroll:
            self.text_display.moveCursor(QTextCursor.End)

    def paint_text(self, text: str):
        self._update_display(text)

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Return, Qt.Key_Enter):
            self._finalize_live_segment()
        elif e.key() == Qt.Key_Space:
            self.toggle_io()
        elif e.key() == Qt.Key_Escape:
            self.live_text = ""
            self.text_ready.emit("")

    def closeEvent(self, event):
        if self.recording:
            self.end_io()
        event.accept()


def main():
    app = QApplication(sys.argv)
    w = EchoGateWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()