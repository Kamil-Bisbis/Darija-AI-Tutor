# whisper_gui.py
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

# stdlib
import sys, threading, time, queue
from datetime import datetime

# third party
import numpy as np
import sounddevice as sd
import torch
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QTextEdit,
    QComboBox, QLabel, QHBoxLayout, QProgressBar, QCheckBox
)
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

# local
from ui.mic_button import MicHoldButton
from asr.decoder import run_decode
from utils.arabizi import (
    arabic_to_arabizi, has_arabic_chars, detect_lang,
    mentions_darija_word, normalize_mishears
)
from llm.tutor_client import ask_llm
from llm.topics import extract_topics

class PushToTalkWindow(QMainWindow):
    text_ready = Signal(str)
    break_line = Signal()
    tutor_text = Signal(str)
    progress = Signal(int)
    finalize_sig = Signal(float)

    def __init__(self):
        super().__init__()

        # runtime
        self.recording = False
        self.is_holding = False
        self.inbuf = queue.Queue()
        self.fs = 16000
        self.worker = None
        self.live_text = ""
        self.seg_t0 = time.time()
        self._autoscroll = True
        self.mic_sr = 16000
        self.stream = None
        self._active_mic = None

        # models
        self.device = torch.device("cpu")
        self.model = None
        self.processor = None
        self.lang_hint = "auto"  # always 'auto' with mixed Whisper

        # personalization topics (rolling window)
        self.user_topics = []

        # UI
        self._init_ui()
        self._connect_signals()
        self.load_backend()

    # ---------------- UI ----------------
    def _init_ui(self):
        main_layout = QVBoxLayout()

        # Controls
        controls = QHBoxLayout()
        controls.addStretch(1)
        lbl = QLabel("Model:")
        lbl.setStyleSheet("color:#ccc;")
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Standard", "Advanced"])  # small / medium
        self.model_combo.setCurrentText("Standard")
        self.model_combo.setFixedWidth(180)
        controls.addWidget(lbl)
        controls.addWidget(self.model_combo)

        # Arabizi display toggle (affects live TRANSCRIPT display only)
        self.cb_arabizi = QCheckBox("Arabizi")
        self.cb_arabizi.setChecked(True)
        controls.addWidget(self.cb_arabizi)
        controls.addStretch(1)
        main_layout.addLayout(controls)
        main_layout.addSpacing(8)

        # Two mic buttons
        top_box = QHBoxLayout(); top_box.addStretch(1)

        self.mic_en = MicHoldButton(
            diameter=132,
            base_color=(58,158,112),
            hold_color=(60,200,135),
            halo_color=(80,220,160,70),
        )
        col_en = QVBoxLayout()
        lab_en = QLabel("English"); lab_en.setStyleSheet("color:#ccc;"); lab_en.setAlignment(Qt.AlignHCenter)
        col_en.addWidget(self.mic_en, alignment=Qt.AlignCenter); col_en.addWidget(lab_en, alignment=Qt.AlignCenter)
        top_box.addLayout(col_en); top_box.addSpacing(24)

        self.mic_ar = MicHoldButton(
            diameter=132,
            base_color=(178,68,68),
            hold_color=(220,82,82),
            halo_color=(255,120,120,70),
        )
        col_ar = QVBoxLayout()
        lab_ar = QLabel("Darija"); lab_ar.setStyleSheet("color:#ccc;"); lab_ar.setAlignment(Qt.AlignHCenter)
        col_ar.addWidget(self.mic_ar, alignment=Qt.AlignCenter); col_ar.addWidget(lab_ar, alignment=Qt.AlignCenter)
        top_box.addLayout(col_ar); top_box.addStretch(1)
        main_layout.addLayout(top_box, stretch=1)
        main_layout.addSpacing(8)

        # Transcript
        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        self.text_display.setStyleSheet("QTextEdit { background: #232629; color: #ECEFF4; }")
        self.text_display.setFont(QFont("Menlo", 11))
        main_layout.addWidget(self.text_display, stretch=2)

        central = QWidget(); central.setLayout(main_layout)
        self.setCentralWidget(central)
        self.statusBar().showMessage("hold-to-talk ready")

        # Progress
        self.prog = QProgressBar(); self.prog.setRange(0, 100); self.prog.setValue(0)
        self.prog.setTextVisible(True); self.prog.setFixedWidth(160)
        self.statusBar().addPermanentWidget(self.prog); self.prog.setVisible(False)

        self.setWindowTitle("Darija AI Tutor")
        self.setGeometry(100, 100, 900, 640)

        # autoscroll tracking
        sb = self.text_display.verticalScrollBar()
        def _on_scroll(_): self._autoscroll = (sb.value() >= sb.maximum() - 2)
        sb.valueChanged.connect(_on_scroll)

    def _connect_signals(self):
        self.text_ready.connect(self.paint_text)
        self.break_line.connect(self.insert_blank)
        self.tutor_text.connect(self._append_tutor)
        self.progress.connect(self._on_progress)
        self.finalize_sig.connect(self._finalize_live_segment)

        self.mic_en.pressed.connect(lambda: self.begin_io("en"))
        self.mic_en.released.connect(self.end_io)
        self.mic_ar.pressed.connect(lambda: self.begin_io("ar"))
        self.mic_ar.released.connect(self.end_io)

        self.model_combo.currentTextChanged.connect(self.load_backend)

    # ---------------- transcript painters ----------------
    def paint_text(self, text: str):
        self._update_display(text)

    def _append_tutor(self, line: str):
        cur = self.text_display.textCursor()
        cur.movePosition(QTextCursor.End)
        ts = datetime.now().strftime("%H:%M:%S")
        cur.insertText(f"[{ts}] {line}\n")
        self.text_display.moveCursor(QTextCursor.End)

    def insert_blank(self):
        cur = self.text_display.textCursor()
        cur.movePosition(QTextCursor.End)
        cur.insertText("\n")
        self.text_display.moveCursor(QTextCursor.End)

    def _update_display(self, text: str):
        doc = self.text_display.document()
        cur = self.text_display.textCursor()
        if getattr(self, "live_anchor_pos", None) is None:
            now = time.time()
            start = datetime.fromtimestamp(getattr(self, "seg_t0", now))
            ts_header = f"\n[{start.strftime('%H:%M:%S')} - ...] "
            cur.movePosition(QTextCursor.End)
            cur.insertText(ts_header)
            self.live_anchor_pos = cur.position()

        cur = QTextCursor(doc)
        cur.setPosition(self.live_anchor_pos)
        cur.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
        cur.removeSelectedText()
        cur.insertText(text)
        self.text_display.moveCursor(QTextCursor.End)

    def _on_progress(self, p: int):
        self.prog.setVisible(True)
        self.prog.setValue(max(0, min(100, int(p))))

    # ---------------- Models ----------------
    def load_backend(self):
        was_recording = self.recording
        if was_recording: self.end_io()

        NAME_MAP = {
            "Standard": "openai/whisper-small",
            "Advanced": "openai/whisper-medium",
        }
        choice = self.model_combo.currentText() if hasattr(self, "model_combo") else "Standard"
        repo = NAME_MAP[choice].replace(" ", "").replace("\u00A0","").replace("\u200B","").strip()
        print("HF repo =", repr(repo))
        self.lang_hint = "auto"

        self.statusBar().showMessage(f"Loading {choice} from {repo} on cpu...")
        QApplication.processEvents()

        self.processor = WhisperProcessor.from_pretrained(repo)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            repo, torch_dtype=torch.float32
        ).to(torch.device("cpu"))
        self.device = torch.device("cpu")

        # silence HF warning about do_sample/temperature
        try:
            gc = self.model.generation_config
            if hasattr(gc, "temperature"):
                gc.temperature = None
            if hasattr(gc, "do_sample"):
                gc.do_sample = False
        except Exception:
            pass

        try:
            if getattr(self.model.generation_config, "forced_decoder_ids", None) is not None:
                self.model.generation_config.forced_decoder_ids = None
        except Exception: pass
        try:
            self.model.generation_config.use_cache = False
        except Exception: pass

        self.statusBar().showMessage(f"Loaded: {choice} on cpu")
        if was_recording: self.begin_io("en")

    # ---------------- IO ----------------
    def begin_io(self, forced_input_lang: str | None = None):
        self.active_input_lang = (forced_input_lang or self.lang_hint or "auto").lower()
        self.lang_hint = self.active_input_lang  # hint for this segment only

        if self.model is None or self.processor is None: self.load_backend()
        if self.recording: return

        self.recording = True
        self.is_holding = True
        self.live_text = ""
        self.seg_t0 = time.time()

        sd.default.dtype = ("float32", "float32")
        sd.default.channels = 1
        self.stream = sd.InputStream(channels=1, dtype="float32", callback=self.on_audio)
        self.stream.start()
        self.mic_sr = int(self.stream.samplerate)

        self.prog.setVisible(True); self.prog.setValue(0)

        self._active_mic = self.mic_ar if self.active_input_lang == "ar" else self.mic_en
        self._active_mic.start_hold()

        def _emit_cb(t: str):
            # Show live transcript (Darija → optionally Arabizi for display only)
            if self.active_input_lang == "ar" and self.cb_arabizi.isChecked():
                t = arabic_to_arabizi(t)
            self.text_ready.emit(t)

        self.worker = threading.Thread(
            target=run_decode,
            args=(self, self.fs, self.processor, self.model, self.inbuf,
                  _emit_cb, self.finalize_sig.emit, self.device, self.progress.emit,
                  self.active_input_lang),
            daemon=True
        )
        self.worker.start()
        self.statusBar().showMessage("recording...")

    def end_io(self):
        if not self.recording: return
        self.recording = False
        self.is_holding = False

        try:
            if self.stream: self.stream.stop(); self.stream.close()
        except Exception: pass
        self.stream = None

        try:
            if self._active_mic is not None: self._active_mic.end_hold()
        except Exception: pass

        if self.worker and self.worker.is_alive(): self.worker.join(timeout=0.75)
        self.worker = None
        self.statusBar().showMessage("processing...")

    def on_audio(self, indata, frames, time_info, status):
        if status: print(status)
        self.inbuf.put(indata.copy())
        try:
            if self._active_mic is not None:
                self._active_mic.update_audio(indata.astype(np.float32).flatten())
        except Exception:
            pass

    # ---------------- finalize + LLM ----------------
    def _finalize_live_segment(self, end_time=None):
        if getattr(self, "live_anchor_pos", None) is None: return ""
        doc = self.text_display.document()
        cur = self.text_display.textCursor()
        cur.setPosition(self.live_anchor_pos)
        cur.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
        live_text = cur.selectedText()

        header_cursor = QTextCursor(doc)
        header_cursor.setPosition(self.live_anchor_pos)
        header_cursor.movePosition(QTextCursor.StartOfBlock)
        header_cursor.setPosition(self.live_anchor_pos, QTextCursor.KeepAnchor)
        header = header_cursor.selectedText()

        ts = datetime.fromtimestamp(end_time or time.time()).strftime("%H:%M:%S")
        header_cursor.removeSelectedText()
        header_cursor.insertText(header.replace("...]", f"{ts}]"))

        cur.setPosition(self.live_anchor_pos)
        cur.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
        cur.removeSelectedText()
        cur.insertText(live_text + "\n")
        self.live_anchor_pos = None
        self.text_display.moveCursor(QTextCursor.End)

        # Route + Topics + LLM
        text = live_text.strip()
        if text:
            lang_in = getattr(self, "active_input_lang", None) or detect_lang(text)
            want_script = "arabic" if has_arabic_chars(live_text) else "arabizi"
            try:
                self._topics = extract_topics(text, getattr(self, "_topics", []))
            except Exception:
                pass
            def _router_worker():
                try:
                    from llm.router import route
                    topics_tuple = tuple(getattr(self, "_topics", []) or [])
                    reply = route(text, lang_in, want_script, topics=topics_tuple)
                except Exception as e:
                    reply = f"(LLM error: {e})"
                self.tutor_text.emit(f"[Tutor] {reply}")
                self.statusBar().showMessage("idle")
            threading.Thread(target=_router_worker, daemon=True).start()

        self.statusBar().showMessage("idle")
        self.prog.setValue(100); self.prog.setVisible(False)

        # reset hints
        self.lang_hint = "auto"
        if hasattr(self, "active_input_lang"): delattr(self, "active_input_lang")
        if hasattr(self, "_active_mic"): delattr(self, "_active_mic")
        return live_text

    # Spacebar defaults to English mic
    def keyPressEvent(self, e):
        if e.isAutoRepeat(): return
        if e.key() == Qt.Key_Space: self.begin_io("en")

    def keyReleaseEvent(self, e):
        if e.isAutoRepeat(): return
        if e.key() == Qt.Key_Space: self.end_io()

    def closeEvent(self, event):
        if self.recording: self.end_io()
        event.accept()

def main():
    app = QApplication(sys.argv)
    w = PushToTalkWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()