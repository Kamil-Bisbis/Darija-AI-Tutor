# ui/mic_button.py
from __future__ import annotations
import numpy as np
from PySide6.QtCore import Qt, QTimer, QRectF, Signal
from PySide6.QtGui import QPainter, QColor, QPainterPath
from PySide6.QtWidgets import QWidget

class MicHoldButton(QWidget):
    pressed = Signal()
    released = Signal()

    def __init__(self, diameter: int = 132, gate: float = 0.015, parent=None,
                 base_color=(58,158,112), hold_color=(60,200,135), halo_color=(80,220,160,70)):
        super().__init__(parent)
        self.setMinimumSize(diameter, diameter)

        # colors
        self._base_color = QColor(*base_color)
        self._hold_color = QColor(*hold_color)
        self._halo_color = QColor(*halo_color)

        # interaction/animation state
        self._holding = False
        self._scale = 1.0
        self._gate = float(gate)
        self._bars = [0.0] * 24
        self._target = [0.0] * 24

        # animation tick
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(33)  # ~30 FPS

    # ---------- public API ----------
    def start_hold(self):
        if not self._holding:
            self._holding = True
            self._target = [0.0] * len(self._target)
            self.pressed.emit()
            self.update()

    def end_hold(self):
        if self._holding:
            self._holding = False
            self._target = [0.0] * len(self._target)
            self.released.emit()
            self.update()

    def update_audio(self, samples: np.ndarray):
        if samples is None or samples.size == 0:
            return
        s = np.abs(samples)
        n_bars = len(self._bars)
        chunk = max(1, len(s) // n_bars)
        vals = []
        for i in range(0, len(s), chunk):
            a = float(s[i:i + chunk].mean())
            if a <= self._gate:
                vals.append(0.0)
            else:
                vals.append(max(0.0, min(1.0, (a - self._gate) / (0.25 - self._gate))))
        if len(vals) < n_bars:
            vals += [0.0] * (n_bars - len(vals))
        self._target = vals[:n_bars]

    # ---------- internals ----------
    def _tick(self):
        target_scale = 1.06 if self._holding else 1.0
        self._scale += (target_scale - self._scale) * 0.20
        for i in range(len(self._bars)):
            self._bars[i] += (self._target[i] - self._bars[i]) * 0.22
        self.update()

    # ---------- Qt events ----------
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.start_hold(); e.accept()
        else:
            e.ignore()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.end_hold(); e.accept()
        else:
            e.ignore()

    def paintEvent(self, _):
        p = QPainter(self)
        try:
            p.setRenderHint(QPainter.Antialiasing, True)
            w, h = self.width(), self.height()
            d = int(min(w, h) * self._scale)
            cx, cy = w // 2, h // 2
            r = d // 2
            outer = QRectF(cx - r, cy - r, d, d)

            # outer circle
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(48,49,54))
            p.drawEllipse(outer)

            # halo while holding
            if self._holding:
                p.setBrush(self._halo_color)
                p.drawEllipse(cx - r - 6, cy - r - 6, d + 12, d + 12)

            # inner disc
            inner = outer.adjusted(6, 6, -6, -6)
            p.setBrush(self._hold_color if self._holding else self._base_color)
            p.drawEllipse(inner)

            # clip to inner circle and draw bars
            clip = QPainterPath(); clip.addEllipse(inner)
            p.setClipPath(clip)

            n = len(self._bars)
            iw, ih = inner.width(), inner.height()
            left, top = inner.left(), inner.top()
            bar_w = max(2.0, iw / (n * 1.4))
            spacing = (iw - n * bar_w) / (n + 1)
            p.setBrush(QColor(245,252,250))
            p.setPen(Qt.NoPen)
            max_h = ih * 0.56

            for i, a in enumerate(self._bars):
                bh = max(3.0, max_h * float(a))
                x = left + spacing * (i + 1) + bar_w * i
                y = top + (ih - bh) / 2.0
                p.drawRoundedRect(QRectF(x, y, bar_w, bh), 3, 3)
        finally:
            p.end()
