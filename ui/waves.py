import numpy as np
from PySide6.QtCore import Qt, QTimer, QRectF
from PySide6.QtGui import QPainter, QColor, QLinearGradient, QPainterPath, QFont
from PySide6.QtWidgets import QWidget

class WaveScope(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(100)
        self.waves = [0.0] * 30
        self.target_waves = [0.0] * 30
        self.is_recording = False
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._animate)
        self._timer.start(33)

    def start(self):
        self.is_recording = True
        self.waves = [0.1] * 30

    def stop(self):
        self.is_recording = False
        self.target_waves = [0.0] * 30

    def update_audio(self, data: np.ndarray):
        if data.size == 0:
            return
        normalized = np.abs(data) / (np.max(np.abs(data)) + 1e-10)
        chunk = max(1, len(normalized) // 30)
        t = [normalized[i:i + chunk].mean() * 1.2 for i in range(0, len(normalized), chunk)]
        t = t[:30]
        # normal jitter for visual life
        t = [max(0.0, v * (1.0 + np.random.normal(0.0, 0.12))) for v in t]
        if len(t) < 30:
            t += [0.0] * (30 - len(t))
        self.target_waves = t

    def _animate(self):
        for i in range(30):
            target = self.target_waves[i] if self.is_recording else 0.0
            self.waves[i] += (target - self.waves[i]) * 0.18
        self.update()

    def paintEvent(self, _):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w = self.width()
        h = self.height()
        mid = h / 2
        bar_w = w / (len(self.waves) * 1.5)
        max_h = h * 0.85
        grad = QLinearGradient(0, 0, w, 0)
        if self.is_recording:
            grad.setColorAt(0, QColor(52, 211, 153))
            grad.setColorAt(1, QColor(16, 185, 129))
        else:
            grad.setColorAt(0, QColor(52, 211, 153, 200))
            grad.setColorAt(1, QColor(16, 185, 129, 200))
        painter.setPen(Qt.NoPen)
        painter.setBrush(grad)
        radius = 6
        for i, a in enumerate(self.waves):
            x = w * i / len(self.waves)
            bar_h = max_h * a
            r = QRectF(x + bar_w / 2, mid - bar_h / 2, bar_w, bar_h)
            if self.is_recording and a > 0.1:
                glow = QPainterPath()
                glow.addRoundedRect(r, radius, radius)
                painter.fillPath(glow, QColor(52, 211, 153, 40))
            painter.drawRoundedRect(r, radius, radius)
        painter.end()
