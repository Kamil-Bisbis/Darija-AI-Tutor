from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QColor, QPen
from PySide6.QtWidgets import QWidget

class LevelBar(QWidget):
    def __init__(self, gate=0.009, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMinimumHeight(28)
        self.level = 0.0
        self.peak = 0.0
        self.gate = gate
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._decay_peak)
        self._timer.start(40)

    def _decay_peak(self):
        self.peak *= 0.94
        self.update()

    def set_level(self, rms: float):
        self.level = max(0.0, min(1.0, rms * 12))
        self.peak = max(self.peak, self.level)
        self.update()

    def set_gate(self, g: float):
        self.gate = g
        self.update()

    def paintEvent(self, _):
        w, h = self.width(), self.height()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(28, 28, 28))
        bar_w = int(self.level * w)
        painter.fillRect(0, 0, bar_w, h, QColor(60, 180, 104))
        peak_x = int(self.peak * w)
        pen = QPen(QColor(240, 240, 240, 220), 2)
        painter.setPen(pen)
        painter.drawLine(peak_x, 0, peak_x, h)
        gate_x = int(max(0.0, min(1.0, self.gate * 12)) * w)
        painter.setPen(QPen(QColor(255, 120, 80, 200), 2, Qt.DashLine))
        painter.drawLine(gate_x, 0, gate_x, h)
        painter.end()
