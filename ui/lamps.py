from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QColor, QFont
from PySide6.QtWidgets import QWidget

class TalkLamp(QWidget):
    def __init__(self):
        super().__init__()
        self._on = False
        self.setMinimumHeight(26)

    def set_state(self, on: bool):
        if self._on != on:
            self._on = on
            self.update()

    def paintEvent(self, _):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        bg = QColor(50, 50, 50) if not self._on else QColor(46, 204, 113)
        fg = QColor(220, 220, 220) if not self._on else QColor(12, 28, 18)
        painter.setBrush(bg)
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(0, 0, w, h, h/2, h/2)
        painter.setPen(fg)
        painter.setFont(QFont("Menlo", 11, QFont.Bold))
        text = "Listening" if self._on else "Idle"
        painter.drawText(self.rect(), Qt.AlignCenter, text)
        painter.end()
