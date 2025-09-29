# utils/turn_manager.py
from __future__ import annotations

class TurnManager:
    def __init__(self, turns: list[dict]):
        if not turns:
            raise ValueError("Empty lesson turns")
        self.turns = turns
        self.idx = 0
        self.fails = 0

    @property
    def current(self) -> dict:
        return self.turns[self.idx]

    def should_advance(self, intent_ok: bool, edit_distance: int) -> bool:
        hard_ok = intent_ok and edit_distance <= 3
        soft_ok = intent_ok and edit_distance <= 5 and self.fails >= 2
        return hard_ok or soft_ok

    def on_result(self, intent_ok: bool, edit_distance: int) -> tuple[bool, bool]:
        """
        Returns (advance, speak_hint).
        """
        if self.should_advance(intent_ok, edit_distance):
            self.fails = 0
            return True, False
        self.fails += 1
        if self.fails == 2:
            return False, True
        return False, False

    def advance(self) -> None:
        self.idx += 1
        if self.idx >= len(self.turns):
            self.idx = 0  # loop forever
            # if you prefer to stop at the end, handle it in caller
