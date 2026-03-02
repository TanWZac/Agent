from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class RetrievedNote:
    text: str
    score: float


class NotepadRAG:
    """Persistent notepad + lightweight retrieval for grounding responses."""

    def __init__(self, note_file: str = "data/notepad.txt") -> None:
        self.note_file = Path(note_file)
        self.note_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.note_file.exists():
            self.note_file.write_text("", encoding="utf-8")

    def append(self, note: str) -> None:
        normalized = note.strip()
        if not normalized:
            return
        with self.note_file.open("a", encoding="utf-8") as f:
            f.write(normalized + "\n")

    def load_notes(self) -> List[str]:
        content = self.note_file.read_text(encoding="utf-8")
        return [line.strip() for line in content.splitlines() if line.strip()]

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))

    def retrieve(self, query: str, k: int = 3) -> List[RetrievedNote]:
        notes = self.load_notes()
        if not notes:
            return []

        q_tokens = self._tokenize(query)
        scored: List[RetrievedNote] = []

        for note in notes:
            n_tokens = self._tokenize(note)
            if not n_tokens:
                continue
            overlap = len(q_tokens & n_tokens)
            score = overlap / len(q_tokens | n_tokens) if (q_tokens | n_tokens) else 0.0
            if score > 0:
                scored.append(RetrievedNote(text=note, score=score))

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:k]
