import pytest

from src.core.exceptions import NotepadFullError
from src.store.file_store import FileNoteStore


def test_append_and_load(tmp_path):
    note_file = tmp_path / "notes.txt"
    rag = FileNoteStore(str(note_file))

    rag.append("User likes concise answers")
    rag.append("Project deadline is Friday")

    notes = rag.load_notes()
    assert notes == ["User likes concise answers", "Project deadline is Friday"]


def test_append_empty_note_ignored(tmp_path):
    note_file = tmp_path / "notes.txt"
    rag = FileNoteStore(str(note_file))

    rag.append("")
    rag.append("   ")
    assert rag.load_notes() == []


def test_retrieve_returns_relevant_notes(tmp_path):
    note_file = tmp_path / "notes.txt"
    rag = FileNoteStore(str(note_file))

    rag.append("User lives in Berlin")
    rag.append("Preferred language is Python")
    rag.append("Vacation is in July")

    hits = rag.retrieve("Where does the user live?", k=2)
    assert hits
    assert "Berlin" in hits[0].text


def test_retrieve_stop_words_dont_dominate(tmp_path):
    """Semantic search should rank Python-related note higher."""
    note_file = tmp_path / "notes.txt"
    rag = FileNoteStore(str(note_file))

    rag.append("The weather is nice today")
    rag.append("Python version 3.12 released")

    hits = rag.retrieve("Python programming language", k=2)
    assert hits
    assert "Python" in hits[0].text


def test_retrieve_empty_notepad(tmp_path):
    note_file = tmp_path / "notes.txt"
    rag = FileNoteStore(str(note_file))

    hits = rag.retrieve("anything")
    assert hits == []


def test_notepad_full_error(tmp_path):
    note_file = tmp_path / "notes.txt"
    # Set max to 1 byte to trigger limit immediately
    rag = FileNoteStore(str(note_file), max_size_mb=1)
    # Write enough data to exceed 1MB
    note_file.write_text("x" * (1024 * 1024 + 1), encoding="utf-8")

    with pytest.raises(NotepadFullError):
        rag.append("This should fail")


def test_clear(tmp_path):
    note_file = tmp_path / "notes.txt"
    rag = FileNoteStore(str(note_file))

    rag.append("some note")
    rag.clear()
    assert rag.load_notes() == []
