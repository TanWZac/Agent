"""Tests for the store factory and ChromaDB backend."""

import pytest

from src.store import NoteStore
from src.store.factory import create_note_store
from src.store.file_store import FileNoteStore


def test_factory_creates_file_store(tmp_path):
    store = create_note_store(backend="file", note_file=str(tmp_path / "notes.txt"))
    assert isinstance(store, FileNoteStore)
    assert isinstance(store, NoteStore)


def test_factory_creates_chroma_store(tmp_path):
    store = create_note_store(
        backend="chroma",
        persist_directory=str(tmp_path / "chroma_db"),
        collection_name="test_notes",
    )
    from src.store.chroma_store import ChromaNoteStore

    assert isinstance(store, ChromaNoteStore)
    assert isinstance(store, NoteStore)


def test_factory_invalid_backend():
    with pytest.raises(ValueError, match="Unknown store backend"):
        create_note_store(backend="postgres")


def test_factory_creates_sqlite_store(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'notes.db'}"
    store = create_note_store(backend="sqlite", db_url=db_url)
    from src.store.sql_store import SqlNoteStore

    assert isinstance(store, SqlNoteStore)
    assert isinstance(store, NoteStore)


def test_sqlite_store_append_load_clear(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'notes.db'}"
    store = create_note_store(backend="sqlite", db_url=db_url)
    store.append("User likes Python")
    store.append("Project uses FastAPI")

    notes = store.load_notes()
    assert "User likes Python" in notes
    assert "Project uses FastAPI" in notes

    store.clear()
    assert store.load_notes() == []


def test_chroma_store_append_and_load(tmp_path):
    store = create_note_store(
        backend="chroma",
        persist_directory=str(tmp_path / "chroma_db"),
        collection_name="test",
    )
    store.append("User likes Python")
    store.append("Project uses FastAPI")

    notes = store.load_notes()
    assert "User likes Python" in notes
    assert "Project uses FastAPI" in notes


def test_chroma_store_retrieve(tmp_path):
    store = create_note_store(
        backend="chroma",
        persist_directory=str(tmp_path / "chroma_db"),
        collection_name="test_retrieve",
    )
    store.append("User lives in Berlin Germany")
    store.append("Preferred language is Python")
    store.append("Vacation planned for July")

    hits = store.retrieve("Where does the user live?", k=2)
    assert hits
    assert "Berlin" in hits[0].text


def test_chroma_store_clear(tmp_path):
    store = create_note_store(
        backend="chroma",
        persist_directory=str(tmp_path / "chroma_db"),
        collection_name="test_clear",
    )
    store.append("something")
    store.clear()
    assert store.load_notes() == []


def test_chroma_store_empty_retrieve(tmp_path):
    store = create_note_store(
        backend="chroma",
        persist_directory=str(tmp_path / "chroma_db"),
        collection_name="test_empty",
    )
    hits = store.retrieve("anything")
    assert hits == []


def test_stores_share_interface(tmp_path):
    """Both backends satisfy the same NoteStore interface."""
    file_store = create_note_store(backend="file", note_file=str(tmp_path / "f.txt"))
    chroma_store = create_note_store(
        backend="chroma",
        persist_directory=str(tmp_path / "c"),
        collection_name="iface_test",
    )
    sqlite_store = create_note_store(
        backend="sqlite",
        db_url=f"sqlite:///{tmp_path / 'iface.db'}",
    )

    for store in [file_store, chroma_store, sqlite_store]:
        store.append("test note")
        assert store.load_notes()
        store.clear()
        assert store.load_notes() == []
