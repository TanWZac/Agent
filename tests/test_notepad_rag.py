from src.notepad_rag import NotepadRAG


def test_append_and_load(tmp_path):
    note_file = tmp_path / "notes.txt"
    rag = NotepadRAG(str(note_file))

    rag.append("User likes concise answers")
    rag.append("Project deadline is Friday")

    notes = rag.load_notes()
    assert notes == ["User likes concise answers", "Project deadline is Friday"]


def test_retrieve_returns_relevant_notes(tmp_path):
    note_file = tmp_path / "notes.txt"
    rag = NotepadRAG(str(note_file))

    rag.append("User lives in Berlin")
    rag.append("Preferred language is Python")
    rag.append("Vacation is in July")

    hits = rag.retrieve("Where does the user live?", k=2)
    assert hits
    assert "Berlin" in hits[0].text
