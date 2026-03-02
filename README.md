# LangGraph Notepad RAG Agent

This project creates a **feature-branch ready patch** with:

- A LangGraph agent capable of tool-calling.
- Web search support (DuckDuckGo via LangChain community tool).
- A persistent notepad saved to a `.txt` file.
- Lightweight RAG retrieval over saved notes to ground responses in prior conversation context.

## How it works

1. The agent uses a tool-enabled `ChatOpenAI` model inside a LangGraph flow.
2. Tools exposed to the model:
   - `search_web(query)` for web search.
   - `save_note(note)` to write memory to notepad text file.
   - `retrieve_notes(question)` to fetch top relevant notes.
3. `NotepadRAG` stores notes in `data/notepad.txt` and runs token-overlap retrieval (Jaccard-style scoring).
4. The CLI app appends each user/assistant turn to the notepad, creating persistent conversational memory.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
export OPENAI_API_KEY=your_key_here
```

## Run

```bash
notepad-agent --note-file data/notepad.txt
```

Type `exit` to quit.

## Test

```bash
pytest -q
```
