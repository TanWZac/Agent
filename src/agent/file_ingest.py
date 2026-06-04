"""File ingestion via MarkItDown — converts uploaded files to Markdown for agent consumption.

Supports office documents (docx, xlsx, pptx), PDFs, images, HTML, and many other formats.
See https://github.com/microsoft/markitdown for the full list of supported formats.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List

from markitdown import MarkItDown

from src.core.exceptions import ToolExecutionError
from src.core.logging import get_logger

logger = get_logger("agent.file_ingest")

# Maximum file size allowed (default 50 MB)
MAX_FILE_SIZE_BYTES = int(os.getenv("MAX_UPLOAD_FILE_SIZE_MB", "50")) * 1024 * 1024

# Chunk configuration for vector store persistence
CHUNK_SIZE = int(os.getenv("FILE_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("FILE_CHUNK_OVERLAP", "200"))

# Allowed file extensions (security: prevent arbitrary file processing)
ALLOWED_EXTENSIONS = frozenset({
    ".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt",
    ".html", ".htm", ".txt", ".md", ".csv", ".tsv", ".json",
    ".xml", ".rtf", ".epub", ".ipynb", ".wav", ".mp3",
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp",
    ".zip",
})


def _validate_file(file_path: Path, file_size: int | None = None) -> None:
    """Validate file before processing."""
    ext = file_path.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ToolExecutionError(
            f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}"
        )

    size = file_size or file_path.stat().st_size
    if size > MAX_FILE_SIZE_BYTES:
        raise ToolExecutionError(
            f"File too large ({size / 1024 / 1024:.1f} MB). "
            f"Maximum allowed: {MAX_FILE_SIZE_BYTES / 1024 / 1024:.0f} MB."
        )


def convert_file_to_markdown(file_path: str | Path) -> str:
    """Convert a file to Markdown text using MarkItDown.

    :param file_path: Path to the file to convert.
    :return: Markdown-formatted content of the file.
    :raises ToolExecutionError: If conversion fails or file is invalid.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise ToolExecutionError(f"File not found: {file_path}")

    _validate_file(file_path)

    try:
        md = MarkItDown()
        result = md.convert(str(file_path))
        logger.info("Converted file '%s' to markdown (%d chars)", file_path.name, len(result.text_content))
        return result.text_content
    except Exception as e:
        logger.error("Failed to convert file '%s': %s", file_path.name, e)
        raise ToolExecutionError(f"Failed to convert file '{file_path.name}': {e}") from e


def convert_bytes_to_markdown(content: bytes, filename: str) -> str:
    """Convert file bytes to Markdown using a temporary file.

    :param content: Raw file bytes.
    :param filename: Original filename (used for extension detection).
    :return: Markdown-formatted content.
    :raises ToolExecutionError: If conversion fails or file is invalid.
    """
    file_path = Path(filename)
    _validate_file(file_path, file_size=len(content))

    suffix = file_path.suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        return convert_file_to_markdown(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks for vector store ingestion.

    Uses paragraph boundaries when possible to avoid splitting mid-sentence.

    :param text: Full markdown text to chunk.
    :param chunk_size: Target characters per chunk.
    :param overlap: Number of overlapping characters between chunks.
    :return: List of text chunks.
    """
    if not text.strip():
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    paragraphs = text.split("\n\n")
    current_chunk = ""

    for para in paragraphs:
        # If adding this paragraph exceeds chunk_size, finalize current chunk
        if current_chunk and len(current_chunk) + len(para) + 2 > chunk_size:
            chunks.append(current_chunk.strip())
            # Keep overlap from end of current chunk
            if overlap > 0 and len(current_chunk) > overlap:
                current_chunk = current_chunk[-overlap:] + "\n\n" + para
            else:
                current_chunk = para
        else:
            current_chunk = current_chunk + "\n\n" + para if current_chunk else para

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def ingest_file_to_store(
    file_path: str | Path,
    store,
    filename: str | None = None,
) -> int:
    """Convert a file to markdown, chunk it, and persist all chunks to the note store.

    :param file_path: Path to the file to ingest.
    :param store: NoteStore instance to persist chunks into.
    :param filename: Display name for the file (defaults to basename).
    :return: Number of chunks stored.
    """
    markdown_content = convert_file_to_markdown(file_path)
    fname = filename or Path(file_path).name
    chunks = chunk_text(markdown_content)

    for i, chunk in enumerate(chunks):
        note = f"[File: {fname} | Chunk {i + 1}/{len(chunks)}]\n{chunk}"
        store.append(note)

    logger.info("Ingested file '%s' into store: %d chunks", fname, len(chunks))
    return len(chunks)


def ingest_bytes_to_store(
    content: bytes,
    filename: str,
    store,
) -> int:
    """Convert file bytes to markdown, chunk, and persist to the note store.

    :param content: Raw file bytes.
    :param filename: Original filename.
    :param store: NoteStore instance to persist chunks into.
    :return: Number of chunks stored.
    """
    markdown_content = convert_bytes_to_markdown(content, filename)
    chunks = chunk_text(markdown_content)

    for i, chunk in enumerate(chunks):
        note = f"[File: {filename} | Chunk {i + 1}/{len(chunks)}]\n{chunk}"
        store.append(note)

    logger.info("Ingested file '%s' into store: %d chunks", filename, len(chunks))
    return len(chunks)
