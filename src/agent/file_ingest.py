"""File ingestion via MarkItDown — converts uploaded files to Markdown for agent consumption.

Supports office documents (docx, xlsx, pptx), PDFs, images, HTML, and many other formats.
See https://github.com/microsoft/markitdown for the full list of supported formats.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from markitdown import MarkItDown

from src.core.exceptions import ToolExecutionError
from src.core.logging import get_logger

logger = get_logger("agent.file_ingest")

# Maximum file size allowed (default 50 MB)
MAX_FILE_SIZE_BYTES = int(os.getenv("MAX_UPLOAD_FILE_SIZE_MB", "50")) * 1024 * 1024

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
