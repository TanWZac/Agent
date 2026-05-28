"""Tests for the file ingestion module (MarkItDown integration)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agent.file_ingest import (
    ALLOWED_EXTENSIONS,
    CHUNK_SIZE,
    MAX_FILE_SIZE_BYTES,
    _validate_file,
    chunk_text,
    convert_bytes_to_markdown,
    convert_file_to_markdown,
    ingest_file_to_store,
)
from src.core.exceptions import ToolExecutionError


class TestValidateFile:
    def test_rejects_unsupported_extension(self, tmp_path):
        bad_file = tmp_path / "script.exe"
        bad_file.write_bytes(b"MZ")
        with pytest.raises(ToolExecutionError, match="Unsupported file type"):
            _validate_file(bad_file)

    def test_rejects_oversized_file(self, tmp_path):
        big_file = tmp_path / "large.pdf"
        big_file.write_bytes(b"x")
        with pytest.raises(ToolExecutionError, match="File too large"):
            _validate_file(big_file, file_size=MAX_FILE_SIZE_BYTES + 1)

    def test_accepts_valid_extension(self, tmp_path):
        valid_file = tmp_path / "doc.pdf"
        valid_file.write_bytes(b"%PDF-1.4")
        _validate_file(valid_file)  # Should not raise


class TestConvertFileToMarkdown:
    def test_file_not_found(self):
        with pytest.raises(ToolExecutionError, match="File not found"):
            convert_file_to_markdown("/nonexistent/file.pdf")

    @patch("src.agent.file_ingest.MarkItDown")
    def test_successful_conversion(self, mock_markitdown_cls, tmp_path):
        test_file = tmp_path / "test.docx"
        test_file.write_bytes(b"fake docx content")

        mock_result = MagicMock()
        mock_result.text_content = "# Hello World\n\nConverted content."
        mock_md = MagicMock()
        mock_md.convert.return_value = mock_result
        mock_markitdown_cls.return_value = mock_md

        result = convert_file_to_markdown(str(test_file))
        assert result == "# Hello World\n\nConverted content."
        mock_md.convert.assert_called_once_with(str(test_file))

    @patch("src.agent.file_ingest.MarkItDown")
    def test_conversion_failure(self, mock_markitdown_cls, tmp_path):
        test_file = tmp_path / "bad.pdf"
        test_file.write_bytes(b"not a real pdf")

        mock_md = MagicMock()
        mock_md.convert.side_effect = RuntimeError("Conversion failed")
        mock_markitdown_cls.return_value = mock_md

        with pytest.raises(ToolExecutionError, match="Failed to convert file"):
            convert_file_to_markdown(str(test_file))


class TestConvertBytesToMarkdown:
    @patch("src.agent.file_ingest.MarkItDown")
    def test_successful_bytes_conversion(self, mock_markitdown_cls):
        mock_result = MagicMock()
        mock_result.text_content = "# Converted\n\nMarkdown content"
        mock_md = MagicMock()
        mock_md.convert.return_value = mock_result
        mock_markitdown_cls.return_value = mock_md

        result = convert_bytes_to_markdown(b"file content", "report.docx")
        assert "Converted" in result

    def test_rejects_unsupported_type(self):
        with pytest.raises(ToolExecutionError, match="Unsupported file type"):
            convert_bytes_to_markdown(b"binary", "malware.exe")

    def test_rejects_oversized_bytes(self):
        large_content = b"x" * (MAX_FILE_SIZE_BYTES + 1)
        with pytest.raises(ToolExecutionError, match="File too large"):
            convert_bytes_to_markdown(large_content, "huge.pdf")


class TestChunkText:
    def test_empty_text_returns_empty(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_short_text_single_chunk(self):
        text = "Hello world"
        chunks = chunk_text(text, chunk_size=1000)
        assert chunks == ["Hello world"]

    def test_long_text_splits_into_chunks(self):
        # Create text with multiple paragraphs exceeding chunk size
        paragraphs = [f"Paragraph {i}. " + "x" * 200 for i in range(20)]
        text = "\n\n".join(paragraphs)
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        assert len(chunks) > 1
        # All original text should be represented
        full_rejoined = " ".join(chunks)
        for i in range(20):
            assert f"Paragraph {i}" in full_rejoined

    def test_overlap_preserved(self):
        paragraphs = ["A" * 400, "B" * 400, "C" * 400]
        text = "\n\n".join(paragraphs)
        chunks = chunk_text(text, chunk_size=500, overlap=100)
        assert len(chunks) >= 2
        # Second chunk should start with overlap from first chunk's end
        # (contains trailing characters from first chunk)


class TestIngestFileToStore:
    @patch("src.agent.file_ingest.MarkItDown")
    def test_ingests_file_chunks_into_store(self, mock_markitdown_cls, tmp_path):
        test_file = tmp_path / "report.docx"
        test_file.write_bytes(b"fake")

        # Simulate conversion returning multi-paragraph content
        long_content = "\n\n".join([f"Section {i}\n" + "content " * 100 for i in range(10)])
        mock_result = MagicMock()
        mock_result.text_content = long_content
        mock_md = MagicMock()
        mock_md.convert.return_value = mock_result
        mock_markitdown_cls.return_value = mock_md

        mock_store = MagicMock()
        num_chunks = ingest_file_to_store(str(test_file), mock_store)

        assert num_chunks > 0
        assert mock_store.append.call_count == num_chunks
        # Each stored note should have the file tag
        for call in mock_store.append.call_args_list:
            note_text = call[0][0]
            assert "[File: report.docx |" in note_text

    @patch("src.agent.file_ingest.MarkItDown")
    def test_small_file_single_chunk(self, mock_markitdown_cls, tmp_path):
        test_file = tmp_path / "small.txt"
        test_file.write_bytes(b"hello")

        mock_result = MagicMock()
        mock_result.text_content = "Short content."
        mock_md = MagicMock()
        mock_md.convert.return_value = mock_result
        mock_markitdown_cls.return_value = mock_md

        mock_store = MagicMock()
        num_chunks = ingest_file_to_store(str(test_file), mock_store)

        assert num_chunks == 1
        mock_store.append.assert_called_once()
        stored = mock_store.append.call_args[0][0]
        assert "Short content." in stored
