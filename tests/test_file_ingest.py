"""Tests for the file ingestion module (MarkItDown integration)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agent.file_ingest import (
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE_BYTES,
    _validate_file,
    convert_bytes_to_markdown,
    convert_file_to_markdown,
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
