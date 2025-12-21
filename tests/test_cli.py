from typer.testing import CliRunner
from unittest.mock import patch

from multidoc_rag.app import app


runner = CliRunner()


def test_ingest_command_calls_loader_and_vectorstore(tmp_path):
    with patch("multidoc_rag.loader.load_pdfs") as mock_load, patch(
        "multidoc_rag.vectorstore.build_vectorstore"
    ) as mock_build:
        mock_load.return_value = [type("D", (), {"page_content": "hello world"})()]
        result = runner.invoke(app, ["ingest", "--data-dir", "./data", "--persist-dir", str(tmp_path)])
        assert result.exit_code == 0
        mock_load.assert_called_once()
        mock_build.assert_called_once()
