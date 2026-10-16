import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from llm_config import load_config, PROVIDERS
from notebook_support import load_notebook_config, run_notebook


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = (
    "interview_scoring_system.ipynb",
    "Interview_answer_scoring_system-colab_version.ipynb",
)


class NotebookSecretsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.path = self.directory / "llm_config.json"
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def configure(self, provider, **profile):
        self.path.write_text(json.dumps({
            "provider": provider, "providers": {provider: profile}
        }), encoding="utf-8")

    def test_secret_getter_only_reads_selected_provider(self):
        for provider in ("gemini", "mistral_free", "groq_free", "openrouter"):
            with self.subTest(provider=provider):
                self.configure(provider)
                getter = Mock(return_value="test-selected-key")
                config = load_config(self.path, secret_getter=getter)
                getter.assert_called_once_with(PROVIDERS[provider]["api_key_env"])
                self.assertEqual(config.llm.api_key, "test-selected-key")

    def test_environment_prevents_secret_lookup(self):
        self.configure("groq_free")
        os.environ["GROQ_API_KEY"] = "test-environment-key"
        getter = Mock(side_effect=AssertionError("Must not read Secrets"))
        self.assertEqual(load_config(self.path, secret_getter=getter).llm.api_key, "test-environment-key")
        getter.assert_not_called()

    def test_key_file_prevents_secret_lookup(self):
        self.configure("mistral_free")
        (self.directory / "mistral.key").write_text("test-file-key", encoding="utf-8")
        getter = Mock()
        self.assertEqual(load_config(self.path, secret_getter=getter).llm.api_key, "test-file-key")
        getter.assert_not_called()

    def test_local_mode_never_reads_colab_secret(self):
        self.configure("local")
        getter = Mock(side_effect=AssertionError("Must not read Secrets"))
        self.assertEqual(load_config(self.path, secret_getter=getter).llm.api_key, "")
        getter.assert_not_called()
        # Colab need not be installed, even when its optional mode is enabled.
        self.assertEqual(load_notebook_config(self.path, use_colab_secrets=True).llm.provider, "local")

    def test_custom_secret_name(self):
        self.configure("groq_free", api_key_env="MY_GROQ_KEY")
        getter = Mock(return_value="test-key")
        self.assertEqual(load_config(self.path, secret_getter=getter).api_key_env, "MY_GROQ_KEY")
        getter.assert_called_once_with("MY_GROQ_KEY")

    def test_secret_failure_hides_backend_exception(self):
        self.configure("groq_free")
        getter = Mock(side_effect=RuntimeError("sensitive-backend-value"))
        with self.assertRaises(ValueError) as error:
            load_config(self.path, secret_getter=getter)
        self.assertNotIn("sensitive-backend-value", str(error.exception))
        self.assertIn("GROQ_API_KEY", str(error.exception))

    def test_empty_or_non_string_secret_is_rejected(self):
        self.configure("groq_free")
        for value in (None, "", " ", 42):
            with self.subTest(value=value), self.assertRaises(ValueError):
                load_config(self.path, secret_getter=Mock(return_value=value))

    def test_colab_getter_is_lazy_and_uses_selected_name(self):
        self.configure("openrouter")
        getter = Mock(return_value="test-colab-key")
        fake = SimpleNamespace(userdata=SimpleNamespace(get=getter))
        with patch.dict(sys.modules, {"google.colab": fake}):
            config = load_notebook_config(self.path, use_colab_secrets=True)
        getter.assert_called_once_with("OPENROUTER_API_KEY")
        self.assertEqual(config.llm.api_key, "test-colab-key")

    def test_secret_passed_only_in_child_environment(self):
        self.configure("groq_free")
        (self.directory / "app.py").write_text("# subprocess fixture\n", encoding="utf-8")
        getter = Mock(return_value="test-private-secret")
        fake = SimpleNamespace(userdata=SimpleNamespace(get=getter))
        output = self.directory / "scored_answers.csv"
        output.write_text("score,confidence,reasoning\n3,0.8,ok\n", encoding="utf-8")
        with patch.dict(sys.modules, {"google.colab": fake}), patch(
            "notebook_support.subprocess.run", return_value=subprocess.CompletedProcess([], 0)
        ) as run:
            frame = run_notebook(self.path, self.directory, use_colab_secrets=True)
        command = run.call_args.args[0]
        self.assertNotIn("test-private-secret", " ".join(command))
        self.assertEqual(run.call_args.kwargs["env"]["GROQ_API_KEY"], "test-private-secret")
        self.assertTrue(run.call_args.kwargs["capture_output"])
        self.assertNotIn("GROQ_API_KEY", os.environ)
        self.assertFalse((self.directory / "groq.key").exists())
        self.assertFalse(frame.attrs["partial_results"])

    def test_failed_run_does_not_display_old_csv(self):
        self.configure("local")
        (self.directory / "app.py").write_text("# fixture\n", encoding="utf-8")
        (self.directory / "scored_answers.csv").write_text("score\n999\n", encoding="utf-8")
        with patch("notebook_support.subprocess.run", return_value=subprocess.CompletedProcess([], 1)):
            with patch("notebook_support.pd.read_csv") as read_csv:
                with self.assertRaises(subprocess.CalledProcessError):
                    run_notebook(self.path, self.directory)
                read_csv.assert_not_called()

    def test_partial_results_are_explicit(self):
        self.configure("local")
        (self.directory / "app.py").write_text("# fixture\n", encoding="utf-8")
        (self.directory / "scored_answers.csv").write_text("score\n3\n", encoding="utf-8")
        with patch("notebook_support.subprocess.run", return_value=subprocess.CompletedProcess([], 2)):
            with self.assertWarnsRegex(RuntimeWarning, "partial results"):
                frame = run_notebook(self.path, self.directory)
        self.assertTrue(frame.attrs["partial_results"])

    def test_explicit_config_is_required(self):
        with self.assertRaises(ValueError):
            load_notebook_config(self.path)

    def test_configured_secondary_secrets_are_forwarded_privately(self):
        self.path.write_text(json.dumps({
            "primary": {"provider": "groq_free"},
            "secondary": {"provider": "gemini"},
        }), encoding="utf-8")
        (self.directory / "app.py").write_text("# fixture\n", encoding="utf-8")
        (self.directory / "scored_answers.csv").write_text("score\n3\n", encoding="utf-8")
        values = {"GROQ_API_KEY": "test-primary", "GEMINI_API_KEY": "test-secondary"}
        getter = Mock(side_effect=values.__getitem__)
        fake = SimpleNamespace(userdata=SimpleNamespace(get=getter))
        with patch.dict(sys.modules, {"google.colab": fake}), patch(
            "notebook_support.subprocess.run", return_value=subprocess.CompletedProcess([], 0)
        ) as run:
            run_notebook(self.path, self.directory, use_colab_secrets=True)
        self.assertEqual(run.call_args.kwargs["env"]["GROQ_API_KEY"], "test-primary")
        self.assertEqual(run.call_args.kwargs["env"]["GEMINI_API_KEY"], "test-secondary")
        self.assertNotIn("GROQ_API_KEY", os.environ)
        self.assertNotIn("GEMINI_API_KEY", os.environ)
        self.assertNotIn("test-secondary", " ".join(run.call_args.args[0]))

    def test_missing_secondary_secret_does_not_block_primary_subprocess(self):
        self.path.write_text(json.dumps({
            "primary": {"provider": "local"}, "secondary": {"provider": "groq_free"},
        }), encoding="utf-8")
        (self.directory / "app.py").write_text("# fixture\n", encoding="utf-8")
        (self.directory / "scored_answers.csv").write_text("score\n3\n", encoding="utf-8")
        getter = Mock(side_effect=RuntimeError("Secret access denied"))
        fake = SimpleNamespace(userdata=SimpleNamespace(get=getter))
        with patch.dict(sys.modules, {"google.colab": fake}), patch(
            "notebook_support.subprocess.run", return_value=subprocess.CompletedProcess([], 0)
        ) as run:
            with self.assertWarnsRegex(RuntimeWarning, "primary will still run"):
                result = run_notebook(self.path, self.directory, use_colab_secrets=True)
        run.assert_called_once()
        self.assertEqual(result.iloc[0]["score"], 3)
        getter.assert_called_once_with("GROQ_API_KEY")


class NotebookExecutionTests(unittest.TestCase):
    def test_notebooks_are_clean_and_code_cells_compile(self):
        for name in NOTEBOOKS:
            with self.subTest(notebook=name):
                notebook = json.loads((ROOT / name).read_text(encoding="utf-8"))
                self.assertEqual(notebook["nbformat"], 4)
                for index, cell in enumerate(notebook["cells"]):
                    if cell["cell_type"] == "code":
                        self.assertEqual(cell["outputs"], [])
                        self.assertIsNone(cell["execution_count"])
                        compile("".join(cell["source"]), f"{name}:cell{index}", "exec")

    def test_local_notebook_end_to_end(self):
        self.execute_notebook(NOTEBOOKS[0])

    def test_colab_notebook_end_to_end_without_colab_or_cloud_key(self):
        self.execute_notebook(NOTEBOOKS[1])

    def execute_notebook(self, name):
        class Handler(BaseHTTPRequestHandler):
            calls = 0

            def log_message(self, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                Handler.calls += 1
                content = json.dumps({"score": 3, "confidence": 0.9, "reasoning": "Test rubric."})
                body = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                work = Path(directory)
                for module in ("app.py", "local_llm.py", "llm_config.py", "llm_providers.py",
                               "scoring_engine.py", "notebook_support.py", "security.py", "guardrails.py"):
                    shutil.copy2(ROOT / module, work / module)
                shutil.copytree(ROOT / "specs", work / "specs")
                (work / "input.csv").write_text("Question\nAnswer\n", encoding="utf-8")
                (work / "rules.json").write_text(json.dumps({
                    "questions": [{"Question": ["Score from 1 to 5.", 1, 5]}]
                }), encoding="utf-8")
                (work / "llm_config.json").write_text(json.dumps({
                    "provider": "local", "input_csv": "input.csv", "rules_json": "rules.json",
                    "providers": {"local": {
                        "base_url": f"http://127.0.0.1:{server.server_port}/v1",
                        "model": "notebook-test",
                    }},
                }), encoding="utf-8")
                original_app = (work / "app.py").read_bytes()
                namespace = {"REPO_DIR": work, "__name__": "__main__"}
                notebook = json.loads((ROOT / name).read_text(encoding="utf-8"))
                with contextlib.chdir(work), patch.object(sys, "path", list(sys.path)), contextlib.redirect_stdout(io.StringIO()):
                    for index, cell in enumerate(notebook["cells"]):
                        if cell["cell_type"] == "code" and "setup" not in cell["metadata"].get("tags", []):
                            exec(compile("".join(cell["source"]), f"{name}:cell{index}", "exec"), namespace)
                frame = namespace["result_df"]
                self.assertEqual(len(frame), 1)
                self.assertEqual(frame.iloc[0]["score"], 3)
                self.assertEqual(frame.attrs["provider"], "local")
                self.assertFalse(frame.attrs["partial_results"])
                self.assertTrue((work / "scored_answers.csv").is_file())
                self.assertEqual((work / "app.py").read_bytes(), original_app)
                self.assertEqual(Handler.calls, 1)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
