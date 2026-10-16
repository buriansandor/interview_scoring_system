"""Real loopback HTTP tests; no actual LLM or provider API is contacted."""

import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch
from urllib.error import HTTPError

from local_llm import LocalLLMClient, LocalLLMConfig


ROOT = Path(__file__).resolve().parents[2]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        self.server.received.append((self.path, dict(self.headers), json.loads(self.rfile.read(length))))
        if self.server.redirect:
            self.send_response(307)
            self.send_header("Location", "/redirected")
            self.end_headers()
            return
        responses = getattr(self.server, "evaluation_responses", None)
        value = responses.pop(0) if responses else {"score": 3, "confidence": 0.8, "reasoning": "A rubrika alapján."}
        content = json.dumps(value)
        body = json.dumps({"choices": [{"message": {"role": "assistant", "content": content}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class LocalHTTPTests(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.received = []
        self.server.redirect = False
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}/v1"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def test_chat_protocol_and_proxy_bypass(self):
        client = LocalLLMClient(LocalLLMConfig(base_url=self.base_url, api_key="test-only"))
        with patch.dict(os.environ, {"HTTP_PROXY": "http://127.0.0.1:1", "NO_PROXY": ""}):
            value = json.loads(client.complete([{"role": "user", "content": "Kérdés"}]))
        self.assertEqual(value["score"], 3)
        path, headers, payload = self.server.received[0]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(headers["Authorization"], "Bearer test-only")
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertFalse(payload["stream"])
        self.assertNotIn("tools", payload)
        self.assertEqual(payload["messages"][0]["content"], "Kérdés")

    def test_response_format_modes(self):
        for mode in ("none", "json_schema"):
            client = LocalLLMClient(LocalLLMConfig(base_url=self.base_url, json_mode=mode))
            client.complete([{"role": "user", "content": "Test"}])
            payload = self.server.received[-1][2]
            if mode == "none":
                self.assertNotIn("response_format", payload)
            else:
                self.assertTrue(payload["response_format"]["json_schema"]["strict"])
                self.assertEqual(payload["response_format"]["type"], "json_schema")

    def test_redirect_is_not_followed(self):
        self.server.redirect = True
        client = LocalLLMClient(LocalLLMConfig(base_url=self.base_url))
        with self.assertRaises(HTTPError) as error:
            client.complete([{"role": "user", "content": "Sensitive answer"}])
        self.assertEqual(error.exception.code, 307)
        error.exception.close()
        self.assertEqual(len(self.server.received), 1)

    def test_app_end_to_end_without_gemini_key(self):
        self._run_app()

    def test_app_end_to_end_with_default_json_config(self):
        self._run_app("llm_config.json")

    def test_app_end_to_end_with_custom_json_config(self):
        self._run_app("backend-settings.json")

    def _run_app(self, config_name=None):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / "specs").mkdir()
            for filename in ("context.md", "specification.md"):
                shutil.copy2(ROOT / "specs" / filename, work / "specs" / filename)
            (work / "rules.json").write_text(
                json.dumps({"questions": [{"Kérdés?": ["Evaluate on a scale of 1 to 5.", 1, 5]}]}),
                encoding="utf-8",
            )
            with (work / "answers.csv").open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Kérdés?"])
                writer.writerow(["Ez egy tesztválasz."])
            (work / "must-not-read.key").write_text("not-a-real-key", encoding="utf-8")
            environment = dict(os.environ)
            environment.pop("GEMINI_API_KEY", None)
            environment.update({
                "LLM_PROVIDER": "local",
                "LOCAL_LLM_BASE_URL": self.base_url,
                "LOCAL_LLM_MODEL": "test-model",
                "LOCAL_LLM_JSON_MODE": "json_object",
                "LOCAL_LLM_TIMEOUT": "5",
                "LOCAL_LLM_MAX_ATTEMPTS": "1",
                "LOCAL_LLM_MAX_REFLECTIONS": "0",
                "LOCAL_LLM_API_KEY": "",
                "HTTP_PROXY": "http://127.0.0.1:1",
                "HTTPS_PROXY": "http://127.0.0.1:1",
                "NO_PROXY": "",
            })
            command = [sys.executable, str(ROOT / "app.py")]
            expected_model = "test-model"
            if config_name:
                expected_model = "configured-model"
                (work / config_name).write_text(json.dumps({
                    "provider": "local",
                    "providers": {"local": {
                        "base_url": self.base_url, "model": expected_model,
                    }},
                    "settings": {"max_reflections": 0},
                }), encoding="utf-8")
                # No Gemini key is needed even if the old environment selects Gemini.
                environment["LLM_PROVIDER"] = "gemini"
                if config_name != "llm_config.json":
                    command.extend(["--config", str(work / config_name)])
            run = subprocess.run(
                command, cwd=work, env=environment,
                text=True, capture_output=True, timeout=30,
            )
            self.assertEqual(run.returncode, 0, run.stderr)
            with (work / "scored_answers.csv").open(encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 1)
            self.assertEqual(float(rows[0]["score"]), 3)
            self.assertEqual(float(rows[0]["confidence"]), 0.8)
            self.assertEqual(rows[0]["reasoning"], "A rubrika alapján.")
            self.assertNotIn("Loaded GEMINI_API_KEY", run.stderr)
            self.assertEqual(self.server.received[-1][2]["model"], expected_model)

    def test_cloud_compatible_provider_payloads(self):
        for provider in ("groq_free", "mistral_free", "openrouter"):
            with self.subTest(provider=provider):
                model = "openrouter/free" if provider == "openrouter" else "test-model"
                client = LocalLLMClient(LocalLLMConfig(
                    base_url=self.base_url, model=model, provider=provider, api_key="test-only",
                ))
                client.complete([{"role": "user", "content": "test"}])
                _, headers, payload = self.server.received[-1]
                self.assertEqual(headers["Authorization"], "Bearer test-only")
                self.assertEqual(payload["model"], model)
                if provider == "openrouter":
                    self.assertEqual(payload["provider"], {
                        "max_price": {"prompt": 0, "completion": 0}, "allow_fallbacks": False,
                    })
                else:
                    self.assertNotIn("provider", payload)

    def test_request_cap_saves_partial_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / "specs").mkdir()
            for name in ("context.md", "specification.md"):
                shutil.copy2(ROOT / "specs" / name, work / "specs" / name)
            (work / "input.csv").write_text("Question\nAnswer1\nAnswer2\n", encoding="utf-8")
            (work / "rules.json").write_text(json.dumps({
                "questions": [{"Question": ["Score the answer.", 1, 5]}]
            }), encoding="utf-8")
            (work / "llm_config.json").write_text(json.dumps({
                "provider": "local", "input_csv": "input.csv", "rules_json": "rules.json",
                "settings": {"max_requests_per_run": 1},
                "providers": {"local": {"base_url": self.base_url}},
            }), encoding="utf-8")
            run = subprocess.run(
                [sys.executable, str(ROOT / "app.py")], cwd=work,
                capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(run.returncode, 2, run.stderr)
            with (work / "scored_answers.csv").open(encoding="utf-8") as file:
                self.assertEqual(len(list(csv.DictReader(file))), 1)
            self.assertEqual(len(self.server.received), 1)

    def _run_two_stage(self, responses, secondary=True, request_cap=20):
        self.server.evaluation_responses = responses
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / "specs").mkdir()
            for name in ("context.md", "specification.md"):
                shutil.copy2(ROOT / "specs" / name, work / "specs" / name)
            (work / "input.csv").write_text("Question\nAnswer\n", encoding="utf-8")
            (work / "rules.json").write_text(json.dumps({
                "questions": [{"Question": ["FULL RUBRIC", 1, 5]}]
            }), encoding="utf-8")
            (work / "llm_config.json").write_text(json.dumps({
                "primary": {"provider": "local", "model": "first"},
                "secondary": {"provider": "local", "model": "second"} if secondary else None,
                "input_csv": "input.csv", "rules_json": "rules.json",
                "settings": {"max_attempts": 1, "max_requests_per_run": request_cap},
                "review": {"primary_weight": 3, "secondary_weight": 1},
                "providers": {"local": {"base_url": self.base_url}},
            }), encoding="utf-8")
            run = subprocess.run(
                [sys.executable, str(ROOT / "app.py")], cwd=work,
                capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(run.returncode, 2 if request_cap == 1 else 0, run.stderr)
            with (work / "scored_answers.csv").open(encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 1)
            return rows[0]

    def test_csv_records_weighted_score_and_normalized_disagreement(self):
        row = self._run_two_stage([
            {"score": 2, "confidence": 0.2, "reasoning": "Missing details."},
            {"score": 4, "confidence": 0.8, "reasoning": "Rubric evidence."},
        ])
        self.assertEqual(float(row["score"]), 2.5)
        self.assertEqual(float(row["d"]), 0.5)
        self.assertEqual(row["primary_score"], "2")
        self.assertEqual(row["secondary_score"], "4")
        self.assertEqual(row["confidence"], "")
        self.assertEqual(row["needs_review"], "True")
        self.assertEqual(row["respondent_row"], "1")
        self.assertEqual(row["second_step_reason"], "low_confidence")
        self.assertEqual([request[2]["model"] for request in self.server.received], ["first", "second"])

    def test_csv_technical_failure_has_no_fake_primary_or_mean(self):
        row = self._run_two_stage([
            {"score": 99, "confidence": 0.8, "reasoning": "Invalid scale."},
            {"score": 4, "confidence": 0.8, "reasoning": "Independent score."},
        ])
        self.assertEqual(row["score"], "4")
        self.assertEqual(row["primary_score"], "")
        self.assertEqual(row["d"], "")
        self.assertEqual(row["status"], "secondary_only")
        self.assertEqual(row["second_step_reason"], "technical_failure")

    def test_csv_both_failed_still_records_error_row(self):
        row = self._run_two_stage([
            {"score": 99, "confidence": 0.8, "reasoning": "Invalid scale."},
            {"score": 99, "confidence": 0.8, "reasoning": "Still invalid."},
        ])
        self.assertEqual(row["score"], "")
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["needs_review"], "True")

    def test_csv_budget_during_review_saves_valid_primary(self):
        row = self._run_two_stage([
            {"score": 2, "confidence": 0.2, "reasoning": "Uncertain."},
        ], request_cap=1)
        self.assertEqual(row["score"], "2")
        self.assertEqual(row["secondary_status"], "request_budget_exhausted")
        self.assertEqual(row["needs_review"], "True")
        self.assertEqual(len(self.server.received), 1)


if __name__ == "__main__":
    unittest.main()
