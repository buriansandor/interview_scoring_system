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


ROOT = Path(__file__).resolve().parents[1]


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
        content = json.dumps({"score": 3, "confidence": 0.8, "reasoning": "A rubrika alapján."})
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
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            for filename in ("context.md", "specification.md"):
                shutil.copy2(ROOT / filename, work / filename)
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
            run = subprocess.run(
                [sys.executable, str(ROOT / "app.py")], cwd=work, env=environment,
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
            self.assertEqual(self.server.received[-1][2]["model"], "test-model")


if __name__ == "__main__":
    unittest.main()
