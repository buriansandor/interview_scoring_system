import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from llm_config import load_config, PROVIDERS
from local_llm import LocalLLMClient, LocalLLMConfig, RequestBudgetExceeded


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.path = self.directory / "llm_config.json"
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def load(self, data):
        self.path.write_text(json.dumps(data), encoding="utf-8")
        return load_config(self.path)

    def test_minimal_local_config(self):
        config = self.load({"provider": "local"})
        self.assertEqual(config.llm.provider, "local")
        self.assertEqual(config.llm.model, "interview-scorer")
        self.assertEqual(config.llm.api_key, "")
        self.assertEqual(config.llm.max_requests_per_run, 50)

    def test_each_cloud_provider(self):
        for provider in ("gemini", "mistral_free", "groq_free", "openrouter"):
            with self.subTest(provider=provider):
                os.environ[PROVIDERS[provider]["api_key_env"]] = "test-only"
                config = self.load({"provider": provider})
                self.assertEqual(config.llm.provider, provider)
                self.assertEqual(config.llm.api_key, "test-only")
                self.assertEqual(config.llm.base_url, PROVIDERS[provider]["base_url"])

    def test_key_file_relative_to_config(self):
        (self.directory / "groq.key").write_text(" file-test-key\n", encoding="utf-8")
        self.assertEqual(self.load({"provider": "groq_free"}).llm.api_key, "file-test-key")

    def test_environment_key_has_priority(self):
        (self.directory / "groq.key").write_text("file-key", encoding="utf-8")
        os.environ["GROQ_API_KEY"] = "env-key"
        self.assertEqual(self.load({"provider": "groq_free"}).llm.api_key, "env-key")

    def test_unselected_keys_are_not_required(self):
        config = self.load({"provider": "local", "providers": {
            name: {"api_key_file": f"{name}.key"} for name in PROVIDERS
        }})
        self.assertEqual(config.llm.provider, "local")

    def test_does_not_use_another_provider_key(self):
        (self.directory / "gemini.key").write_text("wrong-provider-key", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Missing API key"):
            self.load({"provider": "groq_free"})

    def test_inline_key_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown"):
            self.load({"providers": {"local": {"api_key": "do-not-log"}}})

    def test_config_wins_over_legacy_environment(self):
        os.environ["LLM_PROVIDER"] = "gemini"
        os.environ["LOCAL_LLM_MODEL"] = "ignored-model"
        self.assertEqual(self.load({"provider": "local"}).llm.model, "interview-scorer")

    def test_relative_input_and_output_paths(self):
        config = self.load({"input_csv": "data/answers.csv", "rules_json": "rules.json",
                            "output_csv": "result.csv"})
        self.assertEqual(config.input_csv, self.directory / "data/answers.csv")
        self.assertEqual(config.rules_json, self.directory / "rules.json")
        self.assertEqual(config.output_csv, self.directory / "result.csv")

    def test_input_cannot_be_overwritten(self):
        with self.assertRaises(ValueError):
            self.load({"input_csv": "same.csv", "output_csv": "same.csv"})

    def test_output_must_not_be_null(self):
        with self.assertRaises(ValueError):
            self.load({"output_csv": None})

    def test_output_cannot_overwrite_key(self):
        with self.assertRaises(ValueError):
            self.load({"output_csv": "secret.csv",
                       "providers": {"local": {"api_key_file": "secret.csv"}}})

    def test_unknown_fields_and_invalid_types(self):
        values = [
            {"provider": "groqq"}, {"provider": []}, {"providerr": "local"},
            {"settings": {"max_attempts": True}}, {"settings": {"max_tokens": "512"}},
            {"settings": {"max_requests_per_run": 0}},
            {"settings": {"temperature": float("nan")}},
            {"providers": {"local": {"json_mode": "invalid"}}},
            {"providers": {"local": {"model": 123}}},
            {"providers": {"local": {"api_key_file": 12}}},
            {"providers": {"groq_free": {"base_url": "http://untrusted"}}},
            {"providers": {"local": {"requests_per_minute": -1}}},
        ]
        for value in values:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.load(value)

    def test_paid_openrouter_model_rejected(self):
        os.environ["OPENROUTER_API_KEY"] = "test"
        with self.assertRaisesRegex(ValueError, "only accepts"):
            self.load({"provider": "openrouter", "providers": {
                "openrouter": {"model": "some/paid-model"}
            }})

    def test_explicit_free_openrouter_model(self):
        os.environ["OPENROUTER_API_KEY"] = "test"
        config = self.load({"provider": "openrouter", "providers": {
            "openrouter": {"model": "some/model:free"}
        }})
        self.assertEqual(config.llm.model, "some/model:free")

    def test_cloud_url_cannot_be_changed(self):
        with self.assertRaises(ValueError):
            self.load({"provider": "local", "providers": {
                "openrouter": {"base_url": "http://wrong"}
            }})

    def test_explicit_missing_config_fails(self):
        with self.assertRaises(ValueError):
            load_config(self.path, require_file=True)

    def test_legacy_local_environment(self):
        os.environ["LLM_PROVIDER"] = "local"
        os.environ["LOCAL_LLM_MODEL"] = "legacy-model"
        self.assertEqual(load_config(self.path).llm.model, "legacy-model")

    def test_legacy_gemini_never_uses_groq_key(self):
        (self.directory / "groq.key").write_text("other-provider-key", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_config(self.path)

    def test_legacy_gemini_named_key(self):
        (self.directory / "gemini.key").write_text("legacy-gemini-key", encoding="utf-8")
        self.assertEqual(load_config(self.path).llm.api_key, "legacy-gemini-key")

    def test_api_key_not_in_repr(self):
        self.assertNotIn("sensitive-value", repr(LocalLLMConfig(api_key="sensitive-value")))

    def test_request_budget_counts_failures(self):
        client = LocalLLMClient(LocalLLMConfig(max_requests_per_run=1))
        client._before_request()
        with self.assertRaises(RequestBudgetExceeded):
            client._before_request()
        self.assertEqual(client.request_count, 1)

    def test_rate_limit(self):
        client = LocalLLMClient(LocalLLMConfig(requests_per_minute=10))
        with patch("local_llm.time.monotonic", side_effect=[100, 101, 106]), patch("local_llm.time.sleep") as sleep:
            client._before_request()
            client._before_request()
        sleep.assert_called_once_with(5)

    def test_budget_includes_reflection_and_does_not_switch_provider(self):
        client = LocalLLMClient(LocalLLMConfig(max_requests_per_run=1))
        def complete(messages):
            client._before_request()
            return json.dumps({"score": 3, "confidence": 0.2, "reasoning": "Uncertain"})
        with patch.object(client, "complete", side_effect=complete):
            with self.assertRaises(RequestBudgetExceeded):
                client.score("system", "question", "rules", 1, 5, "answer")


class GeminiAdapterTests(unittest.TestCase):
    def test_gemini_uses_shared_scoring_and_sdk_json_schema(self):
        from llm_providers import build_client, GeminiScoringClient
        config = LocalLLMConfig(
            provider="gemini", model="gemini-2.5-flash", api_key="test-only",
            json_mode="json_schema", max_reflections=0,
        )
        with patch("google.genai.Client") as sdk:
            sdk.return_value.models.generate_content.return_value.text = json.dumps({
                "score": 4, "confidence": 0.9, "reasoning": "Megfelel."
            })
            client = build_client(config)
            value = json.loads(client.score("system", "question", "rules", 1, 5, "answer"))
            self.assertIsInstance(client, GeminiScoringClient)
            self.assertEqual(value["score"], 4)
            call = sdk.return_value.models.generate_content.call_args
            self.assertEqual(call.kwargs["model"], "gemini-2.5-flash")
            self.assertEqual(call.kwargs["config"].response_mime_type, "application/json")
            self.assertEqual(call.kwargs["config"].response_json_schema["type"], "object")
            self.assertEqual(client.request_count, 1)
            self.assertFalse(sdk.call_args.kwargs["vertexai"])
            self.assertEqual(sdk.call_args.kwargs["http_options"]["retry_options"]["attempts"], 1)


if __name__ == "__main__":
    unittest.main()
