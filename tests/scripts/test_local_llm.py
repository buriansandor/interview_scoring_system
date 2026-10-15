import json
import os
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from local_llm import LocalLLMClient, LocalLLMConfig, validate_evaluation


def result(score=3, confidence=0.8):
    return json.dumps({"score": score, "confidence": confidence, "reasoning": "A rubrika szerint."})


class LocalLLMTests(unittest.TestCase):
    def client(self, **kwargs):
        return LocalLLMClient(LocalLLMConfig(**kwargs))

    def run_score(self, client):
        return client.score("System instructions", "Kérdés?", "Rubrika", 1, 5, "Válasz.")

    def test_valid_result(self):
        self.assertEqual(validate_evaluation(result(), 1, 5)["score"], 3)

    def test_reject_invalid_scores(self):
        for score in (True, "3", 0, 6, float("nan"), float("inf")):
            with self.subTest(score=score), self.assertRaises(ValueError):
                validate_evaluation(result(score=score), 1, 5)

    def test_reject_invalid_confidence(self):
        for confidence in (True, "0.8", -0.1, 1.1, float("nan")):
            with self.subTest(confidence=confidence), self.assertRaises(ValueError):
                validate_evaluation(result(confidence=confidence), 1, 5)

    def test_reject_missing_extra_empty_fields(self):
        for text in ('{}', '[]', '{"score":3,"confidence":0.8,"reasoning":""}',
                     '{"score":3,"confidence":0.8,"reasoning":"ok","extra":1}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                validate_evaluation(text, 1, 5)

    def test_fenced_json(self):
        self.assertEqual(validate_evaluation("```json\n" + result() + "\n```", 1, 5)["score"], 3)

    def test_environment(self):
        with patch.dict(os.environ, {"LOCAL_LLM_BASE_URL": "http://localhost:11434/v1",
                                     "LOCAL_LLM_MODEL": "my-model"}, clear=True):
            config = LocalLLMConfig.from_env()
        self.assertEqual(config.model, "my-model")
        self.assertEqual(config.base_url, "http://localhost:11434/v1")
        self.assertEqual(config.api_key, "")

    def test_invalid_configuration(self):
        for kwargs in ({"base_url": "file:///tmp"}, {"base_url": "http://a/?key=x"},
                       {"base_url": "http://user:secret@localhost"}, {"model": ""},
                       {"timeout": float("nan")}, {"max_attempts": 0},
                       {"max_reflections": -1}, {"json_mode": "other"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                LocalLLMConfig(**kwargs)

    def test_local_success(self):
        client = self.client()
        with patch.object(client, "complete", return_value=result()) as call:
            self.assertEqual(json.loads(self.run_score(client))["score"], 3)
        self.assertEqual(call.call_count, 1)
        self.assertIn("No tools are available", call.call_args.args[0][0]["content"])

    def test_validation_retry(self):
        client = self.client()
        with patch.object(client, "complete", side_effect=[result(score=10), result()]) as call:
            self.assertEqual(json.loads(self.run_score(client))["score"], 3)
        self.assertEqual(call.call_count, 2)

    def test_low_confidence_is_preserved(self):
        client = self.client()
        with patch.object(client, "complete", return_value=result(confidence=0.2)) as call:
            self.assertEqual(json.loads(self.run_score(client))["confidence"], 0.2)
        self.assertEqual(call.call_count, 2)

    def test_total_attempt_bound(self):
        client = self.client(max_attempts=2)
        with patch.object(client, "complete", return_value="bad JSON") as call:
            self.assertIsNone(self.run_score(client))
        self.assertEqual(call.call_count, 2)

    def test_network_failure_stays_local(self):
        client = self.client(max_attempts=2)
        with patch.object(client, "complete", side_effect=URLError("offline")) as call, patch("local_llm.time.sleep"):
            self.assertIsNone(self.run_score(client))
        self.assertEqual(call.call_count, 2)

    def test_permanent_http_error_does_not_retry(self):
        client = self.client()
        error = HTTPError("http://localhost/v1", 401, "Unauthorized", {}, None)
        with patch.object(client, "complete", side_effect=error) as call:
            self.assertIsNone(self.run_score(client))
        self.assertEqual(call.call_count, 1)

    def test_reflection_failure_keeps_valid_result(self):
        client = self.client(max_attempts=2)
        with patch.object(client, "complete", side_effect=[result(confidence=0.2), "bad"]):
            self.assertEqual(json.loads(self.run_score(client))["confidence"], 0.2)


if __name__ == "__main__":
    unittest.main()
