import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from llm_config import ReviewSettings, load_config, resolve_secondary_config
from local_llm import LocalLLMClient, LocalLLMConfig, RequestBudgetExceeded, SharedRequestBudget
from scoring_engine import TwoStageScorer, validate_scale


def evaluation(score=3, confidence=0.8, reasoning="Observable evidence in the answer."):
    return json.dumps({"score": score, "confidence": confidence, "reasoning": reasoning})


class FakeClient:
    def __init__(self, *outputs, model="primary"):
        self.config = LocalLLMConfig(model=model)
        self.outputs = list(outputs)
        self.calls = []
        self.request_count = 0

    def score(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        value = self.outputs.pop(0)
        if isinstance(value, Exception):
            raise value
        self.request_count += 1
        return value


class TwoStageScoringTests(unittest.TestCase):
    def evaluate(self, primary, secondary=None, policy=None):
        factory = Mock(return_value=secondary) if secondary is not None else None
        scorer = TwoStageScorer(primary, policy, factory, secondary.config if secondary else None)
        return scorer.evaluate("System", "Original question?", "FULL RUBRIC", 1, 5, "Original answer")

    def test_confident_primary_skips_reviewer(self):
        primary, secondary = FakeClient(evaluation(3, 0.8)), FakeClient(evaluation(5))
        result = self.evaluate(primary, secondary)
        self.assertEqual(result["score"], 3)
        self.assertEqual(result["status"], "primary_only")
        self.assertEqual(secondary.calls, [])
        self.assertIsNone(result["d"])
        self.assertFalse(result["needs_review"])
        self.assertFalse(primary.calls[0][1]["allow_reflection"])

    def test_low_confidence_two_valid_scores_use_configured_weights(self):
        primary = FakeClient(evaluation(2, 0.2))
        secondary = FakeClient(evaluation(4, 0.9), model="reviewer")
        result = self.evaluate(primary, secondary, ReviewSettings(primary_weight=3, secondary_weight=1))
        self.assertEqual(result["score"], 2.5)
        self.assertEqual(result["primary_score"], 2)
        self.assertEqual(result["secondary_score"], 4)
        self.assertEqual(result["primary_confidence"], 0.2)
        self.assertEqual(result["secondary_confidence"], 0.9)
        self.assertEqual(result["d"], 0.5)
        self.assertEqual(result["absolute_disagreement"], 2)
        self.assertTrue(result["needs_review"])
        self.assertIn("large_disagreement", result["review_reasons"])
        self.assertEqual(result["effective_primary_weight"], 0.75)
        self.assertIsNone(result["confidence"])
        self.assertEqual(result["confidence_source"], "not_defined_for_weighted_score")
        self.assertEqual(result["second_step_reason"], "low_confidence")

    def test_failed_reviewer_retains_primary_and_requires_review(self):
        result = self.evaluate(FakeClient(evaluation(2, 0.2)), FakeClient(None))
        self.assertEqual(result["score"], 2)
        self.assertEqual(result["confidence"], 0.2)
        self.assertEqual(result["status"], "primary_retained_review_failed")
        self.assertEqual(result["aggregation_method"], "primary_only")
        self.assertTrue(result["needs_review"])
        self.assertIsNone(result["d"])

    def test_technical_fallback_uses_only_secondary_score(self):
        primary, secondary = FakeClient(None), FakeClient(evaluation(4, 0.9))
        result = self.evaluate(primary, secondary)
        self.assertEqual(result["score"], 4)
        self.assertEqual(result["status"], "secondary_only")
        self.assertEqual(result["second_step_reason"], "technical_failure")
        self.assertIsNone(result["primary_score"])
        self.assertIsNone(result["d"])
        self.assertEqual(result["effective_primary_weight"], 0)
        self.assertEqual(result["effective_secondary_weight"], 1)
        self.assertIsNone(secondary.calls[0][1]["review_context"]["primary_evaluation"])

    def test_both_fail_no_final_score(self):
        result = self.evaluate(FakeClient(None), FakeClient(None))
        self.assertIsNone(result["score"])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["aggregation_method"], "none")
        self.assertTrue(result["needs_review"])

    def test_disagreement_threshold_inclusive_and_mean_retained(self):
        result = self.evaluate(
            FakeClient(evaluation(2, 0.1)), FakeClient(evaluation(3, 0.9)),
            ReviewSettings(disagreement_threshold=0.25),
        )
        self.assertEqual(result["d"], 0.25)
        self.assertEqual(result["score"], 2.5)
        self.assertTrue(result["needs_review"])

    def test_review_agreement_and_confident_reviewer(self):
        result = self.evaluate(FakeClient(evaluation(3, 0.2)), FakeClient(evaluation(3, 0.9)))
        self.assertEqual(result["d"], 0)
        self.assertFalse(result["needs_review"])

    def test_second_evaluation_can_remain_uncertain(self):
        result = self.evaluate(FakeClient(evaluation(3, 0.2)), FakeClient(evaluation(3, 0.1)))
        self.assertEqual(result["secondary_confidence"], 0.1)
        self.assertTrue(result["needs_review"])
        self.assertIn("low_secondary_confidence", result["review_reasons"])

    def test_missing_secondary_reuses_primary_in_new_call(self):
        primary = FakeClient(evaluation(2, 0.2), evaluation(3, 0.3))
        result = self.evaluate(primary)
        self.assertEqual(len(primary.calls), 2)
        self.assertEqual(result["secondary_mode"], "same_model_review")
        self.assertEqual(result["score"], 2.5)
        self.assertEqual(primary.calls[1][1]["review_context"]["reason"], "low_confidence")

    def test_missing_secondary_retries_primary_after_technical_failure(self):
        primary = FakeClient(None, evaluation(4, 0.8))
        result = self.evaluate(primary)
        self.assertEqual(result["score"], 4)
        self.assertEqual(result["secondary_mode"], "same_model_review")
        self.assertEqual(result["second_step_reason"], "technical_failure")
        self.assertEqual(len(primary.calls), 2)

    def test_secondary_receives_original_inputs_and_written_primary_evaluation(self):
        primary, secondary = FakeClient(evaluation(2, 0.2, "Missing stated evidence")), FakeClient(evaluation())
        self.evaluate(primary, secondary)
        args, kwargs = secondary.calls[0]
        self.assertEqual(args[1:], ("Original question?", "FULL RUBRIC", 1, 5, "Original answer"))
        self.assertEqual(kwargs["review_context"]["primary_evaluation"]["reasoning"], "Missing stated evidence")
        self.assertFalse(kwargs["allow_reflection"])

    def test_exact_confidence_threshold_does_not_trigger_second_stage(self):
        primary, secondary = FakeClient(evaluation(3, 0.5)), FakeClient(None)
        self.evaluate(primary, secondary)
        self.assertFalse(secondary.calls)

    def test_custom_confidence_threshold(self):
        result = self.evaluate(
            FakeClient(evaluation(3, 0.7)), FakeClient(evaluation(3, 0.9)),
            ReviewSettings(confidence_threshold=0.8),
        )
        self.assertEqual(result["status"], "weighted_review")

    def test_invalid_primary_score_never_enters_mean(self):
        result = self.evaluate(FakeClient(evaluation(99)), FakeClient(evaluation(4)))
        self.assertEqual(result["score"], 4)
        self.assertIsNone(result["primary_score"])
        self.assertEqual(result["aggregation_method"], "secondary_only")

    def test_invalid_secondary_score_never_enters_mean(self):
        result = self.evaluate(FakeClient(evaluation(2, 0.2)), FakeClient(evaluation(99)))
        self.assertEqual(result["score"], 2)
        self.assertIsNone(result["secondary_score"])

    def test_secondary_configuration_failure_does_not_block_primary(self):
        primary = FakeClient(evaluation(3, 0.8))
        factory = Mock(side_effect=ValueError("Missing secondary key"))
        scorer = TwoStageScorer(primary, secondary_factory=factory)
        self.assertEqual(scorer.evaluate("s", "q", "r", 1, 5, "a")["score"], 3)
        factory.assert_not_called()

    def test_secondary_configuration_failure_retains_valid_primary(self):
        factory = Mock(side_effect=ValueError("Missing secondary key"))
        scorer = TwoStageScorer(FakeClient(evaluation(2, 0.2)), secondary_factory=factory)
        result = scorer.evaluate("s", "q", "r", 1, 5, "a")
        self.assertEqual(result["score"], 2)
        self.assertEqual(result["secondary_status"], "configuration_error")
        self.assertTrue(result["needs_review"])

    def test_zero_primary_weight_is_explicitly_supported(self):
        result = self.evaluate(
            FakeClient(evaluation(2, 0.2)), FakeClient(evaluation(4)),
            ReviewSettings(primary_weight=0, secondary_weight=1),
        )
        self.assertEqual(result["score"], 4)
        self.assertEqual(result["d"], 0.5)

    def test_very_large_finite_weights_do_not_overflow(self):
        weights = ReviewSettings(primary_weight=1e308, secondary_weight=1e308)
        self.assertEqual(weights.normalized_weights, (0.5, 0.5))

    def test_invalid_review_settings(self):
        for kwargs in (
            {"primary_weight": 0, "secondary_weight": 0}, {"primary_weight": -1},
            {"secondary_weight": float("nan")}, {"confidence_threshold": 1.1},
            {"disagreement_threshold": -1}, {"primary_weight": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ReviewSettings(**kwargs)

    def test_invalid_scales_are_rejected(self):
        for lo, hi in ((1, 1), (2, 1), (True, 5), ("1", 5), (0, float("inf")), (-1e308, 1e308)):
            with self.subTest(scale=(lo, hi)), self.assertRaises(ValueError):
                validate_scale(lo, hi)

    def test_shared_budget_exhausted_before_second_stage_preserves_primary(self):
        client = LocalLLMClient(LocalLLMConfig(max_attempts=1), SharedRequestBudget(1))
        def complete(messages):
            client._before_request()
            return evaluation(2, 0.2)
        with patch.object(client, "complete", side_effect=complete):
            result = self.evaluate(client)
        self.assertEqual(result["score"], 2)
        self.assertTrue(result["stop_requested"])
        self.assertTrue(result["needs_review"])
        self.assertEqual(result["secondary_status"], "request_budget_exhausted")

    def test_shared_budget_applies_across_distinct_clients(self):
        budget = SharedRequestBudget(1)
        primary = LocalLLMClient(LocalLLMConfig(), budget)
        secondary = LocalLLMClient(LocalLLMConfig(model="reviewer"), budget)
        primary._before_request()
        with self.assertRaises(RequestBudgetExceeded):
            secondary._before_request()
        self.assertEqual(budget.used, 1)

    def test_review_prompt_has_no_hidden_reasoning_claim_and_preserves_uncertainty(self):
        client = LocalLLMClient(LocalLLMConfig())
        with patch.object(client, "complete", return_value=evaluation(2, 0.1)) as call:
            raw = client.score(
                "System", "Question", "COMPLETE RUBRIC", 1, 5, "Answer",
                allow_reflection=False,
                review_context={"reason": "low_confidence", "primary_evaluation": json.loads(evaluation(2, 0.2))},
            )
        self.assertEqual(call.call_count, 1)
        messages = call.call_args.args[0]
        self.assertIn("cannot access the first model's hidden", messages[0]["content"])
        self.assertIn("Do not force a higher confidence", messages[0]["content"])
        self.assertIn("COMPLETE RUBRIC", messages[1]["content"])
        self.assertEqual(json.loads(raw)["confidence"], 0.1)


class TwoRoleConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.path = self.directory / "llm_config.json"
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def load(self, data):
        self.path.write_text(json.dumps(data), encoding="utf-8")
        return load_config(self.path)

    def test_primary_and_secondary_can_use_different_models_on_same_provider(self):
        config = self.load({
            "primary": {"provider": "local", "model": "first"},
            "secondary": {"provider": "local", "model": "second"},
        })
        self.assertEqual(config.llm.model, "first")
        self.assertEqual(config.secondary_llm.model, "second")

    def test_secondary_credentials_are_lazy(self):
        config = self.load({"primary": {"provider": "local"}, "secondary": {"provider": "groq_free"}})
        self.assertEqual(config.llm.provider, "local")
        with self.assertRaises(ValueError):
            resolve_secondary_config(config)

    def test_secondary_secret_can_be_resolved_on_demand(self):
        config = self.load({"primary": {"provider": "local"}, "secondary": {"provider": "groq_free"}})
        getter = Mock(return_value="test-secondary-key")
        self.assertEqual(resolve_secondary_config(config, secret_getter=getter).api_key, "test-secondary-key")
        getter.assert_called_once_with("GROQ_API_KEY")

    def test_secondary_null_uses_primary(self):
        config = self.load({"primary": {"provider": "local"}, "secondary": None})
        self.assertIsNone(config.secondary_llm)
        self.assertIs(resolve_secondary_config(config), config.llm)

    def test_legacy_provider_configuration_is_still_supported(self):
        self.assertEqual(self.load({"provider": "local"}).llm.provider, "local")

    def test_ambiguous_primary_selection_is_rejected(self):
        with self.assertRaises(ValueError):
            self.load({"provider": "local", "primary": {"provider": "local"}})

    def test_paid_secondary_openrouter_model_is_rejected(self):
        with self.assertRaises(ValueError):
            self.load({"primary": {"provider": "local"}, "secondary": {
                "provider": "openrouter", "model": "paid/model",
            }})

    def test_invalid_roles_and_review_options_are_rejected(self):
        for data in (
            {"primary": "local"}, {"secondary": "groq_free"}, {"secondary": {}},
            {"review": {"confidence_threshold": "0.5"}},
            {"review": {"primary_weight": 0, "secondary_weight": 0}},
            {"secondary": {"provider": "groq_free", "base_url": "http://other"}},
        ):
            with self.subTest(data=data), self.assertRaises(ValueError):
                self.load(data)

    def test_distinct_key_files_need_distinct_environment_names(self):
        with self.assertRaises(ValueError):
            self.load({
                "primary": {"provider": "local", "api_key_file": "one.key"},
                "secondary": {"provider": "local", "api_key_file": "two.key"},
            })


if __name__ == "__main__":
    unittest.main()
