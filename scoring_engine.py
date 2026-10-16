"""Two-stage scoring: technical fallback and uncertainty review are distinct."""

import logging
import math

from llm_config import ReviewSettings, resolve_secondary_config
from llm_providers import build_client
from local_llm import RequestBudgetExceeded, SharedRequestBudget, validate_evaluation

logger = logging.getLogger("agent")

AUDIT_COLUMNS = [
    "status", "aggregation_method", "needs_review", "review_reasons", "second_step_reason", "secondary_mode",
    "primary_provider", "primary_model", "primary_status", "primary_score",
    "primary_confidence", "primary_reasoning", "primary_attempts",
    "secondary_provider", "secondary_model", "secondary_status", "secondary_score",
    "secondary_confidence", "secondary_reasoning", "secondary_attempts",
    "primary_weight", "secondary_weight", "effective_primary_weight", "effective_secondary_weight",
    "absolute_disagreement", "d", "range_min", "range_max",
    "confidence_threshold", "disagreement_threshold", "confidence_source",
    "failure_reason", "stop_requested",
]


def validate_scale(range_min, range_max):
    for value in (range_min, range_max):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("Scoring bounds must be finite numbers.")
    width = range_max - range_min
    if not math.isfinite(width) or width <= 0:
        raise ValueError("range_max must exceed range_min by a finite positive amount.")
    return width


class TwoStageScorer:
    def __init__(self, primary, review=None, secondary_factory=None, secondary_metadata=None):
        self.primary = primary
        self.review = review or ReviewSettings()
        self.secondary_factory = secondary_factory
        self.secondary_metadata = secondary_metadata
        self._secondary = None

    def _stage(self, client, args, context=None):
        start = client.request_count
        try:
            raw = client.score(*args, allow_reflection=False, review_context=context)
            result = validate_evaluation(raw, args[3], args[4]) if raw is not None else None
            return result, "valid" if result is not None else "failed", client.request_count - start
        except RequestBudgetExceeded:
            return None, "request_budget_exhausted", client.request_count - start
        except (ValueError, TypeError, IndexError):
            return None, "failed", client.request_count - start

    def evaluate(self, system_prompt, question, rules, range_min, range_max, answer):
        width = validate_scale(range_min, range_max)
        args = (system_prompt, question, rules, range_min, range_max, answer)
        policy = self.review
        record = dict.fromkeys(AUDIT_COLUMNS)
        record.update(
            score=None, confidence=None, reasoning="No valid evaluation.",
            status="failed", aggregation_method="none", needs_review=True,
            review_reasons="no_valid_score",
            second_step_reason="", secondary_mode="", primary_status="not_attempted",
            secondary_status="not_attempted", primary_attempts=0, secondary_attempts=0,
            primary_provider=self.primary.config.provider, primary_model=self.primary.config.model,
            primary_weight=policy.primary_weight, secondary_weight=policy.secondary_weight,
            effective_primary_weight=0.0, effective_secondary_weight=0.0,
            range_min=range_min, range_max=range_max,
            confidence_threshold=policy.confidence_threshold,
            disagreement_threshold=policy.disagreement_threshold,
            confidence_source="none", failure_reason="", stop_requested=False,
        )
        primary, primary_status, attempts = self._stage(self.primary, args)
        record.update(primary_status=primary_status, primary_attempts=attempts)
        if primary_status == "request_budget_exhausted":
            if attempts == 0:
                # No evaluation started: preserve the established partial-save behavior.
                raise RequestBudgetExceeded("Shared request budget exhausted before evaluation.")
            record.update(
                failure_reason="primary_request_budget_exhausted",
                review_reasons="request_budget_exhausted", stop_requested=True,
            )
            return record
        if primary is not None:
            self._store_evaluation(record, "primary", primary)
            self._use_primary(record, primary)
            if primary["confidence"] >= policy.confidence_threshold:
                record.update(status="primary_only", needs_review=False, review_reasons="")
                return record

        reason = "low_confidence" if primary is not None else "technical_failure"
        record["second_step_reason"] = reason
        record["secondary_mode"] = "configured_secondary" if self.secondary_factory else "same_model_review"
        metadata = self.secondary_metadata or self.primary.config
        record.update(secondary_provider=metadata.provider, secondary_model=metadata.model)
        try:
            if self.secondary_factory is None:
                secondary_client = self.primary
            else:
                if self._secondary is None:
                    self._secondary = self.secondary_factory()
                secondary_client = self._secondary
            secondary, status, attempts = self._stage(
                secondary_client, args,
                {"reason": reason, "primary_evaluation": primary},
            )
        except (ValueError, OSError):
            # A missing reviewer credential must not prevent the primary attempt.
            logger.warning("Configured secondary could not be initialized; no other provider attempted.")
            secondary, status, attempts = None, "configuration_error", 0
        record.update(secondary_status=status, secondary_attempts=attempts)
        if secondary is None:
            record.update(
                status="primary_retained_review_failed" if primary is not None else "failed",
                needs_review=True, failure_reason=f"secondary_{status}",
                review_reasons=f"secondary_{status}",
                stop_requested=status == "request_budget_exhausted",
            )
            return record

        self._store_evaluation(record, "secondary", secondary)
        if primary is None:
            record.update(
                score=secondary["score"], confidence=secondary["confidence"],
                reasoning=secondary["reasoning"], status="secondary_only",
                aggregation_method="secondary_only", confidence_source="secondary",
                effective_secondary_weight=1.0,
                needs_review=secondary["confidence"] < policy.confidence_threshold,
                review_reasons=(
                    "low_secondary_confidence"
                    if secondary["confidence"] < policy.confidence_threshold else ""
                ),
                failure_reason="primary_failed",
            )
            return record

        primary_weight, secondary_weight = policy.normalized_weights
        difference = abs(primary["score"] - secondary["score"])
        review_reasons = []
        if secondary["confidence"] < policy.confidence_threshold:
            review_reasons.append("low_secondary_confidence")
        if difference / width >= policy.disagreement_threshold:
            review_reasons.append("large_disagreement")
        record.update(
            score=primary["score"] + (secondary["score"] - primary["score"]) * secondary_weight,
            confidence=None, confidence_source="not_defined_for_weighted_score",
            reasoning="Weighted mean of two valid evaluations; see both recorded reasonings.",
            status="weighted_review", aggregation_method="weighted_average",
            effective_primary_weight=primary_weight, effective_secondary_weight=secondary_weight,
            absolute_disagreement=difference, d=difference / width,
            needs_review=bool(review_reasons), review_reasons=";".join(review_reasons),
        )
        return record

    @staticmethod
    def _store_evaluation(record, prefix, evaluation):
        for key in ("score", "confidence", "reasoning"):
            record[f"{prefix}_{key}"] = evaluation[key]

    @staticmethod
    def _use_primary(record, evaluation):
        record.update(
            score=evaluation["score"], confidence=evaluation["confidence"],
            reasoning=evaluation["reasoning"], aggregation_method="primary_only",
            confidence_source="primary", effective_primary_weight=1.0,
        )


def build_scorer(config):
    budget = SharedRequestBudget(config.llm.max_requests_per_run)
    primary = build_client(config.llm, request_budget=budget)
    secondary_factory = None
    if config.secondary_llm is not None:
        secondary_factory = lambda: build_client(resolve_secondary_config(config), request_budget=budget)
    return TwoStageScorer(
        primary, config.review, secondary_factory=secondary_factory,
        secondary_metadata=config.secondary_llm,
    )
