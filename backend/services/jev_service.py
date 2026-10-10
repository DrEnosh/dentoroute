"""TypeSafe Jev integration. ALL Jev-specific code lives in this file so it is easy to change.

Built on the official `typesafe-sdk` (https://docs.typesafe.ai/sdk/python):

    client.system_one(state, questions)  ->  response.answers[<id>]
        Choice -> choice, probabilities, confidence
        Score  -> score (0..n-1), probabilities (keyed by level), confidence
        Noul   -> noul (probability of YES, 0..1; no separate confidence)

Design (docs: "ask atomic questions, compose in code"): seven narrow questions are sent in ONE request;
they are evaluated in parallel and independently. Jev only produces decisions - routing/business logic
lives in routing_service.py (deterministic Python).
"""
from __future__ import annotations

import logging
import math
import time
from typing import Any

from backend.repositories import cache_repo
from backend.schemas.case import CaseState
from backend.schemas.decision import ChoiceDecision, JevDecision, NoulDecision, ScoreDecision
from backend.schemas.labels import COMPLEXITY_LEVELS, REFERRAL_LEVELS, SPECIALTIES, URGENCY_LEVELS
from backend.services import demo_simulator
from backend.utils.config import Settings, get_settings
from backend.utils.errors import (
    InvalidModelOutputError, JevServiceError, MissingApiKeyError,
)
from backend.utils.hashing import stable_hash
from backend.utils.logging_utils import get_logger, log_event

logger = get_logger("jev")

# Bump when question text/levels change: invalidates cached responses.
QUESTION_SET_VERSION = "1.0"
PROB_SUM_TOLERANCE = 0.02

DISCLAIMER_PREFIX = (
    "Context: synthetic dental case in a research prototype. This is a routing estimate, not a diagnosis. "
)

# --------------------------------------------------------------------------
# Question definitions: single source of truth (also shown in Research Mode)
# --------------------------------------------------------------------------
QUESTION_SPECS: dict[str, dict[str, Any]] = {
    "urgency": {
        "type": "score",
        "labels": URGENCY_LEVELS,
        "instructions": (
            "Based only on the structured case information, how urgently does this dental case need to be "
            "seen by a dental professional? Judge routing urgency, not a diagnosis."
        ),
        "criteria": [
            "ROUTINE: no pain or only mild, brief discomfort; no swelling, fever, trauma, bleeding or spreading "
            "signs; an ordinary scheduled appointment is appropriate.",
            "PRIORITY: moderate pain or a concerning but stable finding (for example localized swelling, a "
            "fractured tooth, a persistent mouth lesion, or pain after a procedure); should be seen within "
            "days, with no systemic or rapidly worsening signs.",
            "URGENT: severe pain or acute findings that call for being seen within about a day, such as "
            "swelling together with fever or progression, a tooth knocked out, or severe trauma.",
            "EMERGENCY: signs that may threaten the airway or point to rapidly spreading infection, such as "
            "difficulty breathing or swallowing, or swelling with fever and restricted mouth opening.",
        ],
    },
    "specialty": {
        "type": "choice",
        "labels": SPECIALTIES,
        "instructions": (
            "Based only on the structured case information, which dental service is the best routing "
            "destination for this case? Choose exactly one."
        ),
        "criteria": {
            "general_dentistry": "Routine examination, early cavities, simple restorations, administrative requests, "
                                 "or cases that do not clearly belong to a specialty.",
            "endodontics": "Tooth-pulp or root-related pain patterns: lingering, spontaneous or throbbing tooth pain, "
                           "pain on biting, problems after a root canal, painful fractured tooth.",
            "periodontics": "Gum and supporting-tissue problems: bleeding or receding gums, loose teeth from gum "
                            "disease, gum swelling or abscess-like gum lumps.",
            "oral_surgery": "Surgical problems: wisdom-tooth (third molar) problems, facial swelling from tooth "
                            "infection needing drainage or removal, significant pain after extraction.",
            "prosthodontics": "Crowns, bridges, dentures, lost fillings and other restoration or prosthesis problems.",
            "orthodontics": "Braces, wires, brackets, retainers and other orthodontic appliance problems.",
            "pediatric_dentistry": "Dental problems in young children (roughly under 13).",
            "oral_medicine": "Mouth lesions or ulcers of shorter duration, and jaw-joint or jaw-muscle complaints "
                             "such as clicking, locking or jaw pain.",
            "oral_pathology": "Mouth lesions or ulcers that have persisted for a long time (about six weeks or more) "
                              "and need specialist assessment.",
            "emergency_dental_service": "Time-critical problems needing an emergency service: airway or swallowing "
                                        "difficulty with swelling, swelling with fever and restricted jaw opening, "
                                        "or a tooth knocked out.",
        },
    },
    "infection_concern": {
        "type": "noul",
        "instructions": (
            "Does the available case information indicate meaningful concern for an active odontogenic "
            "infection or spreading infection risk?"
        ),
        "criteria": {
            "true": "Findings such as swelling together with fever, progression, lymph node enlargement or "
                    "restricted jaw opening; pus or drainage; or difficulty swallowing.",
            "false": "No signs pointing to an active or spreading infection.",
        },
    },
    "same_day_review": {
        "type": "noul",
        "instructions": "Does the available case information support same-day human clinical review?",
        "criteria": {
            "true": "Findings suggest the case should be looked at by a person today rather than waiting for a "
                    "routine slot. This is a routing flag, not a diagnosis.",
            "false": "A routine or scheduled slot is adequate.",
        },
    },
    "human_review": {
        "type": "noul",
        "instructions": "Should this case be escalated to human review before an automated workflow proceeds?",
        "criteria": {
            "true": "The case is high-risk, ambiguous, incomplete, or involves findings an automated workflow "
                    "should not handle alone.",
            "false": "The case is clear and low-risk enough for the automated workflow to proceed.",
        },
    },
    "complexity": {
        "type": "score",
        "labels": COMPLEXITY_LEVELS,
        "instructions": "How complex is this case to manage, considering the patient and the findings?",
        "criteria": [
            "SIMPLE: a single straightforward issue in an otherwise healthy patient, or an administrative request.",
            "MODERATE: one or two complicating factors, such as a medical history item, localized swelling or trauma.",
            "COMPLEX: several complicating factors, such as infection concern with medical comorbidity, "
            "multiple medications, or a mouth lesion.",
            "HIGHLY_COMPLEX: many interacting risk factors, such as airway compromise, spreading infection with "
            "immunocompromise, or very sparse information with acute findings.",
        ],
    },
    "referral_priority": {
        "type": "score",
        "labels": REFERRAL_LEVELS,
        "instructions": "How high a priority should this case have in the referral queue?",
        "criteria": [
            "LOW: no acute findings; can wait for the normal referral queue.",
            "MODERATE: some concerning findings or complexity; should be ahead of routine referrals.",
            "HIGH: acute findings; should be handled near the front of the queue.",
            "CRITICAL: potentially life-threatening or rapidly worsening presentation; front of the queue.",
        ],
    },
}


def question_definitions() -> dict[str, Any]:
    """JSON-serialisable question definitions (Research Mode)."""
    return {"version": QUESTION_SET_VERSION, "questions": QUESTION_SPECS}


def build_sdk_questions() -> dict[str, Any]:
    """Convert QUESTION_SPECS into official SDK question objects."""
    from typesafe_sdk import Choice, Noul, NoulCriteria, Score  # imported lazily: demo mode needs no SDK

    out: dict[str, Any] = {}
    for qid, spec in QUESTION_SPECS.items():
        instr = DISCLAIMER_PREFIX + spec["instructions"]
        if spec["type"] == "score":
            out[qid] = Score(instructions=instr, criteria=list(spec["criteria"]))
        elif spec["type"] == "choice":
            out[qid] = Choice(instructions=instr, criteria=dict(spec["criteria"]))
        else:
            c = spec["criteria"]
            out[qid] = Noul(instructions=instr, criteria=NoulCriteria(true=c["true"], false=c["false"]))
    return out


# --------------------------------------------------------------------------
# Validation + normalization (raw System One JSON -> JevDecision)
# --------------------------------------------------------------------------
def _fail(msg: str) -> InvalidModelOutputError:
    return InvalidModelOutputError(
        "Jev returned a response that failed validation, so it was not used.", detail=msg
    )


def _check_prob(x: Any, name: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise _fail(f"{name}: not a finite number ({x!r})")
    if not 0.0 <= float(x) <= 1.0:
        raise _fail(f"{name}: probability {x} outside [0, 1]")
    return float(x)


def _check_distribution(probs: dict[str, Any], expected_keys: list[str], name: str) -> dict[str, float]:
    if set(probs) != set(expected_keys):
        raise _fail(f"{name}: probability keys {sorted(probs)} != expected {sorted(expected_keys)}")
    clean = {k: _check_prob(v, f"{name}.{k}") for k, v in probs.items()}
    total = sum(clean.values())
    if abs(total - 1.0) > PROB_SUM_TOLERANCE:
        raise _fail(f"{name}: probabilities sum to {total:.3f}, expected 1.0")
    return clean


def _normalize_score(ans: dict[str, Any], labels: list[str], name: str) -> ScoreDecision:
    if ans.get("type") != "score":
        raise _fail(f"{name}: expected a score answer, got {ans.get('type')!r}")
    n = len(labels)
    raw = ans.get("probabilities")
    if not isinstance(raw, dict):
        raise _fail(f"{name}: missing probabilities")
    by_index = _check_distribution({str(k): v for k, v in raw.items()}, [str(i) for i in range(n)], name)
    probs = {labels[int(i)]: p for i, p in by_index.items()}
    score = ans.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        raise _fail(f"{name}: invalid score {score!r}")
    if not -1e-6 <= score <= n - 1 + 1e-6:
        raise _fail(f"{name}: score {score} outside 0..{n - 1}")
    conf = _check_prob(ans.get("confidence"), f"{name}.confidence")
    top = max(probs, key=lambda k: probs[k])
    rounded = labels[min(n - 1, max(0, int(round(score))))]
    return ScoreDecision(label=top, score=float(score), rounded_label=rounded, confidence=conf, probabilities=probs)


def _normalize_choice(ans: dict[str, Any], labels: list[str], name: str) -> ChoiceDecision:
    if ans.get("type") != "choice":
        raise _fail(f"{name}: expected a choice answer, got {ans.get('type')!r}")
    probs = _check_distribution(ans.get("probabilities") or {}, labels, name)
    choice = ans.get("choice")
    if choice not in labels:
        raise _fail(f"{name}: choice {choice!r} is not an allowed label")
    return ChoiceDecision(
        label=choice, confidence=_check_prob(ans.get("confidence"), f"{name}.confidence"), probabilities=probs
    )


def _normalize_noul(ans: dict[str, Any], name: str) -> NoulDecision:
    if ans.get("type") != "noul":
        raise _fail(f"{name}: expected a noul answer, got {ans.get('type')!r}")
    return NoulDecision(probability=_check_prob(ans.get("noul"), name))


def normalize_response(
    raw: dict[str, Any], *, mode: str, latency_ms: float | None = None, cached: bool = False,
    usage: dict[str, int] | None = None, request_id: str | None = None,
) -> JevDecision:
    """Validate a raw System One response and convert it to a JevDecision. Raises InvalidModelOutputError."""
    if not isinstance(raw, dict) or not isinstance(raw.get("answers"), dict):
        raise _fail("response has no 'answers' object")
    a = raw["answers"]
    missing = [q for q in QUESTION_SPECS if q not in a]
    if missing:
        raise _fail(f"missing answers for: {', '.join(missing)}")
    try:
        return JevDecision(
            source="jev", mode=mode, model=str(raw.get("model", "unknown")),
            urgency=_normalize_score(a["urgency"], URGENCY_LEVELS, "urgency"),
            specialty=_normalize_choice(a["specialty"], SPECIALTIES, "specialty"),
            infection_concern=_normalize_noul(a["infection_concern"], "infection_concern"),
            same_day_review=_normalize_noul(a["same_day_review"], "same_day_review"),
            human_review=_normalize_noul(a["human_review"], "human_review"),
            complexity=_normalize_score(a["complexity"], COMPLEXITY_LEVELS, "complexity"),
            referral_priority=_normalize_score(a["referral_priority"], REFERRAL_LEVELS, "referral_priority"),
            latency_ms=latency_ms, cached=cached, usage=usage or raw.get("usage"), request_id=request_id,
            raw_response=raw,
        )
    except InvalidModelOutputError:
        raise
    except Exception as exc:  # pydantic validation etc.
        raise _fail(f"normalization error: {exc}") from exc


def _response_to_raw(response: Any) -> dict[str, Any]:
    """Prefer the exact HTTP JSON; fall back to the typed SDK objects."""
    try:
        data = response.raw_http_response.json()
        if isinstance(data, dict) and "answers" in data:
            return data
    except Exception:  # noqa: BLE001
        pass
    return {
        "model": response.model,
        "answers": {k: v.model_dump(mode="json") for k, v in response.answers.items()},
        "usage": response.usage.model_dump() if getattr(response, "usage", None) else None,
    }


# --------------------------------------------------------------------------
# Service
# --------------------------------------------------------------------------
class JevService:
    def __init__(self, settings: Settings | None = None, client: Any | None = None) -> None:
        self._settings = settings
        self._client = client  # injectable for tests (a TypeSafeClient with a mock transport)

    @property
    def settings(self) -> Settings:
        return self._settings or get_settings()

    def cache_key(self, case: CaseState, model: str | None = None) -> str:
        return stable_hash(
            {"state": case.to_jev_state(), "qv": QUESTION_SET_VERSION, "model": model or self.settings.jev_model}
        )

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.settings.has_jev_key:
            raise MissingApiKeyError(
                "TYPESAFE_API_KEY is not set. Add it to your .env file, or switch to Demo Mode."
            )
        from typesafe_sdk import RetryPolicy, TypeSafeClient, TypeSafeError

        try:
            self._client = TypeSafeClient(
                api_key=self.settings.typesafe_api_key.get_secret_value(),  # type: ignore[union-attr]
                model=self.settings.jev_model,
                retry=RetryPolicy(max_retries=self.settings.jev_max_retries, timeout=self.settings.jev_timeout_s),
            )
        except TypeSafeError as exc:
            raise MissingApiKeyError(
                "The TypeSafe API key was rejected as malformed. Check TYPESAFE_API_KEY.", detail=str(exc)
            ) from exc
        return self._client

    # ---------------------------------------------------------------- public
    def evaluate(self, case: CaseState, demo: bool | None = None, use_cache: bool = True) -> JevDecision:
        is_demo = self.settings.resolve_demo_mode(demo)
        if is_demo:
            raw = demo_simulator.simulate(case)
            decision = normalize_response(raw, mode="demo", latency_ms=None)
            log_event(logger, "jev_demo", logging.DEBUG, case_id=case.patient_id, model=decision.model, success=True, validation="ok")
            return decision
        return self._evaluate_live(case, use_cache)

    # ------------------------------------------------------------------ live
    def _evaluate_live(self, case: CaseState, use_cache: bool) -> JevDecision:
        key = self.cache_key(case)
        if use_cache:
            hit = cache_repo.get_cached(key)
            if hit is not None:
                decision = normalize_response(
                    hit["response"], mode="live", latency_ms=hit["latency_ms"], cached=True,
                    usage=hit["usage"], request_id=hit["request_id"],
                )
                log_event(logger, "jev_cache_hit", case_id=case.patient_id, model=decision.model, success=True)
                return decision

        client = self._get_client()
        from typesafe_sdk import (
            TypeSafeAPIConnectionError, TypeSafeAPIError, TypeSafeAPIResponseValidationError,
            TypeSafeAPITimeoutError, TypeSafeAuthenticationError, TypeSafeError, TypeSafeRateLimitError,
        )

        started = time.perf_counter()
        try:
            response = client.system_one(case.to_jev_state(), build_sdk_questions())
        except TypeSafeAuthenticationError as exc:
            self._log_failure(case, started, "auth")
            raise JevServiceError("Jev rejected the API key. Check TYPESAFE_API_KEY.", detail=str(exc)) from exc
        except TypeSafeAPITimeoutError as exc:
            self._log_failure(case, started, "timeout")
            raise JevServiceError("The Jev API timed out. Try again in a moment.", detail=str(exc)) from exc
        except TypeSafeAPIConnectionError as exc:
            self._log_failure(case, started, "connection")
            raise JevServiceError("Could not reach the Jev API. Check your network connection.", detail=str(exc)) from exc
        except TypeSafeRateLimitError as exc:
            self._log_failure(case, started, "rate_limit")
            raise JevServiceError("The Jev API rate limit was reached. Wait a moment and retry.", detail=str(exc)) from exc
        except TypeSafeAPIResponseValidationError as exc:
            self._log_failure(case, started, "response_validation")
            raise InvalidModelOutputError("Jev returned a response the SDK could not parse.", detail=str(exc)) from exc
        except TypeSafeAPIError as exc:
            self._log_failure(case, started, f"http_{getattr(exc, 'status', '?')}")
            raise JevServiceError(
                f"The Jev API returned an error (HTTP {getattr(exc, 'status', '?')}).", detail=str(exc)
            ) from exc
        except TypeSafeError as exc:
            self._log_failure(case, started, "sdk_error")
            raise JevServiceError("The TypeSafe SDK reported an error.", detail=str(exc)) from exc
        latency_ms = (time.perf_counter() - started) * 1000

        raw = _response_to_raw(response)
        usage = raw.get("usage")
        try:  # SDK raises (not returns None) when the header is absent; the ID is only metadata
            request_id = response.request_id
        except Exception:  # noqa: BLE001
            request_id = None
        try:
            decision = normalize_response(raw, mode="live", latency_ms=latency_ms, usage=usage, request_id=request_id)
        except InvalidModelOutputError as exc:
            log_event(logger, "jev_request", case_id=case.patient_id, model=raw.get("model"),
                      latency_ms=round(latency_ms, 1), success=False, validation=f"failed: {exc.detail}")
            raise
        log_event(logger, "jev_request", case_id=case.patient_id, model=decision.model,
                  latency_ms=round(latency_ms, 1), success=True, validation="ok", request_id=request_id)
        if use_cache:
            try:
                cache_repo.put_cached(key, decision.model, raw, latency_ms, usage, request_id)
            except Exception as exc:  # noqa: BLE001  - cache failure must never fail the request
                logger.warning("cache write failed: %s", exc)
        return decision

    @staticmethod
    def _log_failure(case: CaseState, started: float, reason: str) -> None:
        log_event(logger, "jev_request", case_id=case.patient_id, model="jev",
                  latency_ms=round((time.perf_counter() - started) * 1000, 1), success=False, validation=reason)


_service: JevService | None = None


def get_jev_service() -> JevService:
    global _service
    if _service is None:
        _service = JevService()
    return _service
