"""Claude integration - two strictly separated roles.

1. EXPLANATION LAYER (`explain`): runs AFTER Jev + the routing policy. Claude receives only the structured
   facts present in the case plus the engine's decisions, and writes a plain-language summary. It cannot
   change any decision. Its "urgency_signals" are constrained (JSON-schema enum) to facts that exist in the
   input, and the prose is checked for clinical terms the case does not support; on any problem we fall
   back to a deterministic template explanation.

2. BENCHMARK BASELINE (`decide`): Claude answers the SAME questions as Jev for the SAME case, with forced
   structured output (tool use + JSON schema). Used only by the benchmark - never to route real cases.
"""
from __future__ import annotations

import json
import time
from typing import Any

from backend.schemas.case import CaseState
from backend.schemas.decision import ClaudeDecision, JevDecision
from backend.schemas.labels import COMPLEXITY_LEVELS, REFERRAL_LEVELS, SPECIALTIES, URGENCY_LEVELS
from backend.schemas.routing import ExplanationResult, RoutingResult
from backend.services.jev_service import QUESTION_SPECS
from backend.services.signal_service import find_unsupported_claims, positive_signals
from backend.utils.config import Settings, get_settings
from backend.utils.errors import ClaudeUnavailableError, InvalidModelOutputError, MissingApiKeyError
from backend.utils.logging_utils import get_logger, log_event

logger = get_logger("claude")

DISCLAIMER = (
    "This output is generated for an educational research prototype using synthetic data and is not a "
    "diagnosis or treatment recommendation. Human review is required for any real-world decision."
)

EXPLAIN_SYSTEM = """You write short, plain-language case summaries for DentoRoute, a RESEARCH PROTOTYPE that routes SYNTHETIC dental cases.
A separate decision engine has ALREADY made all decisions. You only explain them.

Hard rules:
- Use ONLY the facts in `case_facts` and the decisions in `engine_output`. Never add symptoms, history, findings, causes or numbers that are not provided.
- Do not diagnose. Do not recommend treatment, medication, or doses. Do not say a condition is present; say the engine "flagged" or "assigned a high probability to" something.
- Do not change, second-guess or override any decision, route or probability. Quote probabilities as percentages.
- If confidence is low or human review was triggered, say so plainly and do not smooth it over.
- Refer to the case as a synthetic case. 2-4 sentences per field, no markdown, no bullet characters.
- In `urgency_signals`, list only items copied exactly from `case_facts.present_findings`.
Call the tool `submit_explanation` with your answer."""

DECIDE_SYSTEM = """You are a baseline model in a research benchmark on SYNTHETIC dental cases. This is a routing exercise, not clinical advice or diagnosis.
Answer every question below about the case, using ONLY the information in the case JSON. Choose only from the allowed labels.
Report each probability/confidence as a number between 0 and 1 reflecting how sure you are. Call the tool `submit_decisions`.

QUESTIONS AND LEVEL DEFINITIONS
{questions}"""


# ----------------------------------------------------------------- schemas ---
def _decision_tool() -> dict[str, Any]:
    prob = {"type": "number", "minimum": 0, "maximum": 1}
    props = {
        "urgency": {"type": "string", "enum": URGENCY_LEVELS},
        "urgency_confidence": prob,
        "specialty": {"type": "string", "enum": SPECIALTIES},
        "specialty_confidence": prob,
        "infection_concern": {"type": "boolean"},
        "infection_probability": prob,
        "same_day_review": {"type": "boolean"},
        "same_day_probability": prob,
        "human_review": {"type": "boolean"},
        "human_review_probability": prob,
        "complexity": {"type": "string", "enum": COMPLEXITY_LEVELS},
        "referral_priority": {"type": "string", "enum": REFERRAL_LEVELS},
    }
    return {
        "name": "submit_decisions",
        "description": "Submit structured answers to all routing questions for this synthetic case.",
        "input_schema": {"type": "object", "properties": props, "required": list(props)},
    }


def _explain_tool(allowed_signals: list[str]) -> dict[str, Any]:
    text = {"type": "string"}
    return {
        "name": "submit_explanation",
        "description": "Submit the plain-language explanation of the engine's decisions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": text, "engine_findings": text, "why_route": text,
                "urgency_signals": {"type": "array", "items": {"type": "string", "enum": allowed_signals or ["none"]}},
                "human_review_note": text, "uncertainty": text,
            },
            "required": ["summary", "engine_findings", "why_route", "urgency_signals", "human_review_note", "uncertainty"],
        },
    }


def _questions_block() -> str:
    lines: list[str] = []
    for qid, spec in QUESTION_SPECS.items():
        lines.append(f"\n[{qid}] {spec['instructions']}")
        crit = spec["criteria"]
        if isinstance(crit, dict):
            lines += [f"  - {k}: {v}" for k, v in crit.items()]
        else:
            lines += [f"  - {c}" for c in crit]
    return "\n".join(lines)


# ------------------------------------------------------- template fallback ---
def template_explanation(
    case: CaseState, decision: JevDecision, routing: RoutingResult, warnings: list[str] | None = None,
) -> ExplanationResult:
    """Deterministic explanation. Used in demo mode / without a Claude key / when Claude fails or invents facts."""
    signals = positive_signals(case)
    u = decision.urgency
    urg = u.label.lower()
    sig_text = (
        "The strongest signals in the case data were: " + ", ".join(signals[:4]) + "."
        if signals else "The case data contained no notable findings."
    )
    summary = (
        f"DentoRoute classified this synthetic case as {urg} and assigned the workflow route "
        f"'{routing.workflow_label}'. {sig_text}"
    )
    findings = (
        f"The decision engine assigned urgency '{u.label}' (score {u.score:.2f} on a 0-3 scale, "
        f"confidence {u.confidence:.0%}) and best-matching service '{decision.specialty.label}' "
        f"(probability {decision.specialty.probabilities[decision.specialty.label]:.0%}, "
        f"confidence {decision.specialty.confidence:.0%}). It gave {decision.infection_concern.probability:.0%} to "
        f"infection concern, {decision.same_day_review.probability:.0%} to same-day review and "
        f"{decision.human_review.probability:.0%} to human review."
    )
    why = " ".join(routing.reasons)
    hr = (
        "Human review was triggered by the routing policy before any automated workflow."
        if routing.requires_human
        else "Human review was not triggered by the routing policy, though human review is still required for any real-world decision."
    )
    unc = {
        "high": "Confidence in the urgency and specialty answers is high, but probabilities do not guarantee correctness.",
        "medium": "Confidence is medium: the engine's answers are somewhat spread across options, so treat the route as tentative.",
        "low": "Confidence is low: the engine's answers are spread across several options, which is why the case is not automated.",
    }[routing.confidence_band]
    return ExplanationResult(
        source="template", model=None, summary=summary, engine_findings=findings, why_route=why,
        urgency_signals=signals[:6], human_review_note=hr, uncertainty=unc, disclaimer=DISCLAIMER,
        grounding_ok=True, warnings=warnings or [],
    )


# ------------------------------------------------------------------ service --
class ClaudeService:
    def __init__(self, settings: Settings | None = None, client: Any | None = None) -> None:
        self._settings = settings
        self._client = client  # injectable for tests

    @property
    def settings(self) -> Settings:
        return self._settings or get_settings()

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.settings.has_claude_key:
            raise MissingApiKeyError("ANTHROPIC_API_KEY is not set, so Claude is unavailable.")
        import anthropic

        self._client = anthropic.Anthropic(
            api_key=self.settings.anthropic_api_key.get_secret_value(),  # type: ignore[union-attr]
            timeout=self.settings.claude_timeout_s, max_retries=self.settings.claude_max_retries,
        )
        return self._client

    @property
    def available(self) -> bool:
        return self._client is not None or self.settings.has_claude_key

    # ------------------------------------------------------------- low level
    def _call_tool(self, *, system: str, user: str, tool: dict[str, Any], max_tokens: int, temperature: float | None):
        client = self._get_client()
        import anthropic

        kwargs: dict[str, Any] = dict(
            model=self.settings.claude_model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}], tools=[tool],
            tool_choice={"type": "tool", "name": tool["name"]},
        )
        if temperature is not None:
            kwargs["temperature"] = temperature
        started = time.perf_counter()
        try:
            try:
                resp = client.messages.create(**kwargs)
            except anthropic.BadRequestError as exc:
                if temperature is not None and "temperature" in str(exc).lower():
                    kwargs.pop("temperature", None)  # some models do not accept it
                    resp = client.messages.create(**kwargs)
                else:
                    raise
        except anthropic.AuthenticationError as exc:
            raise ClaudeUnavailableError("Claude rejected the API key. Check ANTHROPIC_API_KEY.", detail=str(exc)) from exc
        except anthropic.APITimeoutError as exc:
            raise ClaudeUnavailableError("The Claude API timed out.", detail=str(exc)) from exc
        except anthropic.APIConnectionError as exc:
            raise ClaudeUnavailableError("Could not reach the Claude API.", detail=str(exc)) from exc
        except anthropic.RateLimitError as exc:
            raise ClaudeUnavailableError("The Claude API rate limit was reached.", detail=str(exc)) from exc
        except anthropic.APIStatusError as exc:
            raise ClaudeUnavailableError(f"The Claude API returned an error (HTTP {exc.status_code}).", detail=str(exc)) from exc
        latency_ms = (time.perf_counter() - started) * 1000
        block = next((b for b in resp.content if getattr(b, "type", None) == "tool_use"), None)
        if block is None or not isinstance(block.input, dict):
            raise InvalidModelOutputError("Claude did not return structured output.", detail="no tool_use block")
        usage = None
        if getattr(resp, "usage", None) is not None:
            usage = {"input_tokens": int(resp.usage.input_tokens), "output_tokens": int(resp.usage.output_tokens)}
        return dict(block.input), latency_ms, usage

    # ----------------------------------------------------------- explanation
    def explain(self, case: CaseState, decision: JevDecision, routing: RoutingResult) -> ExplanationResult:
        signals = positive_signals(case)
        absent = [
            n for n, present in [
                ("fever", case.fever), ("swelling", case.swelling), ("trismus", case.trismus),
                ("trauma", case.trauma), ("difficulty swallowing", case.difficulty_swallowing),
                ("difficulty breathing", case.difficulty_breathing),
            ] if not present
        ]
        payload = {
            "case_facts": {
                "age": case.age, "sex": case.sex, "chief_complaint": case.chief_complaint,
                "duration": case.duration, "present_findings": signals, "explicitly_absent": absent,
            },
            "engine_output": {
                "urgency": {"label": decision.urgency.label, "score_0_to_3": round(decision.urgency.score, 2),
                            "confidence": round(decision.urgency.confidence, 2)},
                "specialty": {"label": decision.specialty.label,
                              "probability": round(decision.specialty.probabilities[decision.specialty.label], 2),
                              "confidence": round(decision.specialty.confidence, 2)},
                "infection_concern_probability": round(decision.infection_concern.probability, 2),
                "same_day_review_probability": round(decision.same_day_review.probability, 2),
                "human_review_probability": round(decision.human_review.probability, 2),
                "complexity": decision.complexity.label, "referral_priority": decision.referral_priority.label,
                "mode": decision.mode,
            },
            "routing": {
                "route": routing.route, "workflow_label": routing.workflow_label,
                "triggered_rule": routing.triggered_rule, "reasons": routing.reasons,
                "confidence_band": routing.confidence_band, "requires_human": routing.requires_human,
            },
        }
        data, latency_ms, _usage = self._call_tool(
            system=EXPLAIN_SYSTEM, user=json.dumps(payload, indent=1), tool=_explain_tool(signals),
            max_tokens=900, temperature=0.2,
        )
        try:
            text_fields = {k: str(data[k]) for k in ("summary", "engine_findings", "why_route", "human_review_note", "uncertainty")}
        except KeyError as exc:
            raise InvalidModelOutputError("Claude's explanation was missing fields.", detail=str(exc)) from exc
        issues = find_unsupported_claims(" ".join(text_fields.values()), case)
        chosen = [s for s in data.get("urgency_signals", []) if s in signals]
        if issues:
            log_event(logger, "claude_explain", case_id=case.patient_id, model=self.settings.claude_model,
                      latency_ms=round(latency_ms, 1), success=False, validation="grounding_failed")
            fallback = template_explanation(
                case, decision, routing,
                warnings=["Claude's explanation mentioned findings not present in the case, so a deterministic template explanation is shown instead."],
            )
            fallback.grounding_ok = False
            fallback.grounding_issues = issues
            return fallback
        log_event(logger, "claude_explain", case_id=case.patient_id, model=self.settings.claude_model,
                  latency_ms=round(latency_ms, 1), success=True, validation="grounded")
        return ExplanationResult(
            source="claude", model=self.settings.claude_model, urgency_signals=chosen or signals[:6],
            disclaimer=DISCLAIMER, grounding_ok=True, latency_ms=latency_ms, **text_fields,
        )

    # --------------------------------------------------------- benchmark
    def decide(self, case: CaseState) -> ClaudeDecision:
        system = DECIDE_SYSTEM.format(questions=_questions_block())
        user = "Case JSON (synthetic):\n" + json.dumps(case.to_jev_state(), indent=1)
        data, latency_ms, usage = self._call_tool(
            system=system, user=user, tool=_decision_tool(), max_tokens=500, temperature=0.0
        )
        return self.parse_decision(data, latency_ms=latency_ms, usage=usage, model=self.settings.claude_model)

    @staticmethod
    def parse_decision(
        data: dict[str, Any], *, latency_ms: float | None, usage: dict[str, int] | None, model: str
    ) -> ClaudeDecision:
        """Validate Claude's structured output. Raises InvalidModelOutputError (counted as schema non-compliance)."""
        try:
            d = ClaudeDecision(model=model, latency_ms=latency_ms, usage=usage, raw_response=data, **{
                k: data[k] for k in (
                    "urgency", "specialty", "infection_concern", "same_day_review", "human_review",
                    "complexity", "referral_priority",
                )
            }, **{k: data.get(k) for k in (
                "urgency_confidence", "specialty_confidence", "infection_probability",
                "same_day_probability", "human_review_probability",
            )})
        except Exception as exc:  # KeyError, pydantic ValidationError
            raise InvalidModelOutputError("Claude's output did not match the required schema.", detail=str(exc)) from exc
        bad = []
        if d.urgency not in URGENCY_LEVELS:
            bad.append(f"urgency={d.urgency!r}")
        if d.specialty not in SPECIALTIES:
            bad.append(f"specialty={d.specialty!r}")
        if d.complexity not in COMPLEXITY_LEVELS:
            bad.append(f"complexity={d.complexity!r}")
        if d.referral_priority not in REFERRAL_LEVELS:
            bad.append(f"referral_priority={d.referral_priority!r}")
        if bad:
            raise InvalidModelOutputError("Claude returned labels outside the allowed set.", detail=", ".join(bad))
        return d


def estimate_cost(usage: dict[str, int] | None, price_in: float | None, price_out: float | None) -> float | None:
    """USD cost from token usage; None (= 'not measured') unless prices were configured. Never guessed."""
    if not usage or price_in is None or price_out is None:
        return None
    return usage.get("input_tokens", 0) / 1e6 * price_in + usage.get("output_tokens", 0) / 1e6 * price_out


_service: ClaudeService | None = None


def get_claude_service() -> ClaudeService:
    global _service
    if _service is None:
        _service = ClaudeService()
    return _service
