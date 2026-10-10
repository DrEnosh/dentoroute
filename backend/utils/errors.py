"""Typed errors with human-readable messages (never crash the app on an AI failure)."""
from __future__ import annotations


class DentoRouteError(Exception):
    """Base error. `user_message` is safe to show in the UI."""

    code = "dentoroute_error"

    def __init__(self, user_message: str, *, detail: str | None = None) -> None:
        super().__init__(detail or user_message)
        self.user_message = user_message
        self.detail = detail or user_message


class MissingApiKeyError(DentoRouteError):
    code = "missing_api_key"


class JevServiceError(DentoRouteError):
    """Jev API timeout / connection / HTTP failure."""

    code = "jev_service_error"


class InvalidModelOutputError(DentoRouteError):
    """Malformed response or invalid probabilities."""

    code = "invalid_model_output"


class ClaudeUnavailableError(DentoRouteError):
    code = "claude_unavailable"


class BenchmarkError(DentoRouteError):
    code = "benchmark_error"


class RepositoryError(DentoRouteError):
    code = "database_error"


class CaseNotFoundError(DentoRouteError):
    code = "case_not_found"


class PolicyError(DentoRouteError):
    """A routing policy file is invalid or less conservative than the hard safety limits."""

    code = "policy_error"
