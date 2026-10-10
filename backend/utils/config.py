"""Application settings, loaded from environment variables / .env. No keys are hard-coded."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "config"
DOCS_DIR = PROJECT_ROOT / "docs"

SAFETY_BANNER = "Research prototype. Not for clinical diagnosis or treatment."
DEMO_BANNER = "Demo Mode — using simulated Jev responses."
TAGLINE = "From dental cases to structured decisions."


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    typesafe_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None

    jev_model: str = "jev-latest"
    claude_model: str = "claude-sonnet-5-5"

    dentoroute_demo_mode: Literal["auto", "true", "false"] = "auto"
    dentoroute_api_url: str = "http://localhost:8000"
    dentoroute_database_url: str = f"sqlite:///{DATA_DIR / 'dentoroute.db'}"

    jev_timeout_s: float = 30.0
    jev_max_retries: int = 2
    claude_timeout_s: float = 60.0
    claude_max_retries: int = 2
    benchmark_workers: int = 6
    # "conservative" (default, config/routing_policy.yaml), "legacy" (v1 baseline) or a path to a policy YAML.
    routing_policy: str = "conservative"

    # Optional pricing (USD / million tokens). None => cost reported as "not measured".
    claude_price_input_per_mtok: float | None = None
    claude_price_output_per_mtok: float | None = None
    jev_price_input_per_mtok: float | None = None
    jev_price_output_per_mtok: float | None = None

    # ---- derived helpers -------------------------------------------------
    @property
    def has_jev_key(self) -> bool:
        return bool(self.typesafe_api_key and self.typesafe_api_key.get_secret_value().strip())

    @property
    def has_claude_key(self) -> bool:
        return bool(self.anthropic_api_key and self.anthropic_api_key.get_secret_value().strip())

    def resolve_demo_mode(self, override: bool | None = None) -> bool:
        """Request-level override > env setting > auto (demo when no TypeSafe key)."""
        if override is not None:
            return override
        if self.dentoroute_demo_mode == "true":
            return True
        if self.dentoroute_demo_mode == "false":
            return False
        return not self.has_jev_key

    def secrets(self) -> list[str]:
        out = []
        for s in (self.typesafe_api_key, self.anthropic_api_key):
            if s and s.get_secret_value().strip():
                out.append(s.get_secret_value().strip())
        return out


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()


POLICY_FILES = {"conservative": "routing_policy.yaml", "legacy": "routing_policy_legacy.yaml"}


def policy_path(name: str | None = None) -> Path:
    name = name or get_settings().routing_policy
    if name in POLICY_FILES:
        return CONFIG_DIR / POLICY_FILES[name]
    return Path(name)


@lru_cache(maxsize=8)
def _load_policy_file(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_routing_policy(name: str | None = None) -> dict:
    """Active routing policy (env ROUTING_POLICY) or a named one: 'conservative' | 'legacy' | path."""
    return _load_policy_file(str(policy_path(name)))


def reset_policy_cache() -> None:
    _load_policy_file.cache_clear()
