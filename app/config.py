"""Configuration and Google Cloud Secret Manager integration for AutoDeal Copilot."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from google.cloud import secretmanager


class SecretManagerClient:
    """Secure secret loader backed by Google Cloud Secret Manager.

    Ensures zero hardcoded API keys or credentials in source code. Fetches
    secrets directly from Google Cloud Secret Manager (`secretmanager.googleapis.com`)
    and supports environment variable injection on Cloud Run / Agent Runtime.
    """

    def __init__(
        self,
        project_id: Optional[str] = None,
        client: Optional[secretmanager.SecretManagerServiceClient] = None,
    ) -> None:
        self.project_id = project_id or os.environ.get(
            "GOOGLE_CLOUD_PROJECT", "autodeal-copilot-gcp"
        )
        self._client = client

    def _get_client(self) -> secretmanager.SecretManagerServiceClient:
        if self._client is None:
            self._client = secretmanager.SecretManagerServiceClient()
        return self._client

    def get_secret(
        self,
        secret_id: str,
        version_id: str = "latest",
        default: Optional[str] = None,
    ) -> Optional[str]:
        """Retrieves a secret payload from Google Cloud Secret Manager.

        Checks if the secret was already mounted as an environment variable by
        Cloud Run / Agent Runtime Secret Manager integration, then queries the
        Secret Manager API directly.

        Args:
            secret_id: Identifier of the secret in GCP Secret Manager.
            version_id: Secret version (defaults to 'latest').
            default: Fallback value when running in offline unit-test environments.

        Returns:
            The decoded secret string, or `default` if unavailable offline.
        """
        env_override = os.environ.get(secret_id)
        if env_override:
            return env_override

        resource_name = (
            f"projects/{self.project_id}/secrets/{secret_id}/versions/{version_id}"
        )
        try:
            client = self._get_client()
            response = client.access_secret_version(request={"name": resource_name})
            return response.payload.data.decode("utf-8").strip()
        except Exception:
            return default


@dataclass(frozen=True)
class AgentConfig:
    """Centralized configuration for AutoDeal Copilot agents and infrastructure.

    Uses the newest 3rd-generation Gemini models:
    - `fast_model`: `gemini-3-flash-preview` (configurable to `gemini-3.8-flash`
      or `gemini-3-flash-preview` via `AUTODEAL_FAST_MODEL`) for high-speed lookup,
      triage, and coordination.
    - `pro_model`: `gemini-3.1-pro-preview` (configurable via `AUTODEAL_PRO_MODEL`)
      for multi-constraint Top-3 offer ranking and financing optimization.
    """

    app_name: str = "app"
    project_id: str = field(
        default_factory=lambda: os.environ.get(
            "GOOGLE_CLOUD_PROJECT", "autodeal-copilot-gcp"
        )
    )
    location: str = field(
        default_factory=lambda: os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    )
    fast_model: str = field(
        default_factory=lambda: os.environ.get(
            "AUTODEAL_FAST_MODEL", "gemini-3-flash-preview"
        )
    )
    pro_model: str = field(
        default_factory=lambda: os.environ.get(
            "AUTODEAL_PRO_MODEL", "gemini-3.1-pro-preview"
        )
    )
    min_dealer_margin_dollars: float = 500.0
    manager_approval_discount_threshold: float = 1500.0
    max_history_turns: int = 16
    max_token_budget: int = 8000
    sqlite_db_path: str = field(
        default_factory=lambda: os.environ.get(
            "AUTODEAL_DB_PATH", "/tmp/autodeal_copilot_state.db"
        )
    )

    def load_api_key(
        self, secret_client: Optional[SecretManagerClient] = None
    ) -> Optional[str]:
        """Loads the Gemini API key securely via Google Cloud Secret Manager."""
        loader = secret_client or SecretManagerClient(project_id=self.project_id)
        return loader.get_secret(
            secret_id="GOOGLE_API_KEY",
            default=os.environ.get("GEMINI_API_KEY"),
        )


DEFAULT_CONFIG = AgentConfig()
