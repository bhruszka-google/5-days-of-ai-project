"""Unit tests for app/config.py."""

from __future__ import annotations

from unittest.mock import MagicMock

from app.config import AgentConfig, SecretManagerClient


class TestConfigAndSecretManager:
    """Tests for newest 3rd-gen Gemini model config and Secret Manager injection."""

    def test_default_gemini_3_models_configured(self) -> None:
        config = AgentConfig()
        assert config.fast_model == "gemini-3-flash-preview"
        assert config.pro_model == "gemini-3.1-pro-preview"
        assert config.app_name == "app"

    def test_secret_manager_prefers_env_override(self, monkeypatch: object) -> None:
        monkeypatch.setenv("CUSTOM_SECRET_KEY", "env-mounted-secret")
        loader = SecretManagerClient(project_id="test-dealership-proj")
        assert loader.get_secret("CUSTOM_SECRET_KEY") == "env-mounted-secret"

    def test_secret_manager_fetches_secret_payload(self) -> None:
        mock_sm_client = MagicMock()
        mock_response = MagicMock()
        mock_response.payload.data = b"secret-gemini-api-key-123\n"
        mock_sm_client.access_secret_version.return_value = mock_response

        loader = SecretManagerClient(
            project_id="test-dealership-proj", client=mock_sm_client
        )
        secret_val = loader.get_secret("GOOGLE_API_KEY")

        assert secret_val == "secret-gemini-api-key-123"
        mock_sm_client.access_secret_version.assert_called_once_with(
            request={
                "name": "projects/test-dealership-proj/secrets/GOOGLE_API_KEY/versions/latest"
            }
        )

    def test_secret_manager_fallback_when_offline(self) -> None:
        mock_sm_client = MagicMock()
        mock_sm_client.access_secret_version.side_effect = RuntimeError(
            "Offline environment"
        )
        loader = SecretManagerClient(
            project_id="test-dealership-proj", client=mock_sm_client
        )
        assert (
            loader.get_secret("NON_EXISTENT_SECRET", default="fallback-token")
            == "fallback-token"
        )

    def test_agent_config_load_api_key_via_secret_manager(self) -> None:
        mock_loader = MagicMock(spec=SecretManagerClient)
        mock_loader.get_secret.return_value = "loaded-via-gsm"
        config = AgentConfig(project_id="test-proj")
        assert config.load_api_key(secret_client=mock_loader) == "loaded-via-gsm"
