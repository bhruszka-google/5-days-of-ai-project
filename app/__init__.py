"""AutoDeal Copilot — Enterprise Car Salesman Top-3 Offer Copilot package."""

from app import agent
from app.agent import app, create_adk_app, create_dealership_coordinator, root_agent
from app.config import DEFAULT_CONFIG, AgentConfig, SecretManagerClient

__all__ = [
    "AgentConfig",
    "DEFAULT_CONFIG",
    "SecretManagerClient",
    "agent",
    "app",
    "create_adk_app",
    "create_dealership_coordinator",
    "root_agent",
]
