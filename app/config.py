"""
Configuration management for ThriveBot
Uses pydantic-settings for type-safe configuration
"""

from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Google Gemini
    gemini_api_key: str
    
    # Slack Configuration
    slack_bot_token: str
    slack_app_token: str
    slack_signing_secret: str
    
    # RAG Configuration
    vector_store_path: str = "data/vector_store"
    documents_path: str = "data/documents"
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 5
    
    # Application Settings
    app_env: str = "development"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"
    
    # Optional Azure settings
    azure_storage_connection_string: Optional[str] = None
    azure_search_endpoint: Optional[str] = None
    azure_search_key: Optional[str] = None
    
    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        return self.app_env.lower() == "development"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Use this function to access settings throughout the application.
    """
    return Settings()


# Convenience function for accessing settings
settings = get_settings()
