from pydantic import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Settings class"""

    networks_data_container_readonly_connection_string: str
    networks_data_container_readonly: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Return a settings object"""
    return Settings()
