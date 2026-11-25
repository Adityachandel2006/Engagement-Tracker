from pathlib import Path
from pydantic_settings import BaseSettings

_BACKEND_DIR = Path(__file__).resolve().parent
_ENV_FILE = _BACKEND_DIR / ".env" if (_BACKEND_DIR / ".env").exists() else ".env"


class Settings(BaseSettings):
    news_api_key: str = ""
    gnews_api_key: str = ""
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = ""
    youtube_api_key: str = ""
    llm_model: str = ""
    llm_api_key: str = ""
    database_url: str = ""

    model_config = {"env_file": str(_ENV_FILE), "env_file_encoding": "utf-8"}


settings = Settings()
