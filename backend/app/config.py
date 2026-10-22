"""Application settings, loaded from environment variables or a .env file.

Change LLM_PROVIDER / EMBEDDING_PROVIDER to switch vendors without touching code:
  offline   - runs with no API keys (deterministic local stand-ins)
  openai    - OpenAI API              (OPENAI_API_KEY)
  azure     - Azure OpenAI            (AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, deployments)
  anthropic - Anthropic Claude        (ANTHROPIC_API_KEY)    [LLM only]
  bedrock   - AWS Bedrock             (standard AWS credentials)
"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    llm_provider: str = "offline"
    embedding_provider: str = "offline"

    # Model names per provider (override in .env)
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    azure_chat_deployment: str = "gpt-4o-mini"
    azure_embedding_deployment: str = "text-embedding-3-small"
    azure_api_version: str = "2024-10-21"
    anthropic_model: str = "claude-sonnet-4-5"
    bedrock_model: str = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    bedrock_embedding_model: str = "amazon.titan-embed-text-v2:0"

    # RAG tuning knobs
    chunk_size: int = 400
    chunk_overlap: int = 60
    top_k: int = 4
    min_relevance: float = 0.15  # tune per embedding model with eval/; below this, the agent says "I don't know"

    docs_dir: Path = BASE_DIR / "data" / "docs"
    cost_table: Path = BASE_DIR / "data" / "cost_table.json"
    chroma_dir: Path = BASE_DIR / "data" / "chroma"
    collection: str = "benefits_policies"

    cors_origins: list[str] = ["http://localhost:5173"]


settings = Settings()
