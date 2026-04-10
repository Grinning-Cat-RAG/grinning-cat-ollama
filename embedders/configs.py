from typing import Type
from pydantic import ConfigDict
from cat import EmbedderSettings

from .custom import CustomOllamaEmbeddings


class EmbedderOllamaConfig(EmbedderSettings):
    base_url: str = "http://localhost:11434"
    model: str = "mxbai-embed-large"

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "Ollama embedding models",
            "description": "Configuration for Ollama embeddings API",
            "link": "",
        }
    )

    @classmethod
    def pyclass(cls) -> Type[CustomOllamaEmbeddings]:
        return CustomOllamaEmbeddings
