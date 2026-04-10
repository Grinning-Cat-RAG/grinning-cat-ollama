from typing import Type
from pydantic import ConfigDict
from cat import LLMSettings

from .custom import CustomOllama


class LLMOllamaConfig(LLMSettings):
    base_url: str = "http://localhost:11434"
    model: str = "llama3"
    num_ctx: int = 2048
    repeat_last_n: int = 64
    repeat_penalty: float = 1.1
    temperature: float = 0.8

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "Ollama",
            "description": "Configuration for Ollama",
            "link": "https://ollama.ai/library",
        }
    )

    @classmethod
    def pyclass(cls) -> Type[CustomOllama]:
        return CustomOllama
