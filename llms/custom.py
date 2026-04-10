from typing import Any
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_ollama import ChatOllama

from cat import LargeLanguageModel


class CustomOllama(ChatOllama, LargeLanguageModel):
    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any
    ) -> str:
        return str(self.invoke(prompt, stop=stop, run_manager=run_manager, **kwargs))

    @property
    def _llm_type(self) -> str:
        return "ollama"

    def __init__(self, **kwargs: Any) -> None:
        if kwargs.get("base_url", "").endswith("/"):
            kwargs["base_url"] = kwargs["base_url"][:-1]
        super().__init__(**kwargs)
