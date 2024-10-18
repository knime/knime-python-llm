from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_core.outputs import ChatResult
from langchain_core.messages import BaseMessage
from langchain_core.language_models.base import LanguageModelInput
from .base import (
    OutputModeOptions,
)
from typing import List, Dict, Any, Optional


class _ChatOpenAI(BaseChatModel):
    client: ChatOpenAI
    response_format: Optional[str] = None

    def _prepare_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if self.response_format == OutputModeOptions.JSON.name:
            kwargs["response_format"] = {"type": "json_object"}
        return kwargs

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        kwargs = self._prepare_kwargs(kwargs)
        return await self.client._agenerate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        kwargs = self._prepare_kwargs(kwargs)
        return self.client._generate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )

    async def abatch(
        self, sub_batch: List[Any], stop: Optional[List[str]] = None, **kwargs: Any
    ) -> List[Any]:
        kwargs = self._prepare_kwargs(kwargs)
        return await super().abatch(sub_batch, stop=stop, **kwargs)

    def invoke(
        self,
        input: LanguageModelInput,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> BaseMessage:
        kwargs = self._prepare_kwargs(kwargs)
        return super().invoke(input, config, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "wrapper-chat-openai"
