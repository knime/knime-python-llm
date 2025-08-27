# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------
#  Copyright by KNIME AG, Zurich, Switzerland
#  Website: http://www.knime.com; Email: contact@knime.com
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License, Version 3, as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful, but
#  WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, see <http://www.gnu.org/licenses>.
#
#  Additional permission under GNU GPL version 3 section 7:
#
#  KNIME interoperates with ECLIPSE solely via ECLIPSE's plug-in APIs.
#  Hence, KNIME and ECLIPSE are both independent programs and are not
#  derived from each other. Should, however, the interpretation of the
#  GNU GPL Version 3 ("License") under any applicable laws result in
#  KNIME and ECLIPSE being a combined program, KNIME AG herewith grants
#  you the additional permission to use and propagate KNIME together with
#  ECLIPSE with only the license terms in place for ECLIPSE applying to
#  ECLIPSE and the GNU GPL Version 3 applying for KNIME, provided the
#  license terms of ECLIPSE themselves allow for the respective use and
#  propagation of ECLIPSE together with KNIME.
#
#  Additional permission relating to nodes for KNIME that extend the Node
#  Extension (and in particular that are based on subclasses of NodeModel,
#  NodeDialog, and NodeView) and that only interoperate with KNIME through
#  standard APIs ("Nodes"):
#  Nodes are deemed to be separate and independent programs and to not be
#  covered works.  Notwithstanding anything to the contrary in the
#  License, the License does not apply to Nodes, you are not required to
#  license Nodes under the License, and you are granted a license to
#  prepare and propagate Nodes, in each case even if such Nodes are
#  propagated with or for interoperation with KNIME.  The owner of a Node
#  may freely choose the license terms applicable to such Node, including
#  when such Node is propagated with or for interoperation with KNIME.
# ------------------------------------------------------------------------


from typing import Any, List, Optional
from langchain_core.language_models import LLM, BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.callbacks.manager import (
    CallbackManagerForLLMRun,
    AsyncCallbackManagerForLLMRun,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import model_validator, BaseModel
from huggingface_hub import InferenceClient, AsyncInferenceClient, ChatCompletionOutput
from .hf_base import raise_for
import util


class HFLLM(LLM):
    """Custom implementation backed by huggingface_hub.InferenceClient.
    We can't use the implementation of langchain_community because it always requires an api token (and is
    probably going to be deprecated soon) and we also can't use the langchain_huggingface implementation
    since it has torch as a required dependency."""

    model: str
    """Can be a repo id on hugging face hub or the url of a TGI server."""
    provider: str = None  # None for TGI
    hf_api_token: Optional[str] = None
    max_new_tokens: int = 512
    top_k: Optional[int] = None
    top_p: Optional[float] = 0.95
    typical_p: Optional[float] = 0.95
    temperature: Optional[float] = 0.8
    repetition_penalty: Optional[float] = None
    client: Any
    seed: Optional[int] = None

    def _llm_type(self):
        return "hfllm"

    @model_validator(mode="before")
    @classmethod
    def validate_values(cls, values: dict) -> dict:
        values["client"] = InferenceClient(
            model=values["model"],
            provider=values.get("provider"),
            timeout=120,
            token=values.get("hf_api_token"),
        )
        return values

    def _call(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager=None,
        **kwargs: Any,
    ) -> str:
        client: InferenceClient = self.client
        try:
            return client.text_generation(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                repetition_penalty=self.repetition_penalty,
                top_k=self.top_k,
                top_p=self.top_p,
                typical_p=self.typical_p,
                seed=self.seed,
            )
        except Exception as ex:
            raise_for(ex)


class HFChat(BaseChatModel):
    """Custom implementation backed by huggingface_hub.InferenceClient to avoid the torch dependency of
    langchain_huggingface."""

    model: str
    """Can be a repo id on hugging face hub or the url of a TGI server."""
    provider: str = None
    hf_api_token: Optional[str] = None
    max_tokens: int = 200
    top_p: Optional[float] = 1.0
    temperature: Optional[float] = 1.0
    client: Any
    async_client: Any
    _tools: Any = None

    def _llm_type(self):
        return "hfchat"

    @model_validator(mode="before")
    @classmethod
    def validate_values(cls, values: dict) -> dict:
        values["client"] = InferenceClient(
            model=values["model"],
            provider=values.get("provider"),
            timeout=120,
            token=values.get("hf_api_token"),
        )
        values["async_client"] = AsyncInferenceClient(
            model=values["model"],
            provider=values.get("provider"),
            timeout=120,
            token=values.get("hf_api_token"),
        )
        return values

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        client: InferenceClient = self.client
        message_dicts = self._messages_to_dicts(messages)
        try:
            prediction = client.chat_completion(
                message_dicts,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                tools=self._tools,
            )
        except Exception as ex:
            raise_for(ex)
        return self._completion_to_chat_result(prediction)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        async_client: AsyncInferenceClient = self.async_client
        message_dicts = self._messages_to_dicts(messages)
        try:
            prediction = await async_client.chat_completion(
                message_dicts,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                tools=self._tools,
            )
        except Exception as ex:
            raise_for(ex)
        return self._completion_to_chat_result(prediction)

    def _messages_to_dicts(self, messages: list[BaseMessage]) -> list[dict]:
        import langchain_core.messages as lcm

        def _msg_to_dict(msg: BaseMessage):
            if isinstance(msg, lcm.SystemMessage):
                return {"role": "system", "content": msg.content}
            if isinstance(msg, lcm.AIMessage):
                return {"role": "assistant", "content": msg.content}
            if isinstance(msg, lcm.HumanMessage):
                return {"role": "user", "content": msg.content}
            if isinstance(msg, lcm.ToolMessage):
                return {
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": msg.tool_call_id,
                }
            if isinstance(msg, lcm.ChatMessage):
                return {"role": msg.role, "content": msg.content}
            raise NotImplementedError("Received unexpected message type.")

        return [_msg_to_dict(msg) for msg in messages]

    def bind_tools(self, tools):
        raise RuntimeError(
            "Tool calling is currently not supported with HF Hub models because Hugging Face does not "
            "resolve ToolMessages properly."
        )

    def _completion_to_chat_result(
        self, completion: ChatCompletionOutput
    ) -> ChatResult:
        content = ""
        if completion.choices and completion.choices[0].message.content:
            content = completion.choices[0].message.content
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content=content,
                    )
                )
            ]
        )


class HuggingFaceEmbeddings(BaseModel, Embeddings):
    """Custom implementation backed by huggingface_hub.InferenceClient to avoid the torch dependency of
    langchain_huggingface."""

    model: str
    """Can be a model or a repo id on hugging face hub or the url of a TEI server."""
    provider: str = None  # None for TEI
    hf_api_token: Optional[str] = None
    client: Any
    async_client: Any

    @model_validator(mode="before")
    @classmethod
    def validate_values(cls, values: dict) -> dict:
        values["client"] = InferenceClient(
            model=values["model"],
            provider=values.get("provider"),
            timeout=120,
            token=values.get("hf_api_token"),
        )
        values["async_client"] = AsyncInferenceClient(
            model=values["model"],
            provider=values.get("provider"),
            timeout=120,
            token=values.get("hf_api_token"),
        )
        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = [
            text.replace("\n", " ") for text in texts
        ]  # newlines can negatively affect performance according to langchain_huggingface
        try:
            responses = self.client.feature_extraction(text=texts)
        except Exception as ex:
            raise_for(ex)
        return responses.tolist()

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = [
            text.replace("\n", " ") for text in texts
        ]  # newlines can negatively affect performance according to langchain_huggingface
        try:
            responses = await self.async_client.feature_extraction(text=texts)
        except Exception as ex:
            raise_for(ex)
        return responses.tolist()

    def embed_query(self, text: str) -> list[float]:
        response = self.embed_documents([text])[0]
        return response

    async def aembed_query(self, text: str) -> list[float]:
        response = (await self.aembed_documents([text]))[0]
        return response


class HuggingFaceTEIEmbeddings(HuggingFaceEmbeddings):
    batch_size: int = 32

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Overrides HuggingFaceEmbeddings 'embed_documents' to allow batches
        return util.batched_apply(super().embed_documents, texts, self.batch_size)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        # Overrides HuggingFaceEmbeddings 'aembed_documents' to allow batches
        return util.abatched_apply(super().aembed_documents, texts, self.batch_size)
