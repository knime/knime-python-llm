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

"""
Custom LangChain models for IBM watsonx.ai using REST API.

These classes implement the LangChain interfaces for chat models and embeddings
using direct REST API calls instead of the ibm_watsonx_ai SDK.
"""

from typing import Any, Iterator, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.embeddings import Embeddings
from pydantic import Field, PrivateAttr

from ._api_client import WatsonxAPIClient


def _convert_message_to_dict(message: BaseMessage) -> dict:
    """Convert a LangChain message to watsonx.ai API format."""
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content}
    elif isinstance(message, HumanMessage):
        return {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        msg_dict = {"role": "assistant", "content": message.content}
        # Handle tool calls if present
        if message.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": tc.get("name", ""),
                        "arguments": (
                            tc.get("args", {})
                            if isinstance(tc.get("args"), str)
                            else _serialize_args(tc.get("args", {}))
                        ),
                    },
                }
                for tc in message.tool_calls
            ]
        return msg_dict
    elif isinstance(message, ToolMessage):
        return {
            "role": "tool",
            "content": message.content,
            "tool_call_id": message.tool_call_id,
        }
    else:
        # Fallback for other message types
        return {"role": "user", "content": str(message.content)}


def _serialize_args(args: dict) -> str:
    """Serialize tool call arguments to JSON string."""
    import json

    return json.dumps(args)


def _convert_tool_to_watsonx_format(tool: dict) -> dict:
    """Convert a tool definition to watsonx.ai format."""
    # LangChain tool format to watsonx format
    # watsonx expects OpenAI-compatible tool format
    return {
        "type": "function",
        "function": {
            "name": tool.get("title", tool.get("name", "")),
            "description": tool.get("description", ""),
            "parameters": {
                "type": "object",
                "properties": tool.get("properties", {}),
                "required": tool.get("required", []),
            },
        },
    }


class ChatWatsonx(BaseChatModel):
    """
    Chat model for IBM watsonx.ai using REST API.

    This is a drop-in replacement for langchain_ibm.ChatWatsonx that uses
    direct REST API calls instead of the ibm_watsonx_ai SDK.
    """

    model_id: str = Field(description="The model ID to use for chat completion")
    apikey: str = Field(description="IBM Cloud API key")
    url: str = Field(description="Base URL for watsonx.ai API")
    project_id: Optional[str] = Field(
        default=None, description="Watson Studio project ID"
    )
    space_id: Optional[str] = Field(default=None, description="Deployment space ID")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    max_tokens: int = Field(default=1024, description="Maximum tokens in response")
    top_p: float = Field(default=1.0, description="Top-p sampling parameter")
    n_requests: int = Field(
        default=1, description="Number of concurrent requests (not used directly)"
    )

    _client: WatsonxAPIClient = PrivateAttr()
    _bound_tools: Optional[List[dict]] = PrivateAttr(default=None)

    def __init__(self, **data):
        super().__init__(**data)
        self._client = WatsonxAPIClient(api_key=self.apikey, base_url=self.url)
        self._bound_tools = None

    @property
    def _llm_type(self) -> str:
        return "watsonx-chat"

    @property
    def _identifying_params(self) -> dict:
        return {
            "model_id": self.model_id,
            "url": self.url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }

    def bind_tools(self, tools: List[dict], **kwargs) -> "ChatWatsonx":
        """Bind tools to this chat model."""
        # Create a new instance with the same parameters but with tools bound
        new_instance = ChatWatsonx(
            model_id=self.model_id,
            apikey=self.apikey,
            url=self.url,
            project_id=self.project_id,
            space_id=self.space_id,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            n_requests=self.n_requests,
        )
        new_instance._bound_tools = [_convert_tool_to_watsonx_format(t) for t in tools]
        return new_instance

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a chat completion."""
        # Convert messages to API format
        api_messages = [_convert_message_to_dict(m) for m in messages]

        # Make API call
        response = self._client.text_chat(
            model_id=self.model_id,
            messages=api_messages,
            project_id=self.project_id,
            space_id=self.space_id,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            tools=self._bound_tools,
        )

        # Parse response
        choices = response.get("choices", [])
        if not choices:
            raise ValueError("No choices in response from watsonx.ai")

        choice = choices[0]
        message = choice.get("message", {})
        content = message.get("content", "")

        # Handle tool calls
        tool_calls = []
        if "tool_calls" in message:
            import json

            for tc in message["tool_calls"]:
                func = tc.get("function", {})
                args = func.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                tool_calls.append(
                    {
                        "id": tc.get("id", ""),
                        "name": func.get("name", ""),
                        "args": args,
                    }
                )

        ai_message = AIMessage(content=content, tool_calls=tool_calls)

        return ChatResult(
            generations=[ChatGeneration(message=ai_message)],
            llm_output={
                "model_id": self.model_id,
                "usage": response.get("usage", {}),
            },
        )

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream a chat completion."""
        # Convert messages to API format
        api_messages = [_convert_message_to_dict(m) for m in messages]

        # Make streaming API call
        for chunk in self._client.text_chat_stream(
            model_id=self.model_id,
            messages=api_messages,
            project_id=self.project_id,
            space_id=self.space_id,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            tools=self._bound_tools,
        ):
            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield ChatGenerationChunk(
                        message=AIMessageChunk(content=content)
                    )


class WatsonxEmbeddings(Embeddings):
    """
    Embeddings model for IBM watsonx.ai using REST API.

    This is a drop-in replacement for langchain_ibm.WatsonxEmbeddings that uses
    direct REST API calls instead of the ibm_watsonx_ai SDK.
    """

    def __init__(
        self,
        model_id: str,
        apikey: str,
        url: str,
        project_id: Optional[str] = None,
        space_id: Optional[str] = None,
    ):
        """
        Initialize the embeddings model.

        Args:
            model_id: The embedding model ID to use
            apikey: IBM Cloud API key
            url: Base URL for watsonx.ai API
            project_id: Watson Studio project ID
            space_id: Deployment space ID
        """
        self.model_id = model_id
        self.apikey = apikey
        self.url = url
        self.project_id = project_id
        self.space_id = space_id
        self._client = WatsonxAPIClient(api_key=apikey, base_url=url)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        import knime.extension as knext

        try:
            response = self._client.text_embeddings(
                model_id=self.model_id,
                inputs=texts,
                project_id=self.project_id,
                space_id=self.space_id,
            )

            # Extract embeddings from response
            results = response.get("results", [])
            return [r.get("embedding", []) for r in results]
        except Exception:
            raise knext.InvalidParametersError(
                "Failed to embed texts. If you selected a space to run your model, "
                "make sure that the space has a valid runtime service instance. "
                "You can check this at IBM watsonx.ai Studio under Manage tab in your space."
            )

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.

        Args:
            text: Text string to embed

        Returns:
            Embedding vector
        """
        embeddings = self.embed_documents([text])
        return embeddings[0] if embeddings else []

