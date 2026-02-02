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
Custom LangChain ChatModel implementation for IBM watsonx.ai.

This module provides a LangChain-compatible chat model that uses direct REST API
calls to watsonx.ai, replacing the langchain-ibm package dependency.
"""

from typing import Any, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    AIMessage,
    ToolMessage,
)
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool

from ._client import WatsonxClient


def _convert_tool_to_dict(tool: BaseTool) -> dict[str, Any]:
    """
    Convert a LangChain tool to watsonx.ai API format for function calling.

    LangChain tools (BaseTool) define callable functions with name, description,
    and an optional args_schema (Pydantic model or dict) describing the parameters.
    The watsonx.ai API expects tools in OpenAI-compatible format.

    Args:
        tool: A LangChain BaseTool instance

    Returns:
        Dict in OpenAI function calling format:
        {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
    """
    # Handle args_schema - can be a Pydantic model, dict, or None
    if tool.args_schema is None:
        parameters = {"type": "object", "properties": {}}
    elif isinstance(tool.args_schema, dict):
        parameters = tool.args_schema
    elif hasattr(tool.args_schema, "schema"):
        parameters = tool.args_schema.schema()
    else:
        parameters = {"type": "object", "properties": {}}

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": parameters,
        },
    }


def _convert_message_to_dict(message: BaseMessage) -> dict[str, Any]:
    """
    Convert a LangChain message to watsonx.ai API format.

    LangChain uses typed message objects (HumanMessage, AIMessage, etc.) to represent
    conversation turns. The watsonx.ai REST API expects messages in OpenAI-compatible
    format: a list of dicts with 'role' and 'content' keys.

    This function handles the conversion, including special cases like:
    - AIMessage with tool_calls: Includes function call information for tool use
    - ToolMessage: Includes tool_call_id to correlate with the originating call

    Args:
        message: A LangChain BaseMessage (HumanMessage, AIMessage, SystemMessage, ToolMessage)

    Returns:
        Dict with 'role', 'content', and optionally 'tool_calls' or 'tool_call_id'
    """
    if isinstance(message, HumanMessage):
        return {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        msg: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
        # Include tool calls if present
        if hasattr(message, "tool_calls") and message.tool_calls:
            import json

            msg["tool_calls"] = [
                {
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": tc.get("name", ""),
                        "arguments": json.dumps(tc.get("args", {})),
                    },
                }
                for tc in message.tool_calls
            ]
        return msg
    elif isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content}
    elif isinstance(message, ToolMessage):
        return {
            "role": "tool",
            "content": message.content,
            "tool_call_id": message.tool_call_id,
        }
    else:
        # Fallback for other message types
        return {"role": "user", "content": str(message.content)}


def _parse_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Parse tool calls from watsonx.ai API response to LangChain format.

    When a model decides to call a tool/function, the API returns tool_calls in the
    response message. This function converts them from the API format to the format
    expected by LangChain's AIMessage.tool_calls attribute.

    API format (OpenAI-compatible):
        {"id": "...", "type": "function", "function": {"name": "...", "arguments": "{...}"}}

    LangChain format:
        {"id": "...", "name": "...", "args": {...}}

    Note: The API may return arguments as a double-encoded JSON string, which this
    function handles gracefully.

    Args:
        tool_calls: List of tool call dicts from the API response

    Returns:
        List of tool call dicts in LangChain format
    """
    import json

    lc_tool_calls = []
    for tc in tool_calls:
        func = tc.get("function", {})
        # Parse arguments - API may return double-encoded JSON string
        args_str = func.get("arguments", "{}")
        try:
            # Handle double-encoded JSON
            if isinstance(args_str, str) and args_str.startswith('"'):
                args_str = json.loads(args_str)
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
        except json.JSONDecodeError:
            args = {}
        lc_tool_calls.append(
            {
                "id": tc.get("id", ""),
                "name": func.get("name", ""),
                "args": args,
            }
        )
    return lc_tool_calls


class ChatWatsonx(BaseChatModel):
    """
    LangChain ChatModel implementation for IBM watsonx.ai.

    This implementation uses direct REST API calls instead of the ibm-watsonx-ai
    or langchain-ibm packages.
    """

    # Model configuration
    model_id: str
    """The watsonx.ai model ID to use."""

    # Authentication
    apikey: str
    """IBM Cloud API key for authentication."""

    url: str
    """Base URL for the watsonx.ai API."""

    # Project/Space configuration
    project_id: Optional[str] = None
    """Project ID to use for the API calls."""

    space_id: Optional[str] = None
    """Space ID to use for the API calls."""

    # Generation parameters
    temperature: float = 0.7
    """Sampling temperature."""

    max_tokens: int = 1024
    """Maximum number of tokens to generate."""

    top_p: float = 1.0
    """Top-p sampling parameter."""

    n_requests: int = 1
    """Number of concurrent requests for batch processing."""

    # Tools for function calling
    tools: Optional[List[dict]] = None
    """Tools available for function calling."""

    # Internal client (created lazily)
    _client: Optional[WatsonxClient] = None

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True
        underscore_attrs_are_private = True

    def _get_client(self) -> WatsonxClient:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = WatsonxClient(
                api_key=self.apikey,
                base_url=self.url,
            )
        return self._client

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "watsonx-chat"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Return identifying parameters."""
        return {
            "model_id": self.model_id,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Generate a chat completion.

        Args:
            messages: List of messages in the conversation
            stop: Stop sequences (currently not supported by watsonx.ai chat API)
            run_manager: Callback manager
            **kwargs: Additional parameters

        Returns:
            ChatResult with the generated response
        """
        # Convert LangChain messages to watsonx.ai format
        api_messages = [_convert_message_to_dict(m) for m in messages]

        # Make the API call
        client = self._get_client()
        response = client.chat(
            model_id=self.model_id,
            messages=api_messages,
            project_id=self.project_id,
            space_id=self.space_id,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            tools=self.tools,
        )

        # Extract the response
        choices = response.get("choices", [])
        if not choices:
            raise RuntimeError("No response generated from watsonx.ai")

        msg_data = choices[0].get("message", {})
        content = msg_data.get("content", "")

        # Handle tool calls if present
        tool_calls = msg_data.get("tool_calls")
        if tool_calls:
            message = AIMessage(
                content=content or "",
                tool_calls=_parse_tool_calls(tool_calls),
            )
        else:
            message = AIMessage(content=content)

        generation = ChatGeneration(message=message)

        # Extract token usage if available
        usage = response.get("usage", {})
        llm_output = {}
        if usage:
            llm_output["token_usage"] = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }

        return ChatResult(generations=[generation], llm_output=llm_output)

    def bind_tools(self, tools: List[Any], **kwargs: Any) -> "ChatWatsonx":
        """
        Bind tools to the model for function calling.

        Args:
            tools: List of tools (BaseTool instances or dicts)

        Returns:
            A new ChatWatsonx instance with tools bound
        """
        # Convert tools to API format
        formatted_tools = []
        for tool in tools:
            if isinstance(tool, BaseTool):
                formatted_tools.append(_convert_tool_to_dict(tool))
            elif isinstance(tool, dict):
                formatted_tools.append(tool)
            else:
                raise ValueError(f"Unsupported tool type: {type(tool)}")

        # Return a copy with tools bound
        return ChatWatsonx(
            model_id=self.model_id,
            apikey=self.apikey,
            url=self.url,
            project_id=self.project_id,
            space_id=self.space_id,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            n_requests=self.n_requests,
            tools=formatted_tools,
        )
