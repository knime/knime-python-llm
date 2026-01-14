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

from dataclasses import dataclass
from typing import Protocol, Sequence
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage
from langchain_core.language_models.chat_models import BaseChatModel


LANGGRAPH_RECURSION_MESSAGE = "Sorry, need more steps to process this request."
RECURSION_CONTINUE_PROMPT = (
    "I stopped due to reaching the recursion limit. Do you want me to continue?"
)


def validate_ai_message(msg: AIMessage):
    """Handles invalid tool calls and empty responses."""

    finish_reason = msg.response_metadata.get("finish_reason")

    if msg.invalid_tool_calls:
        invalid_tool_call = msg.invalid_tool_calls[0]
        name = invalid_tool_call.get("name", "<unknown tool>")
        error = invalid_tool_call.get("error", None)
        if finish_reason == "length":
            raise ValueError(
                f"The LLM attempted to call the tool '{name}', but ran out of tokens before finishing the request.\n"
                "Tip: Try increasing the token limit in the LLM Selector."
            )
        else:
            user_error = (
                error if error else "No error details were provided by the model."
            )
            raise ValueError(
                f"The LLM attempted to call the tool '{name}', but the request was invalid.\n"
                f"Details: {user_error}\n"
                "Tip: Check your tool definition and arguments, or try rephrasing your prompt."
            )
    elif msg.content == "":
        if finish_reason == "length":
            raise ValueError(
                "The LLM generated an empty response because it used all response tokens for its internal "
                "reasoning. Tip: Try increasing the token limit in the LLM Selector.",
            )
        if finish_reason == "content_filter":
            raise ValueError(
                "The LLM generated an empty response because the message was filtered for harmful content.",
            )


class Conversation(Protocol):
    def append_messages(self, messages): ...

    def append_error(self, error): ...

    def get_messages(self): ...


class Toolset(Protocol):
    # need to be openai compatible
    @property
    def tools(self): ...

    def execute(self, tool_calls) -> list[ToolMessage]: ...


class Context(Protocol):
    def is_canceled(self) -> bool: ...


@dataclass
class AgentConfig:
    iteration_limit: int = 10


class IterationLimitError(RuntimeError):
    pass


class CancelError(RuntimeError):
    pass


class Agent:
    def __init__(
        self,
        conversation: Conversation,
        llm: BaseChatModel,
        toolset: Toolset,
        config: AgentConfig,
    ):
        tools = toolset.tools
        if tools:
            self._agent = llm.bind_tools(toolset.tools)
        else:
            self._agent = llm
        self._conversation = conversation
        self._config = config
        self._toolset = toolset

    def run(self):
        """Run the agent's turn in the conversation."""
        for _ in range(self._config.iteration_limit):
            try:
                response = self._agent.invoke(self._conversation.get_messages())
            except Exception as error:
                self._conversation.append_error(error)
                return
            self._append_messages(response)

            if response.tool_calls:
                try:
                    results = self._toolset.execute(response.tool_calls)
                except Exception as error:
                    self._conversation.append_error(error)
                    return
                self._append_messages(results)
            else:
                return

        raise IterationLimitError("Reached iteration limit")

    def _append_messages(self, messages: Sequence[BaseMessage] | BaseMessage):
        if isinstance(messages, BaseMessage):
            messages = [messages]
        self._conversation.append_messages(messages)
