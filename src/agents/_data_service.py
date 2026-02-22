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

from ._data import DataRegistry
from ._tool import LangchainToolConverter
from ._agent import (
    CancelError,
    IterationLimitError,
    validate_ai_message,
    ITERATION_CONTINUE_PROMPT,
    Agent,
)
from ._parameters import IterationLimitModeForView
from dataclasses import dataclass
import re
import yaml
import queue
import threading
import knime.extension as knext
from langchain_core.messages.human import HumanMessage

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ._conversation import AgentPrompterConversation


@dataclass
class AgentChatWidgetConfig:
    initial_message: str
    conversation_column_name: str
    iteration_limit_handling: str
    show_tool_calls_and_results: bool
    reexecution_trigger: str
    error_column_name: Optional[str]


class AgentChatWidgetDataService:
    def __init__(
        self,
        ctx,
        chat_model,
        conversation: "AgentPrompterConversation",
        toolset,
        agent_config,
        data_registry: DataRegistry,
        widget_config: AgentChatWidgetConfig,
        tool_converter: LangchainToolConverter,
        combined_tools_workflow_info: dict,
    ):
        self._ctx = ctx

        self._chat_model = chat_model
        self._conversation = FrontendConversation(
            conversation, tool_converter, self._check_canceled
        )
        self._agent_config = agent_config
        self._agent = Agent(
            self._conversation, self._chat_model, toolset, self._agent_config
        )

        self._data_registry = data_registry
        self._widget_config = widget_config

        self._get_combined_tools_workflow_info = combined_tools_workflow_info

        self._message_queue = self._conversation.frontend
        self._thread = None
        self._is_canceled = False

    def get_initial_message(self):
        if self._widget_config.initial_message:
            return {
                "type": "ai",
                "content": self._widget_config.initial_message,
            }

    def post_user_message(self, user_message: str):
        self._is_canceled = False
        human_message = HumanMessage(content=user_message)
        self._conversation.append_messages_to_backend(human_message)

        if not self._thread or not self._thread.is_alive():
            while not self._message_queue.empty():
                try:
                    self._message_queue.get_nowait()
                except queue.Empty:
                    continue

            self._thread = threading.Thread(target=self._post_user_message)
            self._thread.start()

        return {"id": human_message.id}

    def get_last_messages(self):
        messages = []

        try:
            msg = self._message_queue.get(timeout=2)
            messages.append(msg)

            while not self._message_queue.empty():
                messages.append(self._message_queue.get_nowait())
        except queue.Empty:
            pass

        return messages

    def is_processing(self):
        is_alive = self._thread and self._thread.is_alive()

        return {"is_processing": is_alive or not self._message_queue.empty()}

    def get_configuration(self):
        return {
            "show_tool_calls_and_results": self._widget_config.show_tool_calls_and_results,
            "reexecution_trigger": self._widget_config.reexecution_trigger,
        }

    def get_combined_tools_workflow_info(self):
        return self._get_combined_tools_workflow_info

    def cancel_agent(self):
        self._is_canceled = True

    # called by java, not the frontend
    def get_view_data(self):
        
        conversation_table = self._conversation.create_output_table(
            self._widget_config.conversation_column_name,
            self._widget_config.error_column_name,
        )

        meta_data, tables = self._data_registry.dump()
        view_data = {
            "data": {"data_registry": meta_data},
            "ports": [conversation_table],
            "portIds": meta_data["ids"],
        }
        return view_data

    def _post_user_message(self):
        from langchain_core.messages import AIMessage

        try:
            self._agent.run()

            messages = self._conversation.get_messages()
            if messages and isinstance(messages[-1], AIMessage):
                validate_ai_message(messages[-1])

        except CancelError as e:
            raise e
        except IterationLimitError:
            self._handle_iteration_limit_error()
        except Exception as e:
            self._conversation.append_error(e)
        finally:
            self._is_canceled = False

    def _handle_iteration_limit_error(self):
        from langchain_core.messages import AIMessage

        if (
            self._widget_config.iteration_limit_handling
            == IterationLimitModeForView.CONFIRM.name
        ):
            messages = [AIMessage(ITERATION_CONTINUE_PROMPT)]
            self._conversation._append_messages(messages)
        else:
            content = (
                f"Iteration limit of {self._agent_config.iteration_limit} reached."
            )
            self._conversation.append_error(Exception(content))

    def _check_canceled(self):
        return self._is_canceled


class FrontendConversation:
    _MESSAGE_ID_SUFFIX_PATTERN = re.compile(r"\n\n\[message_id:\s*([^\]]+)\]\s*$")

    def __init__(
        self,
        backend: "AgentPrompterConversation",
        tool_converter,
        check_canceled,
    ):
        self._frontend = queue.Queue()
        self._backend_messages = backend
        self._tool_converter = tool_converter
        self._check_canceled = check_canceled

    @property
    def frontend(self):
        return self._frontend

    def append_messages(self, messages):
        """Appends messages to both backend and frontend.
        Raises a CancelError and sanitizes the final message if the context was canceled."""
        from langchain_core.messages import AIMessage, BaseMessage

        if isinstance(messages, BaseMessage):
            messages = [messages]

        if self._check_canceled and self._check_canceled():
            self._append_messages(messages[:-1])

            # sanitize last message
            final_message = messages[-1]
            if not (isinstance(final_message, AIMessage) and final_message.tool_calls):
                self._append_messages([final_message])

            if not (
                isinstance(final_message, AIMessage) and not final_message.tool_calls
            ):
                error = CancelError("Execution canceled.")
                self.append_error_to_frontend(error)
                raise error
        else:
            self._append_messages(messages)

    def _append_messages(self, messages):
        """Appends messages to both backend and frontend without checking for cancellation."""
        from langchain_core.messages import HumanMessage

        # will not raise since backend has no context
        self._backend_messages.append_messages(messages)

        for new_message in messages:
            if isinstance(new_message, HumanMessage):
                continue

            fe_messages = self._to_frontend_messages(new_message)
            for fe_msg in fe_messages:
                self._frontend.put(fe_msg)

    def append_messages_to_backend(self, messages):
        """Appends messages only to the backend conversation."""
        # will not raise since backend has no context
        self._backend_messages.append_messages(messages)

    def append_error(self, error: Exception):
        """Appends an error to both backend and frontend."""
        self._backend_messages.append_error(error)
        self.append_error_to_frontend(error)

    def append_error_to_frontend(self, error: Exception):
        """Appends an error only to the frontend."""
        if not isinstance(error, CancelError):
            content = f"An error occurred: {error}"
        else:
            content = str(error)

        error_message = {"type": "error", "content": content}
        self._frontend.put(error_message)

    def get_messages(self):
        return self._backend_messages.get_messages()

    def create_output_table(
        self,
        output_column_name: str,
        error_column_name: str = None,
    ) -> knext.Table:
        return self._backend_messages.create_output_table(
            self._tool_converter, output_column_name, error_column_name
        )

    def _to_frontend_messages(self, message):
        # split the node-view-ids out into a separate message
        content = None
        viewNodeIds = []
        if hasattr(message, "content"):
            raw_content = message.content
            message_id_suffix = ""
            if isinstance(raw_content, str):
                message_id_match = self._MESSAGE_ID_SUFFIX_PATTERN.search(raw_content)
                if message_id_match:
                    message_id_suffix = (
                        f"\n\n[message_id: {message_id_match.group(1).strip()}]"
                    )
                    raw_content = raw_content[: message_id_match.start()]

            split = raw_content.split("View node IDs")
            content = split[0]
            if message_id_suffix:
                content = f"{content}{message_id_suffix}"
            viewNodeIds = split[1].strip().split(",") if len(split) > 1 else []

        fe_message = {
            "id": message.id if hasattr(message, "id") else None,
            "type": message.type,
            "content": content,
            "name": message.name if hasattr(message, "name") else None,
        }

        if message.type == "ai" and hasattr(message, "tool_calls"):
            fe_message["toolCalls"] = [
                self._render_tool_call(tool_call) for tool_call in message.tool_calls
            ]
        elif message.type == "tool":
            fe_message["toolCallId"] = message.tool_call_id
            fe_message["name"] = self._tool_converter.desanitize_tool_name(message.name)

        if len(viewNodeIds) > 0:
            view_msgs = []
            base_id = getattr(message, "id", "msg")
            view_name = (
                fe_message.get("name") if message.type == "tool" else "Node View"
            )
            for idx, viewNodeId in enumerate(viewNodeIds):
                view_msgs.append(
                    {
                        "id": f"{base_id}-view-{idx}",
                        "type": "view",
                        "content": viewNodeId,
                        "name": view_name,
                    }
                )
            return [fe_message] + view_msgs

        return [fe_message]

    def _render_tool_call(self, tool_call):
        args = tool_call.get("args")
        return {
            "id": tool_call["id"],
            "name": self._tool_converter.desanitize_tool_name(tool_call["name"]),
            "args": yaml.dump(args, indent=2) if args else None,
        }
