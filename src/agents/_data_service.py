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
from ._agent import check_for_invalid_tool_calls
import yaml
import queue
import threading

from langchain_core.messages.human import HumanMessage


class AgentChatWidgetDataService:
    def __init__(
        self,
        agent_graph,
        data_registry: DataRegistry,
        initial_message: str,
        previous_messages: list,
        recursion_limit: int,
        show_tool_calls_and_results: bool,
        reexecution_trigger: str,
        tool_converter: LangchainToolConverter,
    ):
        self._agent_graph = agent_graph
        self._data_registry = data_registry
        self._tool_converter = tool_converter
        self._messages = []
        if not previous_messages and (
            data_registry.has_data or tool_converter.has_data_tools
        ):
            self._messages.append(data_registry.create_data_message())
        elif previous_messages:
            self._messages.extend(previous_messages)
        self._initial_message = initial_message
        self._recursion_limit = recursion_limit
        self._show_tool_calls_and_results = show_tool_calls_and_results
        self._reexecution_trigger = reexecution_trigger

        self._message_queue = queue.Queue()
        self._thread = None

    def get_initial_message(self):
        if self._initial_message:
            return {
                "type": "ai",
                "content": self._initial_message,
            }

    def post_user_message(self, user_message: str):
        if not self._thread or not self._thread.is_alive():
            while not self._message_queue.empty():
                try:
                    self._message_queue.get_nowait()
                except queue.Empty:
                    continue

            self._thread = threading.Thread(
                target=self._post_user_message, args=(user_message,)
            )
            self._thread.start()

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
            "show_tool_calls_and_results": self._show_tool_calls_and_results,
            "reexecution_trigger": self._reexecution_trigger,
        }

    # called by java, not the frontend
    def get_view_data(self):
        import json
        from langchain_core.messages.base import messages_to_dict

        return json.dumps(
            {
                "conversation": messages_to_dict(self._messages),
            }
        )

    def _post_user_message(self, user_message: str):
        self._messages.append(HumanMessage(content=user_message))
        config = {
            "recursion_limit": self._recursion_limit,
            "configurable": {"thread_id": "1"},
        }

        try:
            num_messages_at_last_step = len(self._messages)
            state_stream = self._agent_graph.stream(
                {"messages": self._messages},
                config,
                stream_mode="values",  # streams state by state
            )

            final_state = None
            for state in state_stream:
                new_messages = state["messages"][num_messages_at_last_step:]
                num_messages_at_last_step = len(state["messages"])
                final_state = state

                for new_message in new_messages:
                    # already added
                    if isinstance(new_message, HumanMessage):
                        continue

                    fe_messages = self._to_frontend_messages(new_message)
                    for fe_msg in fe_messages:
                        self._message_queue.put(fe_msg)

            if final_state:
                self._messages = final_state["messages"]
                if self._messages:
                    check_for_invalid_tool_calls(self._messages[-1])

        except Exception as e:
            error_message = {"type": "error", "content": f"An error occurred: {e}"}
            if "Recursion limit" in str(e):
                error_message["content"] = (
                    f"Recursion limit of {self._recursion_limit} reached."
                )
            self._message_queue.put(error_message)

    def _to_frontend_messages(self, message):
        # split the node-view-ids out into a separate message
        content = None
        viewNodeIds = []
        if hasattr(message, "content"):
            split = message.content.split("View node IDs")
            content = split[0]
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
