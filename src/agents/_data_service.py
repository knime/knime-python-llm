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
import yaml


class AgentChatViewDataService:
    def __init__(
        self,
        agent_graph,
        data_registry: DataRegistry,
        initial_message: str,
        recursion_limit: int,
        show_tool_calls_and_results: bool,
        tool_converter: LangchainToolConverter,
    ):
        self._agent_graph = agent_graph
        self._data_registry = data_registry
        self._tool_converter = tool_converter
        self._messages = (
            [data_registry.create_data_message()]
            if data_registry.has_data or tool_converter.has_data_tools
            else []
        )
        self._initial_message = initial_message
        self._recursion_limit = recursion_limit
        self._show_tool_calls_and_results = show_tool_calls_and_results

    def get_initial_message(self):
        if self._initial_message:
            return {
                "type": "ai",
                "content": self._initial_message,
            }
        else:
            pass

    def post_user_message(self, user_message: str):
        import threading

        if not hasattr(self, "_thread") or not self._thread.is_alive():
            self._last_messages = []
            self._thread = threading.Thread(
                target=self._post_user_message, args=(user_message, self._last_messages)
            )
            self._thread.start()

    def get_last_messages(self):
        if not hasattr(self, "_thread"):
            return []
        elif self._thread.is_alive():
            # wait with timeout to enable long-polling
            self._thread.join(timeout=5)

        return self._last_messages

    def _post_user_message(self, user_message: str, last_messages: list):
        self._messages.append({"role": "user", "content": user_message})
        config = {"recursion_limit": self._recursion_limit}
        try:
            final_state = self._agent_graph.invoke({"messages": self._messages}, config)
            self._messages = final_state["messages"]
        except Exception as e:
            if "Recursion limit" in str(e):
                last_messages.append(
                    {
                        "type": "error",
                        "content": f"""Recursion limit of {self._recursion_limit} reached. 
                        You can increase the limit by setting the `recursion_limit` parameter.""",
                    }
                )
                return
            else:
                last_messages.append(
                    {
                        "type": "error",
                        "content": f"An error occurred while executing the agent: {e}",
                    }
                )
                return
        from_index = next(
            (
                i
                for i in reversed(range(len(self._messages)))
                if self._messages[i].type == "human"
            ),
            -1,
        )
        for msg in self._messages[from_index + 1 :]:
            frontend_messages = self._to_frontend_messages(msg)
            if self._show_tool_calls_and_results:
                last_messages.extend(frontend_messages)
            else:
                # filter out tool calls and results
                filtered_messages = [
                    m
                    for m in frontend_messages
                    if m["type"] == "view"
                    or (m["type"] == "ai" and len(m["toolCalls"]) == 0)
                ]
                last_messages.extend(filtered_messages)

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
            return [
                {
                    "type": "view",
                    "content": viewNodeId,
                }
                for viewNodeId in viewNodeIds
            ] + [fe_message]

        return [fe_message]

    def _render_tool_call(self, tool_call):
        args = tool_call.get("args")
        return {
            "id": tool_call["id"],
            "name": self._tool_converter.desanitize_tool_name(tool_call["name"]),
            "args": yaml.dump(args, indent=2) if args else None,
        }
