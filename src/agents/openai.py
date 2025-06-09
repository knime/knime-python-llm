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


from models.base import OutputFormatOptions
from models.openai import (
    OpenAIChatModelPortObject,
    OpenAIChatModelPortObjectSpec,
    openai_chat_port_type,
    openai_icon,
)
import knime.extension as knext
from .base import AgentPortObject, AgentPortObjectSpec
from .base import agent_category


import re


class OpenAIFunctionsAgentPortObjectSpec(AgentPortObjectSpec):
    def __init__(self, llm_spec: OpenAIChatModelPortObjectSpec, system_message) -> None:
        super().__init__(llm_spec)
        self._system_message = system_message

    @property
    def system_message(self) -> str:
        return self._system_message

    def serialize(self) -> dict:
        data = super().serialize()
        data["system_message"] = self.system_message
        return data

    @classmethod
    def deserialize(cls, data) -> "OpenAIFunctionsAgentPortObjectSpec":
        return cls(cls.deserialize_llm_spec(data), data["system_message"])


class OpenAiFunctionsAgentPortObject(AgentPortObject):
    def __init__(
        self, spec: AgentPortObjectSpec, llm: OpenAIChatModelPortObject
    ) -> None:
        super().__init__(spec, llm)

    @property
    def spec(self) -> OpenAIFunctionsAgentPortObjectSpec:
        return super().spec

    def validate_tools(self, tools):
        pattern = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
        for tool in tools:
            if not pattern.match(tool.name):
                raise knext.InvalidParametersError(
                    f"Invalid tool name '{tool.name}'. The name must be 1 to 64 characters long and can only contain alphanumeric characters, underscores, and hyphens."
                )
            if not tool.description or not tool.description.strip():
                raise knext.InvalidParametersError(
                    f"Invalid or missing tool description for tool: {tool.name}."
                )

        return tools

    def create_agent(self, ctx, tools):
        from langchain_core.prompts import MessagesPlaceholder
        from langchain_core.prompts import ChatPromptTemplate
        from langchain.agents import create_openai_functions_agent

        llm = self.llm.create_model(ctx, OutputFormatOptions.Text)
        tools = self.validate_tools(tools)
        prompt = ChatPromptTemplate(
            [
                ("system", self.spec.system_message),
                MessagesPlaceholder("chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )

        return create_openai_functions_agent(
            llm=llm,
            tools=tools,
            prompt=prompt,
        )


openai_functions_agent_port_type = knext.port_type(
    "OpenAI Functions Agent",
    OpenAiFunctionsAgentPortObject,
    OpenAIFunctionsAgentPortObjectSpec,
)


@knext.node(
    "OpenAI Functions Agent Creator",
    knext.NodeType.SOURCE,
    openai_icon,
    agent_category,
    keywords=["GenAI", "Gen AI", "Generative AI", "OpenAI", "Azure"],
    is_deprecated=True,
)
@knext.input_port(
    "(Azure) OpenAI Chat Model",
    "The (Azure) OpenAI chat model used by the agent to make decisions.",
    openai_chat_port_type,
)
@knext.output_port(
    "OpenAI Functions Agent",
    "An agent that can use OpenAI functions.",
    openai_functions_agent_port_type,
)
class OpenAIFunctionsAgentCreator:
    """
    Creates an agent that utilizes the function calling feature of (Azure) OpenAI chat models.

    This node creates an agent based on (Azure) OpenAI chat models that support function calling
    (e.g. the 0613 models) and can be primed with a custom system message.

    The *system message* plays an essential role in defining the behavior of the agent and how it interacts with users and tools.
    Before adjusting other model settings, it is recommended to experiment with the system message first, as it has
    the most significant impact on the behavior of the agent.

    An *agent* is an LLM that is configured to pick a tool from
    a set of tools to best answer the user prompts, when appropriate.

    **For Azure**: Make sure to use the correct API, since function calling is only available since API version
    '2023-07-01-preview'. For more information, check the
    [Microsoft Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/function-calling?tabs=python)

    **Note**: These agents do not support tools with whitespaces in their names.

    **Note**: If you use the
    [Credentials Configuration node](https://hub.knime.com/knime/extensions/org.knime.features.js.quickforms/latest/org.knime.js.base.node.configuration.input.credentials.CredentialsDialogNodeFactory)
    and do not select the "Save password in configuration (weakly encrypted)" option for passing the API key for the chat model,
    the Credentials Configuration node will need to be reconfigured upon reopening the workflow, as the credentials flow variable
    was not saved and will therefore not be available to downstream nodes.
    """

    system_message = knext.MultilineStringParameter(
        "System message",
        """Specify the system message defining the behavior of the agent.""",
        """You are a helpful AI assistant. Never solely rely on your own knowledge, but use tools to get information before answering. """,
    )

    def configure(self, ctx, chat_model_spec: OpenAIChatModelPortObjectSpec):
        chat_model_spec.validate_context(ctx)
        return OpenAIFunctionsAgentPortObjectSpec(chat_model_spec, self.system_message)

    def execute(self, ctx, chat_model: OpenAIChatModelPortObject):
        return OpenAiFunctionsAgentPortObject(
            OpenAIFunctionsAgentPortObjectSpec(chat_model.spec, self.system_message),
            chat_model,
        )
