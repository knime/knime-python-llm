from agents.base import AgentPortObjectSpec
from models.openai import (
    OpenAIChatModelPortObject,
    OpenAIChatModelPortObjectSpec,
    openai_chat_port_type,
    openai_icon
)
import knime.extension as knext
from .base import AgentPortObject, AgentPortObjectSpec
from .base import agent_category

from langchain.agents import OpenAIFunctionsAgent
from langchain.prompts import MessagesPlaceholder
from langchain.schema import SystemMessage


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

    def create_agent(self, ctx, tools):
        llm = self.llm.create_model(ctx)
        return OpenAIFunctionsAgent.from_llm_and_tools(
            llm=llm,
            tools=tools,
            extra_prompt_messages=[MessagesPlaceholder(variable_name="chat_history")],
            system_message=SystemMessage(content=self.spec.system_message),
        )


openai_functions_agent_port_type = knext.port_type(
    "OpenAI Functions Agent",
    OpenAiFunctionsAgentPortObject,
    OpenAIFunctionsAgentPortObjectSpec,
)


@knext.node("OpenAI Functions Agent Creator", knext.NodeType.SOURCE, openai_icon, agent_category)
@knext.input_port(
    "OpenAI Chat Model",
    "The OpenAI chat model used by the agent to make decisions.",
    openai_chat_port_type,
)
@knext.output_port(
    "OpenAI Functions Agent",
    "A agent that can use OpenAI functions.",
    openai_functions_agent_port_type,
)
class OpenAIFunctionsAgentCreator:
    """
    Creates an agent that uses the function calling feature of OpenAI chat models.

    This node creates an agent that is based on OpenAI chat models that support function calling and can be primed with a custom system message.

    In general an agent is an LLM that is configured to dynamically pick a tool from a set of tools to best answer the user prompts.
    
    Note that these agents do not support tools with whitespaces in their names.
    """
    system_message = knext.StringParameter(
        "System message",
        "Defines the general behavior of the agent.",
        "You are a helpful AI assistant.",
    )

    def configure(self, ctx, chat_model_spec: OpenAIChatModelPortObjectSpec):
        chat_model_spec.validate_context(ctx)
        return OpenAIFunctionsAgentPortObjectSpec(chat_model_spec, self.system_message)

    def execute(self, ctx, chat_model: OpenAIChatModelPortObject):
        return OpenAiFunctionsAgentPortObject(
            OpenAIFunctionsAgentPortObjectSpec(chat_model.spec, self.system_message),
            chat_model,
        )
