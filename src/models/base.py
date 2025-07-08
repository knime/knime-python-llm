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


# KNIME / own imports
import knime.extension as knext
from knime.extension import Schema
import util
from base import AIPortObjectSpec
from typing import List, Callable, Optional, Type
from functools import partial

model_category = knext.category(
    path=util.main_category,
    level_id="models",
    name="Models",
    description="",
    icon="icons/generic/brain.png",
)


@knext.parameter_group(label="Model Parameters")
class GeneralSettings:
    temperature = knext.DoubleParameter(
        label="Temperature",
        description="""
        Sampling temperature to use, between 0.0 and 100.0. 
        Higher values will make the output more random, 
        while lower values will make it more focused and deterministic.
        """,
        default_value=0.2,
        min_value=0.0,
        max_value=100.0,
        is_advanced=True,
    )

    top_p = knext.DoubleParameter(
        label="Top-p sampling",
        description="""
        An alternative to sampling with temperature, 
        where the model considers the results of the tokens (words) 
        with top_p probability mass. Hence, 0.1 means only the tokens 
        comprising the top 10% probability mass are considered.
        """,
        default_value=0.15,
        min_value=0.01,
        max_value=1.0,
        is_advanced=True,
    )


class GeneralRemoteSettings(GeneralSettings):
    n_requests = knext.IntParameter(
        label="Number of concurrent requests",
        description="""Maximum number of concurrent requests to LLMs that can be made, 
        whether through API calls or to an inference server.
        Exceeding this limit may result in temporary restrictions on your access.

        It is important to plan your usage according to the model provider's rate limits,
        and keep in mind that both software and hardware constraints can impact performance.

        For OpenAI, please refer to the [Limits page](https://platform.openai.com/account/limits) 
        for the rate limits available to you.
        """,
        default_value=1,
        min_value=1,
        is_advanced=True,
        since_version="5.3.0",
    )


def _supports_json_mode(
    ctx: knext.DialogCreationContext,
) -> bool:
    input_spec = ctx.get_input_specs()[0]

    # this check is necessary to be able to open the configuration dialog
    # of the nodes when they are not connected to any other node
    if input_spec is None:
        return False

    return OutputFormatOptions.JSON in input_spec.supported_output_formats


class OutputFormatOptions(knext.EnumParameterOptions):
    Text = (
        "Text",
        "Text output message generated by the model.",
    )

    JSON = (
        "JSON",
        """
        When JSON is selected, the model is constrained to only generate strings 
        that parse into valid JSON object. Make sure you include the string "JSON"
        in your prompt or system message to instruct the model to output valid JSON 
        when this mode is selected.  
        For example: "Tell me a joke. Please only reply in valid JSON."
        Please refer to the OpenAI [guide](https://platform.openai.com/docs/guides/structured-outputs/structured-outputs-vs-json-mode) 
        to see which models currently support JSON outputs.
        """,
    )


def _get_output_format_value_switch() -> knext.EnumParameter:
    return knext.EnumParameter(
        "Output format",
        "Choose between different output formats.",
        OutputFormatOptions.Text.name,
        OutputFormatOptions,
        style=knext.EnumParameter.Style.VALUE_SWITCH,
        since_version="5.4.1",
    ).rule(
        knext.DialogContextCondition(_supports_json_mode),
        knext.Effect.ENABLE,
    )


def _tool_definition_table_present(ctx: knext.DialogCreationContext) -> bool:
    specs = ctx.get_input_specs()
    return len(specs) > 2 and specs[2] is not None


def _assert_tool_title_openai_compatibility(tool_dict: dict) -> None:
    """
    Ensure the tool title is OpenAI-compatible, meaning it follows the required format:

    - Only letters (a-z, A-Z), numbers (0-9), underscores (_), and hyphens (-) are allowed.
    - No spaces or special characters.
    """
    import re

    openai_tool_name_pattern = re.compile(r"^[a-zA-Z0-9_-]+$")

    if "title" not in tool_dict:
        raise ValueError("Tool definitions must have a 'title' property.")

    title = tool_dict["title"]
    if not openai_tool_name_pattern.match(title):
        raise ValueError(
            f"""Invalid tool title: '{title}'.
            
            Tool titles can only contain letters (a-z, A-Z), numbers (0-9), underscores (_), and hyphens (-).
            No spaces or special characters are allowed. Examples of allowed formats: 'get_weather', 'fetchData', 'convert-temperature', 'calculate_area_2'."""
        )


@knext.parameter_group(label="Conversation")
class ChatConversationSettings:
    def __init__(self, port_index=1) -> None:
        self.role_column = knext.ColumnParameter(
            "Role column",
            """Select the column of the conversation table that specifies the role assigned to each message.
            The column can be empty if starting the conversation from scratch.
            
            **Example roles**: 'human', 'ai'.""",
            port_index=port_index,
            column_filter=util.create_type_filter(knext.string()),
        )

        self.content_column = knext.ColumnParameter(
            "Message column",
            """Select the column of the conversation table that specifies the messages.
            The column can be empty if starting the conversation from scratch.""",
            port_index=port_index,
            column_filter=util.create_type_filter(knext.string()),
        )

    def configure(self, input_table_spec: knext.Schema):
        available_columns = [c for c in input_table_spec]
        available_columns.reverse()
        if self.role_column:
            util.check_column(
                input_table_spec,
                self.role_column,
                knext.string(),
                "role",
            )
        else:
            self.role_column = util.pick_default_column(
                available_columns, knext.string()
            )
        available_columns = [c for c in available_columns if c.name != self.role_column]
        if self.content_column:
            util.check_column(
                input_table_spec,
                self.content_column,
                knext.string(),
                "content",
            )
        else:
            try:
                self.content_column = util.pick_default_column(
                    available_columns, knext.string()
                )
            except knext.InvalidParametersError:
                raise knext.InvalidParametersError(
                    "The conversation table must contain at least two string columns. "
                    "One for the message roles and one for the message contents."
                )

        if self.role_column == self.content_column:
            raise knext.InvalidParametersError(
                "The role and content column cannot be the same."
            )

    def create_messages(self, data_frame):  # -> list[ToolMessage | Any | ChatMessage]:
        if data_frame.empty:
            return []

        return data_frame.apply(self._create_message, axis=1).tolist()

    def _create_message(self, row: dict):
        import langchain_core.messages as lcm

        role = row.get(self.role_column)
        if role is None:
            raise ValueError("No role provided.")
        message_type = self._role_to_message_type.get(role.lower(), None)
        content = row.get(self.content_column)
        if message_type == lcm.AIMessage:
            return lcm.AIMessage(content=content)
        if message_type:
            return message_type(content=content)
        else:
            # fallback to be used if the user provides other roles
            # which may or may not work in subsequent calls
            return lcm.ChatMessage(content=content, role=role)

    @property
    def _role_to_message_type(self):
        import langchain_core.messages as lcm

        return {
            "ai": lcm.AIMessage,
            "assistant": lcm.AIMessage,
            "user": lcm.HumanMessage,
            "human": lcm.HumanMessage,
            "tool": lcm.ToolMessage,
        }


@knext.parameter_group(label="Tool Calling", since_version="5.4.3")
class ToolCallingSettings:
    def __init__(self):
        conversation_table_port_index = 1
        tool_definitions_table_port_index = 2

        self.ignore_chat_message_if_last_message_is_tool = knext.BoolParameter(
            "Ignore new message during tool calling",
            """If enabled, the 'New message' specified in the configuration dialog will not be appended
             to the conversation table while the last row is a tool call message.
             
             In most cases, the new message would be what caused the model to invoke tool calling in the first place,
             and will have already been appended to the conversation table.""",
            default_value=lambda v: v
            < knext.Version(
                5, 4, 3
            ),  # False for versions < 5.4.3 for backwards compatibility
        )

        self.tool_name_column = knext.ColumnParameter(
            "Tool name column (‘Conversation History’ table)",
            """Select the column of the conversation table specifying tool names as **strings**.
            
            This column gets populated with the name of tool the model decides to call.""",
            port_index=conversation_table_port_index,
            column_filter=util.create_type_filter(knext.string()),
            include_none_column=True,
            default_value=knext.ColumnParameter.NONE,
        )

        self.tool_call_id_column = knext.ColumnParameter(
            "Tool call ID column (‘Conversation History’ table)",
            """Select the column of the conversation table specifying tool call IDs as **strings**.
            
            This column gets populated with IDs of tool calls, so that they can be referenced by the model.""",
            port_index=conversation_table_port_index,
            column_filter=util.create_type_filter(knext.string()),
            include_none_column=True,
            default_value=knext.ColumnParameter.NONE,
        )

        self.tool_call_arguments_column = knext.ColumnParameter(
            "Tool call arguments column (‘Conversation History’ table)",
            """Select the column of the conversation table specifying tool call arguments as **JSON objects**.
            
            This column gets populated with the expected input arguments for the tool the model decides to call.""",
            port_index=conversation_table_port_index,
            column_filter=util.create_type_filter(knext.logical(dict)),
            include_none_column=True,
            default_value=knext.ColumnParameter.NONE,
        )

        self.tool_definition_column = knext.ColumnParameter(
            "Tool definition column (‘Tool Definitions’ table)",
            """Select the column of the tool definitions table containing definitions of the tools that should be available to the chat model.
            
            Tool definitions take the form of JSON Schema-based objects, specifying the tool's name, description, parameters, and required fields.""",
            port_index=tool_definitions_table_port_index,
            column_filter=util.create_type_filter(knext.logical(dict)),
        )


class ToolChatConversationSettings(ChatConversationSettings):
    def __init__(self, port_index=1):
        super().__init__(port_index)

        self.tool_calling_settings = ToolCallingSettings().rule(
            knext.DialogContextCondition(_tool_definition_table_present),
            knext.Effect.SHOW,
        )

    def configure(self, input_table_spec: knext.Schema, tool_table_spec: knext.Schema):
        super().configure(input_table_spec)

        has_tools = tool_table_spec is not None
        if has_tools:
            # auto configure is not implemented because there is a bug
            # on the java side that results in dialogs not showing changes made
            # by configure calls after the dialog was opened the first time
            # since the input here is optional, it's very likely that the user
            # will have opened the dialog before connecting the tool table
            self._check_tool_column(
                input_table_spec,
                self.tool_calling_settings.tool_name_column,
                knext.string(),
                "tool name",
            )
            self._check_tool_column(
                input_table_spec,
                self.tool_calling_settings.tool_call_id_column,
                knext.string(),
                "tool call ID",
            )
            self._check_tool_column(
                input_table_spec,
                self.tool_calling_settings.tool_call_arguments_column,
                knext.logical(dict),
                "tool call arguments",
            )
            util.check_column(
                tool_table_spec,
                self.tool_calling_settings.tool_definition_column,
                knext.logical(dict),
                "tool definition",
                "tool definition",
            )

    def _check_tool_column(
        self, conversation_table_spec, column, ctype, column_purpose
    ):
        if column == knext.ColumnParameter.NONE:
            raise knext.InvalidParametersError(
                f"Please select a column containing the {column_purpose}."
            )
        util.check_column(
            conversation_table_spec, column, ctype, column_purpose, "tool definitions"
        )

    def _create_message(self, row: dict):
        import langchain_core.messages as lcm
        import pandas as pd

        role = row.get(self.role_column).lower()
        if pd.isna(role):
            raise ValueError("No role provided.")
        content = row.get(self.content_column)
        if pd.isna(content):
            raise ValueError("No content provided.")
        if role == "tool":
            tool_call_id = row.get(self.tool_calling_settings.tool_call_id_column)
            if pd.isna(tool_call_id):
                raise ValueError("No tool call ID provided.")
            return lcm.ToolMessage(
                content=content,
                tool_call_id=row.get(self.tool_calling_settings.tool_call_id_column),
            )
        if role == "ai":
            tool_calls = []
            if (
                self.tool_calling_settings.tool_call_arguments_column in row
                and pd.notna(row[self.tool_calling_settings.tool_call_id_column])
            ):
                tool_calls.append(
                    lcm.ToolCall(
                        name=row[self.tool_calling_settings.tool_name_column],
                        id=row[self.tool_calling_settings.tool_call_id_column],
                        args=row[self.tool_calling_settings.tool_call_arguments_column],
                    )
                )
            return lcm.AIMessage(content=content, tool_calls=tool_calls)
        return super()._create_message(row)


@knext.parameter_group(label="Credentials")
class CredentialsSettings:
    def __init__(self, label, description):
        def _get_default_credentials(identifier):
            return knext.DialogCreationContext.get_credential_names(identifier)

        self.credentials_param = knext.StringParameter(
            label=label,
            description=description,
            choices=lambda a: _get_default_credentials(a),
        )


class LLMPortObjectSpec(AIPortObjectSpec):
    """Most generic spec of LLMs. Used to define the most generic LLM PortType"""

    def __init__(
        self,
        n_requests: int = 1,
    ) -> None:
        super().__init__()
        self._n_requests = n_requests

    @property
    def n_requests(self) -> int:
        return self._n_requests

    @property
    def supported_output_formats(self) -> list[OutputFormatOptions]:
        return [OutputFormatOptions.Text]

    @property
    def is_instruct_model(self) -> bool:
        return True


class LLMPortObject(knext.PortObject):
    def __init__(self, spec: LLMPortObjectSpec) -> None:
        super().__init__(spec)

    def serialize(self) -> bytes:
        return b""

    @classmethod
    def deserialize(cls, spec: LLMPortObjectSpec, storage: bytes):
        return cls(spec)

    def create_model(
        self, ctx: knext.ExecutionContext, output_format: OutputFormatOptions
    ):
        raise NotImplementedError()


llm_port_type = knext.port_type("LLM", LLMPortObject, LLMPortObjectSpec)


class LLMModelType(knext.EnumParameterOptions):
    CHAT = (
        "Chat",
        "The model is trained to follow instructions in a conversational style.",
    )
    INSTRUCT = (
        "Instruct",
        "The model is trained to follow one-shot instructions and is not well suited for conversations.",
    )


def create_model_type_switch() -> knext.EnumParameter:
    return knext.EnumParameter(
        "Model type",
        "The type of the selected model.",
        LLMModelType.CHAT.name,
        LLMModelType,
        since_version="5.5.0",
        is_advanced=True,
        style=knext.EnumParameter.Style.VALUE_SWITCH,
    )


class ChatModelPortObjectSpec(LLMPortObjectSpec):
    """Most generic chat model spec. Used to define the most generic chat model PortType."""

    def __init__(
        self,
        n_requests: int = 1,
    ) -> None:
        super().__init__(n_requests=n_requests)

    @property
    def supports_tools(self) -> bool:
        return True

    @property
    def is_instruct_model(self) -> bool:
        # chat models are typically not instruct models
        # There might be exceptions in subclasses, which will need to implement the necessary logic
        return False


class ChatModelPortObject(LLMPortObject):
    def __init__(self, spec: ChatModelPortObjectSpec) -> None:
        super().__init__(spec)

    def serialize(self):
        return b""

    @classmethod
    def deserialize(cls, spec, data: dict):
        return cls(spec)

    def create_model(
        self, ctx: knext.ExecutionContext, output_format: OutputFormatOptions
    ):
        raise NotImplementedError()


chat_model_port_type = knext.port_type(
    "Chat Model", ChatModelPortObject, ChatModelPortObjectSpec
)


class EmbeddingsPortObjectSpec(AIPortObjectSpec):
    """Most generic embedding model spec. Used to define the most generic embedding model PortType."""


class EmbeddingsPortObject(knext.PortObject):
    def __init__(self, spec: EmbeddingsPortObjectSpec) -> None:
        super().__init__(spec)

    def serialize(self):
        return b""

    @classmethod
    def deserialize(cls, spec, data: dict):
        return cls(spec)

    def create_model(self, ctx: knext.ExecutionContext):
        raise NotImplementedError()


embeddings_model_port_type = knext.port_type(
    "Embedding", EmbeddingsPortObject, EmbeddingsPortObjectSpec
)


def _message_cell_col_filter(column: knext.Column):
    """Accept string, dict, or MessageCell"""

    return column.ktype in [knext.string(), knext.logical(dict), util.message_type()]


class SystemMessageHandling(knext.EnumParameterOptions):
    NONE = (
        "None",
        "No system message will precede the prompts.",
    )
    SINGLE = (
        "Global",
        "A specifiable system message will precede each prompt.",
    )
    COLUMN = (
        "Column",
        "Each prompt includes an individual system message specified in a column.",
    )


def _isinstance_of_port_object(
    ctx: knext.DialogCreationContext, port: int, spec_class: Type[knext.PortObjectSpec]
) -> bool:
    """Returns true if the port object spec is an instance of a specific knext.PortObjectSpec."""
    return isinstance(ctx.get_input_specs()[port], spec_class)


# region LLM Prompter
@knext.node(
    "LLM Prompter",
    knext.NodeType.PREDICTOR,
    "icons/generic/brain.png",
    model_category,
    keywords=[
        "GenAI",
        "Gen AI",
        "Generative AI",
        "Large Language Model",
    ],
)
@knext.input_port("LLM", "A Large Language Model.", llm_port_type)
@knext.input_table("Prompt Table", "A table containing a string column with prompts.")
@knext.output_table(
    "Result Table", "A table containing prompts and their respective answer."
)
class LLMPrompter:
    """
    Interact with an LLM using each row of the input table as an independent prompt.

    For each row in the input table, this node sends one prompt to the LLM and receives a corresponding response.
    Rows and the corresponding prompts are treated in isolation, i.e. the LLM cannot remember the contents of the previous rows or how it responded to them.

    **Note**: If you use the
    [Credentials Configuration node](https://hub.knime.com/knime/extensions/org.knime.features.js.quickforms/latest/org.knime.js.base.node.configuration.input.credentials.CredentialsDialogNodeFactory)
    and do not select the "Save password in configuration (weakly encrypted)" option for passing the API key for the LLM Selector node,
    the Credentials Configuration node will need to be reconfigured upon reopening the workflow, as the credentials flow variable
    was not saved and will therefore not be available to downstream nodes.
    """

    system_message_handling = knext.EnumParameter(
        "Add system message",
        "Specify whether a customizable system message is included in each prompt. This option is only "
        "available for chat models. Note that some chat models (e.g. OpenAI's o1-mini) do not support system messages.",
        default_value=SystemMessageHandling.NONE.name,
        enum=SystemMessageHandling,
        style=knext.EnumParameter.Style.VALUE_SWITCH,
        since_version="5.4.0",
    ).rule(
        knext.DialogContextCondition(
            lambda ctx: ctx.get_input_specs()[0] is not None
            and not ctx.get_input_specs()[0].is_instruct_model
        ),
        knext.Effect.SHOW,
    )

    system_message = knext.MultilineStringParameter(
        "System message",
        """
        The first message given to the model describing how it should behave.

        Example: You are a helpful assistant that has to answer questions truthfully, and
        if you do not know an answer to a question, you should state that.
        """,
        default_value="",
        since_version="5.4.0",
    ).rule(
        knext.And(
            knext.DialogContextCondition(
                lambda ctx: ctx.get_input_specs()[0] is not None
                and not ctx.get_input_specs()[0].is_instruct_model
            ),
            knext.OneOf(system_message_handling, [SystemMessageHandling.SINGLE.name]),
        ),
        knext.Effect.SHOW,
    )

    system_message_column = knext.ColumnParameter(
        "System message column",
        """
        The column containing the system message for each prompt.
        """,
        port_index=1,
        since_version="5.4.0",
    ).rule(
        knext.And(
            knext.DialogContextCondition(
                lambda ctx: ctx.get_input_specs()[0] is not None
                and not ctx.get_input_specs()[0].is_instruct_model
            ),
            knext.OneOf(system_message_handling, [SystemMessageHandling.COLUMN.name]),
        ),
        knext.Effect.SHOW,
    )

    prompt_column = knext.ColumnParameter(
        "Prompt column",
        "Column containing prompts for the LLM.",
        port_index=1,
        column_filter=_message_cell_col_filter,
    )

    response_column_name = knext.StringParameter(
        "Response column name",
        "Name for the column holding the LLM's responses.",
        default_value="Response",
    )

    missing_value_handling = knext.EnumParameter(
        "If there are missing values",
        """Define whether missing or empty values in the prompt column and system message
        column (only applicable if the system message is provided via a column) should 
        result in missing values in the output table or whether the 
        node execution should fail on such values.""",
        default_value=lambda v: (
            util.MissingValueOutputOptions.Fail.name
            if v < knext.Version(5, 5, 0)
            else util.MissingValueOutputOptions.OutputMissingValues.name
        ),
        enum=util.MissingValueOutputOptions,
        style=knext.EnumParameter.Style.VALUE_SWITCH,
        since_version="5.5.0",
    )

    output_format = _get_output_format_value_switch()

    async def aprocess_batch(
        self,
        llm,
        sub_batch: List[str],
        progress_tracker: Callable[[int], None],
    ):
        from langchain_core.language_models import BaseChatModel, BaseLanguageModel

        llm: BaseLanguageModel = llm
        responses = await llm.abatch(sub_batch)
        if isinstance(llm, BaseChatModel):
            # chat models return AIMessage, therefore content field of the response has to be extracted
            responses = [response.content for response in responses]
        if progress_tracker:
            progress_tracker.update_progress(len(sub_batch))
        return responses

    def process_batches_concurrently(
        self,
        prompts: List[str] | List[List],
        llm,
        n_requests: int,
        progress_tracker: Callable[[int], None],
    ):
        import asyncio

        func = partial(self.aprocess_batch, llm, progress_tracker=progress_tracker)
        return asyncio.run(util.abatched_apply(func, prompts, n_requests))

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        llm_spec: LLMPortObjectSpec,
        input_table_spec: knext.Schema,
    ):
        if self.prompt_column:
            util.check_column(
                input_table_spec,
                self.prompt_column,
                [knext.string(), knext.logical(dict), util.message_type()],
                "prompt",
            )
        else:
            self.prompt_column = util.pick_default_column(
                input_table_spec, knext.string()
            )

        if not llm_spec.is_instruct_model:
            if (
                self.system_message_handling == SystemMessageHandling.SINGLE.name
                and self.system_message == ""
            ):
                raise knext.InvalidParametersError(
                    "The system message must not be empty."
                )

            if self.system_message_handling == SystemMessageHandling.COLUMN.name:
                if self.system_message_column:
                    util.check_column(
                        input_table_spec,
                        self.system_message_column,
                        knext.string(),
                        "system message",
                    )
                else:
                    spec_without_prompts = knext.Schema.from_columns(
                        [c for c in input_table_spec if c.name != self.prompt_column]
                    )
                    try:
                        self.system_message_column = util.pick_default_column(
                            spec_without_prompts, knext.string()
                        )
                    except Exception:
                        raise knext.InvalidParametersError(
                            "When using system messages from a column, the input table must contain at least "
                            "two string columns. One for the system messages and one for the prompts."
                        )

                if self.prompt_column == self.system_message_column:
                    raise knext.InvalidParametersError(
                        "The prompt and system message column can not be the same."
                    )

        llm_spec.validate_context(ctx)

        if not self.response_column_name:
            raise knext.InvalidParametersError(
                "The response column name must not be empty."
            )

        output_column_name = util.handle_column_name_collision(
            input_table_spec.column_names, self.response_column_name
        )

        return input_table_spec.append(
            knext.Column(ktype=knext.string(), name=output_column_name)
        )

    def execute(
        self,
        ctx: knext.ExecutionContext,
        llm_port: LLMPortObject,
        input_table: knext.Table,
    ):
        import pyarrow as pa

        # Output rows with missing values if "Output Missing Values" option is selected
        # or fail execution if "Fail" is selected and there are missing values
        missing_value_handling_setting = util.MissingValueOutputOptions[
            self.missing_value_handling
        ]
        num_rows = input_table.num_rows

        output_column_name = util.handle_column_name_collision(
            input_table.schema.column_names, self.response_column_name
        )

        output_table: knext.BatchOutputTable = knext.BatchOutputTable.create()

        n_requests = llm_port.spec.n_requests
        progress_tracker = util.ProgressTracker(total_rows=num_rows, ctx=ctx)
        llm = _initialize_model(llm_port, ctx, self.output_format)
        is_chat_model = not llm_port.spec.is_instruct_model

        def _call_model(messages: list) -> list:
            def get_responses(model):
                return self.process_batches_concurrently(
                    messages,
                    model,
                    n_requests,
                    progress_tracker,
                )

            return _call_model_with_output_format_fallback(
                ctx, llm_port, get_responses, self.output_format, llm
            )

        def apply_model(prompt_table: pa.Table):
            messages = self._extract_messages(prompt_table, is_chat_model)
            output_schema = pa.schema([(self.response_column_name, pa.string())])

            # Check if all prompts are empty/missing
            if not messages:
                missing_values_table = pa.table(
                    {self.response_column_name: [None] * len(messages)},
                    schema=output_schema,
                )
                return missing_values_table

            _validate_json_output_format(self.output_format, messages)

            responses = _call_model(messages)
            response_table = pa.table({output_column_name: responses})
            return response_table

        mapper = self._get_mapper(missing_value_handling_setting, apply_model)

        for batch in input_table.batches():
            util.check_canceled(ctx)
            pa_table = batch.to_pyarrow()

            prompt_table = pa_table

            table_from_batch = pa.Table.from_batches([prompt_table])
            response_table = mapper.map(table_from_batch)

            table_from_batch = pa.Table.from_arrays(
                pa_table.columns + response_table.columns,
                pa_table.column_names + response_table.column_names,
            )
            output_table.append(knext.Table.from_pyarrow(table_from_batch))

        if mapper.all_missing:
            ctx.set_warning("All rows contain missing or empty values.")

        return output_table

    def _get_mapper(self, missing_value_handling, apply_model):
        """
        Returns the appropriate mapper based on the selected missing value handling option.
        """

        input_columns = [self.prompt_column]
        if self.system_message_column:
            input_columns.append(self.system_message_column)

        if missing_value_handling == util.MissingValueOutputOptions.Fail:
            return util.FailOnMissingMapper(columns=input_columns, fn=apply_model)
        elif (
            missing_value_handling == util.MissingValueOutputOptions.OutputMissingValues
        ):
            return util.OutputMissingMapper(
                columns=input_columns,
                fn=apply_model,
            )

    def _is_chat_model(self, llm) -> bool:
        from langchain_core.language_models import BaseChatModel

        return isinstance(llm, BaseChatModel)

    def _extract_messages(self, table, is_chat_model: bool):
        system_messages = self._extract_system_messages(table, is_chat_model)
        user_messages = self._extract_user_messages(table)

        if system_messages is None:
            return [[user_message] for user_message in user_messages]
        else:
            return [
                [system_message, user_message]
                for system_message, user_message in zip(system_messages, user_messages)
            ]

    def _extract_user_messages(self, table):
        prompt_column = table.column(self.prompt_column)
        # TODO make sure this is always a human message
        return [util.to_human_message(prompt.as_py()) for prompt in prompt_column]

    def _extract_system_messages(self, table, is_chat_model: bool):
        import langchain_core.messages as lcm

        if (
            not is_chat_model
            or self.system_message_handling == SystemMessageHandling.NONE.name
        ):
            return None
        elif self.system_message_handling == SystemMessageHandling.SINGLE.name:
            system_message = lcm.SystemMessage(self.system_message)
            return [system_message] * len(table)
        elif self.system_message_handling == SystemMessageHandling.COLUMN.name:
            system_column = table.column(self.system_message_column)
            return [lcm.SystemMessage(val.as_py()) for val in system_column]
        else:
            raise NotImplementedError(
                f"System messages handled via '{SystemMessageHandling[self.system_message_handling].label}' are not implemented yet."
            )


# region Chat Model Prompter (deprecated)
@knext.node(
    "Chat Model Prompter",
    knext.NodeType.PREDICTOR,
    "icons/generic/brain.png",
    model_category,
    keywords=[
        "GenAI",
        "Gen AI",
        "Generative AI",
        "Large Language Model",
    ],
    is_deprecated=True,
)
@knext.input_port("Chat Model", "A chat model.", chat_model_port_type)
@knext.input_table(
    "Conversation History",
    "A table containing the conversation history, or an empty table.",
)
@knext.input_table(
    "Tool Definitions",
    "An optional table providing a set of tools the model can decide to call.",
    optional=True,
)
@knext.output_table(
    "Updated Conversation",
    "A table containing either the extended conversation history, or only the new messages, depending on the configuration.",
)
class ChatModelPrompter:
    """
    Interact with a chat model within a continuous conversation.

    This node prompts a chat model using the provided user message, using an existing conversation history as context.
    An optional table containing tool definitions can be provided to enable tool calling.

    **Conversation history** is a table containing two columns:

    - **Role column**: Indicates the sender of the message (e.g., 'human', 'ai', or 'tool').
    - **Message column**: Contains the content of the message.

    If the conversation history table is non-empty, it will be used as context when sending the new message to the chat model.
    To use only the conversation history table for prompting (without a new message), leave the new message setting empty and
    ensure that the last entry in the table has the 'human' role.

    In order to enable **tool calling**, a table containing tool definitions must be connected to the dynamic
    input port of the node. If tool definitions are provided, the conversation history table must include the
    following columns to support tool calling (*these columns will be populated by the chat model*):

    - **Tool name column**
    - **Tool call ID column**
    - **Tool call arguments column**

    If the chat model decides to call a tool, the node appends a new 'ai' message with the above columns populated based on the selected tool.
    This information can then be used to route the downstream portion of the workflow appropriately.
    The output of the tool can then be fed back into the node by appending a new 'tool' message to the
    conversation history table, with the tool's output being the message content.

    A common way to ensure that the tool call output is presented back to the **LLM Prompter (Conversation)** node is
    to embed the node together with its tools in a [Recursive Loop](https://hub.knime.com/knime/extensions/org.knime.features.base/latest/org.knime.base.node.meta.looper.recursive.RecursiveLoopStartDynamicNodeFactory).

    A **tool definition** is a JSON object describing the corresponding tool and its parameters. The more
    descriptive the definition, the more likely the LLM will call it appropriately.

    Example:

    ```
    {
        "title": "number_adder",
        "type": "object",
        "description": "Adds two numbers.",
        "properties": {
            "a": {
                "title": "A",
                "type": "integer",
                "description": "First value to add"
            },
            "b": {
                "title": "B",
                "type": "integer",
                "description": "Second value to add"
            }
        },
        "required": ["a", "b"]
    }
    ```

    ---

    **Note**: If you use the
    [Credentials Configuration node](https://hub.knime.com/knime/extensions/org.knime.features.js.quickforms/latest/org.knime.js.base.node.configuration.input.credentials.CredentialsDialogNodeFactory)
    and do not select the "Save password in configuration (weakly encrypted)" option for passing the API key for the LLM Selector node,
    the Credentials Configuration node will need to be reconfigured upon reopening the workflow, as the credentials flow variable
    was not saved and will therefore not be available to downstream nodes.
    """

    system_message = knext.MultilineStringParameter(
        "System message",
        """
        Optional instructional message provided to the model at the start of the conversation,
        usually used to define various guidelines and rules for the model to adhere to.

        **Note**: Certain models don't support system messages (e.g. OpenAI's o1-mini).
        For such models, the system message should be left empty.

        **Example**: You are an expert in geospacial analytics, and you only reply using JSON.
        """,
        default_value="",
    )

    chat_message = knext.MultilineStringParameter(
        "New message",
        "Optional next message to prompt the chat model with. If provided, a corresponding row will be appended to the conversation table.",
        default_value="",
    )

    output_format = _get_output_format_value_switch()

    conversation_settings = ToolChatConversationSettings()

    extend_existing_conversation = knext.BoolParameter(
        "Extend existing conversation",
        """If enabled, messages produced by this node will be appended to the provided conversation table.

         Otherwise, the output table will only contain the new message and the model's reply.
         
         **Note**: If an existing conversation is provided, it will still be used as context if this setting is disabled.""",
        default_value=True,
        since_version="5.4.3",
        is_advanced=True,
    )

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        chat_model_spec: ChatModelPortObjectSpec,
        input_table_spec: knext.Schema,
        tool_table_spec: Optional[knext.Schema],
    ) -> Schema:
        self.conversation_settings.configure(input_table_spec, tool_table_spec)

        _validate_json_output_format(
            self.output_format, [self.system_message + self.chat_message]
        )

        if chat_model_spec.is_instruct_model:
            raise knext.InvalidParametersError(
                "The selected model is not a chat model. Try selecting a different model."
            )

        chat_model_spec.validate_context(ctx)
        has_tools = tool_table_spec is not None

        if has_tools and not chat_model_spec.supports_tools:
            raise knext.InvalidParametersError(
                "The selected model does not support tool calling."
            )
        return self._create_output_schema(has_tools)

    def _create_output_schema(self, has_tool_input: bool) -> Schema:
        columns = [
            knext.Column(knext.string(), self.conversation_settings.role_column),
            knext.Column(knext.string(), self.conversation_settings.content_column),
        ]
        if has_tool_input:
            columns += [
                knext.Column(knext.string(), self._tool_name_column),
                knext.Column(knext.string(), self._tool_call_id_column),
                knext.Column(
                    knext.logical(dict),
                    self._tool_call_arguments_column,
                ),
            ]

        return knext.Schema.from_columns(columns)

    @property
    def _tool_name_column(self):
        return self._replace_none_column(
            self.conversation_settings.tool_calling_settings.tool_name_column,
            "tool_name",
        )

    @property
    def _tool_call_id_column(self):
        return self._replace_none_column(
            self.conversation_settings.tool_calling_settings.tool_call_id_column,
            "tool_call_id",
        )

    @property
    def _tool_call_arguments_column(self):
        return self._replace_none_column(
            self.conversation_settings.tool_calling_settings.tool_call_arguments_column,
            "tool_call_arguments",
        )

    def _replace_none_column(self, column, default_name):
        return default_name if column == knext.ColumnParameter.NONE else column

    def _needed_columns(self, has_tool_input: bool):
        columns = [
            self.conversation_settings.role_column,
            self.conversation_settings.content_column,
        ]
        if has_tool_input:
            columns += [
                self._tool_name_column,
                self._tool_call_id_column,
                self._tool_call_arguments_column,
            ]
        return columns

    def execute(
        self,
        ctx: knext.ExecutionContext,
        chat_model: ChatModelPortObject,
        input_table: knext.Table,
        tool_table: Optional[knext.Table],
    ):
        import pandas as pd

        has_tool_input = tool_table is not None
        data_frame: pd.DataFrame = input_table[
            self._needed_columns(has_tool_input)
        ].to_pandas()
        conversation_messages, human_message = self._construct_conversation(data_frame)
        chat = self.initialize_chat_model(ctx, chat_model, tool_table)
        answer = self._invoke_chat_model(chat, conversation_messages)
        response_df = self.create_response_dataframe(
            has_tool_input, human_message, answer
        )

        # conversation history empty
        if data_frame.empty:
            return knext.Table.from_pandas(response_df)

        if not self.extend_existing_conversation:
            return knext.Table.from_pandas(response_df)

        data_frame = pd.concat([data_frame, response_df], ignore_index=True)
        return knext.Table.from_pandas(data_frame)

    def _construct_conversation(self, data_frame):
        import langchain_core.messages as lcm

        conversation_messages = []
        if self.system_message:
            conversation_messages.append(lcm.SystemMessage(content=self.system_message))
        last_message_is_tool = False
        if len(data_frame) > 0:
            conversation_messages += self.conversation_settings.create_messages(
                data_frame
            )

            last_message_is_tool = (
                self.conversation_settings.tool_calling_settings.ignore_chat_message_if_last_message_is_tool
                and (
                    data_frame[self.conversation_settings.role_column].iloc[-1]
                    == "tool"
                )
            )
        human_message = None
        if (
            self.chat_message and not last_message_is_tool
        ):  # if the last message is a tool, the chat message is not added
            human_message = lcm.HumanMessage(content=self.chat_message)
            conversation_messages.append(human_message)

        conversation_messages = self._collapse_subsequent_ai_messages(
            conversation_messages
        )

        return conversation_messages, human_message

    def _invoke_chat_model(self, chat, conversation_messages):
        try:
            return chat.invoke(conversation_messages)
        except Exception as e:
            if getattr(e, "param", None) == "messages[0].role" and "system" in str(e):
                raise ValueError(
                    """The selected model does not support system messages. 
                    Please ensure that the model you are using supports system messages 
                    or remove the system message from the configuration."""
                )
            raise e

    def initialize_chat_model(self, ctx, chat_model, tool_table):
        from langchain_core.language_models import BaseChatModel
        import pandas as pd

        chat: BaseChatModel = _initialize_model(chat_model, ctx, self.output_format)
        if tool_table is not None:
            tool_data_frame: pd.DataFrame = tool_table.to_pandas()
            tools = tool_data_frame[
                self.conversation_settings.tool_calling_settings.tool_definition_column
            ].tolist()

            if any(tool is None for tool in tools):
                raise ValueError("There is a missing value in the tool table.")

            for tool_str in tools:
                _assert_tool_title_openai_compatibility(tool_str)

            chat = chat.bind_tools(tools)
        return chat

    def create_response_dataframe(self, has_tool_input, human_message, answer):
        import pandas as pd

        rows = []
        if human_message is not None:
            rows.append(self._create_row(human_message, has_tool_input))
        for tool_call in answer.tool_calls:
            rows.append(self._create_row(answer, has_tool_input, tool_call))
        if len(answer.tool_calls) == 0:
            rows.append(self._create_row(answer, has_tool_input))

        response_df = pd.DataFrame(rows)
        response_df = response_df.astype(self._get_output_pandas_types(has_tool_input))
        return response_df

    def _get_output_pandas_types(self, has_tool_input: bool) -> dict:
        types = {
            self.conversation_settings.role_column: "string",
            self.conversation_settings.content_column: "string",
        }
        if has_tool_input:
            types[self._tool_name_column] = "string"
            types[self._tool_call_id_column] = "string"
            types[self._tool_call_arguments_column] = knext.logical(dict).to_pandas()
        return types

    def _create_row(
        self, message, include_tool_columns: bool, tool_call: Optional[dict] = None
    ) -> dict:
        row = {
            self.conversation_settings.role_column: message.type,
            self.conversation_settings.content_column: message.content,
        }
        if include_tool_columns:
            if tool_call is None:
                tool_call = {}
            row[self._tool_name_column] = tool_call.get("name")
            row[self._tool_call_id_column] = tool_call.get("id")
            row[self._tool_call_arguments_column] = tool_call.get("args")
        return row

    def _collapse_subsequent_ai_messages(self, messages: list) -> list:
        import langchain_core.messages as lcm

        if len(messages) == 0:
            return []

        collapsed_messages = [messages[0]]
        for message in messages[1:]:
            if (
                isinstance(message, lcm.AIMessage)
                and isinstance(collapsed_messages[-1], lcm.AIMessage)
                and message.content == collapsed_messages[-1].content
            ):
                collapsed_messages[-1].tool_calls += message.tool_calls
            else:
                collapsed_messages.append(message)
        return collapsed_messages


def _call_model_with_output_format_fallback(
    ctx, model_port, response_func, output_format, model=None
):
    import openai

    try:
        if model is None:
            model = _initialize_model(model_port, ctx, output_format)
        return response_func(model)
    except openai.BadRequestError as e:
        if "Invalid parameter: 'response_format'" in str(e):
            ctx.set_warning(
                f"""The selected model does not support the output format '{output_format}', 
                'Text' mode is used as an output format instead."""
            )
            model = _initialize_model(model_port, ctx, OutputFormatOptions.Text.name)
            return response_func(model)
        raise e


def _initialize_model(llm_port, ctx, output_format=OutputFormatOptions.Text.name):
    # string to enum object mapping is used here since the value switch selection returns a string
    output_format = OutputFormatOptions[output_format]

    if output_format not in llm_port.spec.supported_output_formats:
        output_format = OutputFormatOptions.Text
    return llm_port.create_model(ctx, output_format)


def _string_col_filter(column: knext.Column):
    return column.ktype == knext.string()


def _contains_json_keyword(messages: list) -> bool:
    """
    Checks if 'json' keyword appears in messages.

    Messages can be one of:

        For LLM Prompter (checked in execute):
            - prompts ([list of strings])
            - system_message + prompt ([SystemMessage, HumanMessage])

        For LLM Chat Prompter (checked in configure):
            - system_message and/or chat_message (str)

    """
    for message in messages:
        if isinstance(message, str):
            if "json" not in message.lower():
                return False
        elif isinstance(message, list):
            combined_text = ""
            for item in message:
                if isinstance(item.content, str):
                    combined_text += f" {item.content}"
                elif isinstance(item.content, list):
                    # multi-part Message content
                    combined_text += " ".join(
                        content_part["text"]
                        for content_part in item.content
                        if content_part["type"] == "text"
                    )

            if "json" not in combined_text.lower():
                return False
    return True


def _validate_json_output_format(output_format: str, messages) -> None:
    """
    Validates that messages contain the word 'JSON' when JSON output format is selected.
    """
    if output_format != OutputFormatOptions.JSON.name:
        return

    if not _contains_json_keyword(messages):
        raise ValueError(
            """When requesting JSON output, the word 'JSON' must appear in either the system message, chat message, or prompt."""
        )


# region Text Embedder
@knext.node(
    "Text Embedder",
    knext.NodeType.PREDICTOR,
    util.ai_icon,
    model_category,
    keywords=[
        "GenAI",
        "Gen AI",
        "Generative AI",
        "Embeddings",
        "Vector",
        "RAG",
        "Retrieval Augmented Generation",
    ],
)
@knext.input_port(
    "Embedding Model",
    "Used to embed the texts from the input table into numerical vectors.",
    embeddings_model_port_type,
)
@knext.input_table("Input Table", "Input table containing a text column to embed.")
@knext.output_table(
    "Output Table", "The input table with the appended embeddings column."
)
class TextEmbedder:
    """
    Embeds text in a string column using an embedding model.

    This node applies the provided embedding model to create embeddings of the texts contained in a string column of the input table.

    A *text embedding* is a dense vector of floating point values capturing the semantic meaning of the text by mapping it to a high-dimensional space.
    Similarities between embedded entities are then derived by how close they are to each other in said space. These embeddings are often used to find
    semantically similar documents e.g. in vector stores.

    Different embedding models encode text differently, resulting in incomparable embeddings. If this node fails to execute with
    'Execute failed: Error while sending a command.', refer to the description of the node that provided the embedding model.

    **Note**: If you use the
    [Credentials Configuration node](https://hub.knime.com/knime/extensions/org.knime.features.js.quickforms/latest/org.knime.js.base.node.configuration.input.credentials.CredentialsDialogNodeFactory)
    and do not select the "Save password in configuration (weakly encrypted)" option for passing the API key for the Embedding Model Selector node,
    the Credentials Configuration node will need to be reconfigured upon reopening the workflow, as the credentials flow variable
    was not saved and will therefore not be available to downstream nodes.
    """

    text_column = knext.ColumnParameter(
        "Text column",
        "The string column containing the texts to embed.",
        port_index=1,
        column_filter=_string_col_filter,
    )

    embeddings_column_name = knext.StringParameter(
        "Embeddings column name",
        "Name for output column that will hold the embeddings.",
        "Embeddings",
    )

    missing_value_handling = knext.EnumParameter(
        "If there are missing values in the text column",
        """Define whether missing or empty values in the text column should 
        result in missing values in the output table or whether the 
        node execution should fail on such values.""",
        default_value=lambda v: (
            util.MissingValueOutputOptions.Fail.name
            if v < knext.Version(5, 3, 0)
            else util.MissingValueOutputOptions.OutputMissingValues.name
        ),
        enum=util.MissingValueOutputOptions,
        style=knext.EnumParameter.Style.VALUE_SWITCH,
        since_version="5.3.0",
    )

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        embeddings_spec: EmbeddingsPortObjectSpec,
        table_spec: knext.Schema,
    ) -> knext.Schema:
        if self.text_column is None:
            self.text_column = util.pick_default_column(table_spec, knext.string())
        else:
            util.check_column(
                table_spec, self.text_column, knext.string(), "text column"
            )

        embeddings_spec.validate_context(ctx)
        output_column_name = util.handle_column_name_collision(
            table_spec.column_names, self.embeddings_column_name
        )
        return table_spec.append(self._create_output_column(output_column_name))

    def _create_output_column(self, output_column_name) -> knext.Column:
        return knext.Column(knext.list_(inner_type=knext.double()), output_column_name)

    def execute(
        self,
        ctx: knext.ExecutionContext,
        embeddings_obj: EmbeddingsPortObject,
        table: knext.Table,
    ) -> knext.Table:
        import pyarrow as pa

        # Output rows with missing values if "Output Missing Values" option is selected
        # or fail execution if "Fail" is selected and there are missing values
        missing_value_handling_setting = util.MissingValueOutputOptions[
            self.missing_value_handling
        ]

        embeddings_model = embeddings_obj.create_model(ctx)
        output_table = knext.BatchOutputTable.create()
        num_rows = table.num_rows

        if num_rows == 0:
            output_columns = [
                util.OutputColumn(
                    self.embeddings_column_name,
                    knext.list_(knext.double()),
                    pa.list_(pa.float64()),
                )
            ]
            return util.create_empty_table(
                table,
                output_columns,
            )

        i = 0
        output_column_name = util.handle_column_name_collision(
            table.schema.column_names, self.embeddings_column_name
        )

        output_column_field = pa.field(output_column_name, pa.list_(pa.float64()))

        adapter_fn = partial(
            util.table_column_adapter,
            fn=embeddings_model.embed_documents,
            input_column=self.text_column,
            output_column=output_column_field,
        )

        if missing_value_handling_setting == util.MissingValueOutputOptions.Fail:
            mapper = util.FailOnMissingMapper(self.text_column, adapter_fn)
        else:
            mapper = util.OutputMissingMapper(
                columns=self.text_column,
                fn=adapter_fn,
            )

        for batch in table.batches():
            util.check_canceled(ctx)
            pa_table = batch.to_pyarrow()
            table_from_batch = pa.Table.from_batches([pa_table])
            embeddings_array = mapper.map(table_from_batch)
            table_from_batch = pa.Table.from_arrays(
                table_from_batch.columns + embeddings_array.columns,
                table_from_batch.column_names + embeddings_array.column_names,
            )

            output_table.append(knext.Table.from_pyarrow(table_from_batch))

            i += batch.num_rows
            ctx.set_progress(i / num_rows)

        if mapper.all_missing:
            ctx.set_warning("All rows contain missing or empty values.")

        return output_table
