from dataclasses import dataclass
import re
from ._data import Port, DataRegistry, port_to_dict
from ._common import render_structured
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain.tools import StructuredTool
import knime.extension as knext
import logging

_logger = logging.getLogger(__name__)

@dataclass
class WorkflowTool:
    """Mirrors the tool class defined in knime-python so we can use type hints here."""

    name: str
    description: str
    parameter_schema: dict
    tool_bytes: bytes
    input_ports: list[Port]
    output_ports: list[Port]


class LangchainToolConverter:
    def __init__(
        self, data_registry: DataRegistry, ctx, debug: bool
    ):
        self._data_registry = data_registry
        self._ctx = ctx
        self._debug = debug
        self.sanitized_to_original = {}
        self._has_data_tools = False

    @property
    def has_data_tools(self) -> bool:
        """Returns True if there are tools that require data inputs or outputs."""
        return self._has_data_tools

    def _sanitize_tool_name(self, name: str) -> str:
        """Replaces characters that are not alphanumeric, underscores, or hyphens because
        OpenAI rejects the tools otherwise. Also handles duplicates by appending a suffix."""
        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
        # Handle duplicates by appending a suffix if needed
        base = sanitized
        i = 1
        while (
            sanitized in self.sanitized_to_original
            and self.sanitized_to_original[sanitized] != name
        ):
            sanitized = f"{base}_{i}"
            i += 1
        self.sanitized_to_original[sanitized] = name
        return sanitized

    def desanitize_tool_name(self, sanitized_name: str) -> str:
        return self.sanitized_to_original.get(sanitized_name, sanitized_name)

    def desanitize_tool_names(self, msg: BaseMessage) -> BaseMessage:
        """Desanitizes tool calls in a message by reverting the name back to the original user-provided name."""
        if isinstance(msg, AIMessage) and msg.tool_calls:
            # Tool calls can be either a dict or a ToolCall object, so we handle both cases
            for tool_call in msg.tool_calls:
                if isinstance(tool_call, dict):
                    tool_call["name"] = self.desanitize_tool_name(tool_call["name"])
                else:
                    tool_call.name = self.desanitize_tool_name(tool_call.name)
        elif isinstance(msg, ToolMessage) and msg.name:
            msg.name = self.desanitize_tool_name(msg.name)
        return msg

    def to_langchain_tool(
        self,
        tool: WorkflowTool,
    ) -> StructuredTool:
        sanitized_name = self._sanitize_tool_name(tool.name)
        properties = {}
        if tool.parameter_schema:
            properties["configuration"] = _create_configuration_schema(tool)
        description_parts = {"description": tool.description}

        if tool.input_ports:
            self._has_data_tools = True
            properties["data_inputs"] = _create_input_data_schema(tool)
            params_parser = self._create_data_input_parser(tool)
            description_parts["input_ports"] = list(
                map(port_to_dict, tool.input_ports)
            )
        else:
            params_parser = self._create_no_data_input_parser(tool)

        if tool.output_ports:
            self._has_data_tools = True
            outputs_processor = self._create_data_output_processor(tool)
            description_parts["output_ports"] = list(
                map(port_to_dict, tool.output_ports)
            )
        else:
            outputs_processor = self._create_no_data_output_processor()

        args_schema = _create_object_schema(
            description="Parameters for the tool.",
            properties=properties,
        )

        # Create the tool function
        def tool_function(**params: dict) -> str:
            try:
                configuration, inputs = params_parser(params)
                message, outputs = self._ctx._execute_tool(tool, configuration, inputs, self._debug)
                return outputs_processor(message, outputs)
            except Exception as e:
                _logger.exception(e)
                raise
        
        return StructuredTool.from_function(
            func=tool_function,
            name=sanitized_name,
            description=render_structured(**description_parts),
            args_schema=args_schema,
        )
    
    def _create_data_output_processor(self, tool: WorkflowTool):
        def _process_outputs_with_data(message: str, outputs: list[knext.Table]):
            output_references = {}
            for port, output in zip(tool.output_ports, outputs):
                output_reference = self._data_registry.add_table(output, port)
                output_references.update(output_reference)
            return message + "\n\n## Data repository update\n" + render_structured(
                **output_references
            )
        return _process_outputs_with_data

    def _create_no_data_output_processor(self):
        def _process_outputs_no_data(message: str, outputs: list[knext.Table]):
            return message
        return _process_outputs_no_data
    
    def _create_no_data_input_parser(self, tool: WorkflowTool):
        def _parse_params_no_data(params: dict):
            configuration = params.get("configuration", {})
            _validate_required_fields(tool.parameter_schema.keys(), configuration, "configuration parameters")
            return configuration, []
        return _parse_params_no_data
    
    def _create_data_input_parser(self, tool: WorkflowTool):
        def _parse_params_with_data(params: dict):
                configuration = params.get("configuration", {})
                data_inputs = params.get("data_inputs", {})

                _validate_required_fields(
                    tool.parameter_schema.keys(), configuration, "configuration parameters"
                )
                _validate_required_fields(
                    [port.name for port in tool.input_ports],
                    data_inputs,
                    "data inputs for ports",
                )
                inputs = [self._data_registry.get_data(i) for i in data_inputs.values()]
                return configuration, inputs
        return _parse_params_with_data
    
def _validate_required_fields(required_fields, provided_dict, error_prefix):
        missing = [field for field in required_fields if field not in provided_dict]
        if missing:
            raise knext.InvalidParametersError(
                f"Missing {error_prefix}: {', '.join(missing)}"
            )

def _create_object_schema(description: str, properties: dict):
    return {
        "type": "object",
        "description": description,
        "properties": properties,
        "required": list(properties.keys()),
        }

def _create_configuration_schema(tool: WorkflowTool) -> dict:
    """Creates a parameter schema for the tool based on its input ports."""
    return _create_object_schema(
        description="Configures the tool to perform the task at hand.",
        properties=tool.parameter_schema,
    )

def _create_input_data_schema(tool: WorkflowTool) -> knext.Schema:
    return _create_object_schema(description="Data inputs for the tool.",
        properties=_create_input_port_properties(tool.input_ports),
    )

def _create_input_port_properties(ports: list[Port]):
    return {
        port.name: {
            "type": "integer",
            "description": "ID of the data to feed to the port for "
            + port.description,
        }
        for port in ports
    }