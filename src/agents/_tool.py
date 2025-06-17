from dataclasses import dataclass
import re
from ._data import Port, DataRegistry, port_to_dict
from ._common import render_structured
from langchain_core.messages import AIMessage, BaseMessage
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

    def desanitize_tool_calls(self, msg: BaseMessage) -> BaseMessage:
        """Desanitizes tool calls in a message by reverting the name back to the original user-provided name."""
        if isinstance(msg, AIMessage) and msg.tool_calls:
            # Tool calls can be either a dict or a ToolCall object, so we handle both cases
            for tool_call in msg.tool_calls:
                if isinstance(tool_call, dict):
                    tool_call["name"] = self.desanitize_tool_name(tool_call["name"])
                else:
                    tool_call.name = self.desanitize_tool_name(tool_call.name)
        return msg

    def to_langchain_tool(
        self,
        tool: WorkflowTool,
    ) -> StructuredTool:
        sanitized_name = self._sanitize_tool_name(tool.name)
        if tool.input_ports or tool.output_ports:
            self._has_data_tools = True
            return self._to_langchain_tool_with_data(tool, sanitized_name)
        else:
            return self._to_langchain_tool_without_data(tool, sanitized_name)

    def _validate_required_fields(self, required_fields, provided_dict, error_prefix):
        missing = [field for field in required_fields if field not in provided_dict]
        if missing:
            raise knext.InvalidParametersError(
                f"Missing {error_prefix}: {', '.join(missing)}"
            )

    def _to_langchain_tool_without_data(
        self, tool: WorkflowTool, sanitized_name: str
    ) -> StructuredTool:
        args_schema = {
            "type": "object",
            "properties": tool.parameter_schema,
            "required": list(tool.parameter_schema.keys()),
        }

        def func(**params):
            try:
                self._validate_required_fields(
                    tool.parameter_schema.keys(), params, "configuration parameters"
                )
                return self._ctx._execute_tool(tool, params, [], self._debug)[0]
            except Exception as e:
                _logger.exception(e)
                raise

        return StructuredTool.from_function(
            func=func,
            name=sanitized_name,
            description=tool.description,
            args_schema=args_schema,
        )

    def _to_langchain_tool_with_data(
        self,
        tool: WorkflowTool,
        sanitized_name: str,
    ) -> StructuredTool:
        args_schema = {
            "type": "object",
            "properties": {
                "configuration": {
                    "type": "object",
                    "properties": tool.parameter_schema,
                    "description": "Configures the tool to perform the task at hand.",
                    "required": list(tool.parameter_schema.keys()),
                },
                "data_inputs": self._create_input_data_schema(tool),
            },
            "required": ["parameters", "data_inputs"],
        }

        input_ports = tool.input_ports if tool.input_ports else []
        output_ports = tool.output_ports if tool.output_ports else []

        description_with_data_info = render_structured(
            description=tool.description,
            input_ports=list(
                map(port_to_dict, input_ports)
            ),
            output_ports=list(
                map(port_to_dict, output_ports)
            ),
        )

        def func(configuration: dict = None, data_inputs: dict = None) -> str:
            configuration = configuration or {}
            data_inputs = data_inputs or {}

            # Unified configuration validation
            self._validate_required_fields(
                tool.parameter_schema.keys(), configuration, "configuration parameters"
            )
            # Unified data input validation
            self._validate_required_fields(
                [port.name for port in tool.input_ports],
                data_inputs,
                "data inputs for ports",
            )

            try:
                inputs = [self._data_registry.get_data(i) for i in data_inputs.values()]
                message, outputs = self._ctx._execute_tool(
                    tool, configuration, inputs, self._debug
                )
                if not outputs:
                    return message
                output_references = {}
                for port, output in zip(output_ports, outputs):
                    output_reference = self._data_registry.add_table(output, port)
                    output_references.update(output_reference)

                return render_structured(
                    message=message, outputs=output_references
                )
            except Exception as e:
                _logger.exception(e)
                raise

        return StructuredTool.from_function(
            func=func,
            name=sanitized_name,
            description=description_with_data_info,
            args_schema=args_schema,
        )

    def _create_input_data_schema(self, tool: WorkflowTool) -> knext.Schema:
        return {
            "type": "object",
            "description": "The input data the tool requires for the task at hand.",
            "properties": self._create_input_port_properties(tool.input_ports),
            "required": [port.name for port in tool.input_ports],
        }

    def _create_input_port_properties(self, ports: list[Port]):
        return {
            port.name: {
                "type": "integer",
                "description": "ID of the data to feed to the port for "
                + port.description,
            }
            for port in ports
        }