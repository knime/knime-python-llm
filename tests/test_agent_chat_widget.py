import unittest
from unittest.mock import patch
import sys
import types

import knime.extension as knext
import knime.extension.nodes as kn


class _MockBackend:
    def register_port_type(self, name, object_class, spec_class, id=None):
        if id is None:
            id = f"test.{object_class.__module__}.{object_class.__qualname__}"
        return kn.PortType(id, name, object_class, spec_class)


knime_types_module = types.ModuleType("knime.types")
knime_types_tool_module = types.ModuleType("knime.types.tool")
knime_types_tool_module.WorkflowTool = type("WorkflowTool", (), {})
knime_types_message_module = types.ModuleType("knime.types.message")
knime_types_message_module.MessageValue = type("MessageValue", (), {})


def setUpModule():
    """Set up patched KNIME backend and types for this test module."""
    global _backend_patcher
    global _logical_patcher
    global _sysmodules_patcher
    global _inactiveport_present
    global _inactiveport_original
    global AgentChatWidget

    _backend_patcher = patch.object(kn, "_backend", _MockBackend())
    _backend_patcher.start()

    _logical_patcher = patch.object(
        knext, "logical", lambda value_type: knext.string()
    )
    _logical_patcher.start()

    _inactiveport_present = hasattr(knext, "InactivePort")
    _inactiveport_original = getattr(knext, "InactivePort", None)
    knext.InactivePort = object()

    _sysmodules_patcher = patch.dict(
        sys.modules,
        {
            "knime.types": knime_types_module,
            "knime.types.tool": knime_types_tool_module,
            "knime.types.message": knime_types_message_module,
        },
        clear=False,
    )
    _sysmodules_patcher.start()

    from agents.base import AgentChatWidget as _AgentChatWidget

    AgentChatWidget = _AgentChatWidget


def tearDownModule():
    """Tear down patched KNIME backend and types for this test module."""
    _backend_patcher.stop()
    _logical_patcher.stop()

    if _inactiveport_present:
        knext.InactivePort = _inactiveport_original
    else:
        delattr(knext, "InactivePort")

   _sysmodules_patcher.stop()
class _MockContext:
    def __init__(self, view_data, num_data_outputs, combined_tools_workflow):
        self._view_data_value = view_data
        self._num_data_outputs = num_data_outputs
        self._combined_tools_workflow_value = combined_tools_workflow

    def _get_view_data(self):
        return self._view_data_value

    def get_connected_output_port_numbers(self):
        return [1, 1, self._num_data_outputs]

    def _get_combined_tools_workflow(self):
        return self._combined_tools_workflow_value


class AgentChatWidgetOutputTest(unittest.TestCase):
    def test_execute_returns_inactive_ports_when_no_view_data_exists(self):
        node = AgentChatWidget()
        ctx = _MockContext(
            view_data=None,
            num_data_outputs=2,
            combined_tools_workflow="workflow-port",
        )

        combined_tools_workflow, conversation_output, data_outputs = node.execute(
            ctx, None, None, []
        )

        self.assertEqual("workflow-port", combined_tools_workflow)
        self.assertIs(knext.InactivePort, conversation_output)
        self.assertEqual([knext.InactivePort, knext.InactivePort], data_outputs)

    def test_execute_marks_missing_data_outputs_inactive(self):
        node = AgentChatWidget()
        conversation_table = object()
        data_table = object()
        ctx = _MockContext(
            view_data={
                "ports": [conversation_table],
                "data": {"data_registry": {"ids": [], "ports": []}},
                "ports_for_ids": [],
            },
            num_data_outputs=2,
            combined_tools_workflow="workflow-port",
        )

        class _MockDataRegistry:
            def get_last_tables(self, num_tables, fill_missing=True):
                self.args = (num_tables, fill_missing)
                return [data_table]

        registry = _MockDataRegistry()
        with patch("agents._data_service.DataRegistry.load", return_value=registry):
            combined_tools_workflow, conversation_output, data_outputs = node.execute(
                ctx, None, None, []
            )

        self.assertEqual("workflow-port", combined_tools_workflow)
        self.assertIs(conversation_table, conversation_output)
        self.assertEqual((2, False), registry.args)
        self.assertEqual(2, len(data_outputs))
        self.assertIs(data_table, data_outputs[0])
        self.assertIs(knext.InactivePort, data_outputs[1])
