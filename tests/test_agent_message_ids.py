import unittest

from langchain_core.messages import AIMessage, HumanMessage

from agents._conversation import AgentPrompterConversation
from agents._data_service import FrontendConversation


class _BackendConversation:
    def append_messages(self, _messages):
        pass

    def append_error(self, _error):
        pass

    def get_messages(self):
        return []

    def create_output_table(self, *_args, **_kwargs):
        return None


class _ToolConverter:
    def desanitize_tool_name(self, name):
        return name


class AgentMessageIdTests(unittest.TestCase):
    def test_conversation_appends_unique_ids_and_suffix(self):
        conversation = AgentPrompterConversation(error_handling="COLUMN")
        user_msg = HumanMessage(content="Hello")
        ai_msg = AIMessage(content="Hi there")

        conversation.append_messages([user_msg, ai_msg])

        self.assertEqual(user_msg.id, "msg-0001")
        self.assertEqual(ai_msg.id, "msg-0002")
        self.assertTrue(user_msg.content.endswith("[message_id: msg-0001]"))
        self.assertTrue(ai_msg.content.endswith("[message_id: msg-0002]"))

    def test_conversation_reuses_existing_message_id_suffix_once(self):
        conversation = AgentPrompterConversation(error_handling="COLUMN")
        msg = HumanMessage(content="From history\n\n[message_id: imported-42]")

        conversation.append_messages(msg)

        self.assertEqual(msg.id, "imported-42")
        self.assertEqual(msg.content.count("[message_id:"), 1)
        self.assertTrue(msg.content.endswith("[message_id: imported-42]"))

    def test_frontend_message_parser_keeps_suffix_when_view_ids_present(self):
        conversation = FrontendConversation(
            backend=_BackendConversation(),
            tool_converter=_ToolConverter(),
            check_canceled=lambda: False,
        )
        message = AIMessage(
            id="msg-0007",
            content="Result text\n\nView node IDs node-a,node-b\n\n[message_id: msg-0007]",
            tool_calls=[],
        )

        rendered = conversation._to_frontend_messages(message)

        self.assertEqual(rendered[0]["id"], "msg-0007")
        self.assertIn("[message_id: msg-0007]", rendered[0]["content"])
        self.assertEqual(rendered[1]["id"], "msg-0007-view-0")
        self.assertEqual(rendered[2]["id"], "msg-0007-view-1")


if __name__ == "__main__":
    unittest.main()
