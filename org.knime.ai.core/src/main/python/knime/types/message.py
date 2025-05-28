import knime.api.types as kt
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class MessageType(Enum):
    USER = "USER"
    TOOL = "TOOL"
    AI = "AI"


@dataclass
class MessageContentPart:
    """
    Represents a part of the message content.
    """

    type: str  # e.g., "text", "image", etc.
    data: bytes  # the data of the content part as a byte array


@dataclass
class ToolCall:
    """
    Represents a tool call within an AI message.
    """

    tool_name: str
    id: str
    arguments: str  # JSON format


@dataclass
class MessageValue:
    """
    Represents a message to or from an AI model.
    """

    message_type: MessageType
    content: List[MessageContentPart]
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None


class MessageValueFactory(kt.PythonValueFactory):
    """
    Factory for creating MessageValue instances.
    """

    def __init__(self):
        super().__init__(MessageValue)

    def decode(self, storage):
        # storage: dict with keys "0", "1", "2", "3"
        message_type = MessageType(storage["0"])
        content = [
            MessageContentPart(type=part["0"], data=part["1"]) for part in storage["1"]
        ]

        tool_calls = None
        if storage["2"] is not None:
            tool_calls = [
                ToolCall(
                    tool_name=tc["1"],
                    id=tc["0"],
                    arguments=tc["2"],
                )
                for tc in storage["2"]
            ]
        tool_call_id = storage.get("3")
        return MessageValue(
            message_type=message_type,
            content=content,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
        )

    def encode(self, value: MessageValue):
        # value: MessageValue
        storage = {
            "0": value.message_type.value,
            "1": [
                {
                    "0": part.type,
                    "1": part.data,
                }
                for part in value.content
            ],
        }
        if value.tool_calls is not None:
            storage["2"] = [
                {
                    "0": tc.id,
                    "1": tc.tool_name,
                    "2": tc.arguments,
                }
                for tc in value.tool_calls
            ]
        else:
            storage["2"] = None
        storage["3"] = value.tool_call_id
        return storage
