import knime.api.types as kt
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any
import json


class MessageType(Enum):
    USER = "USER"
    TOOL = "TOOL"
    AI = "AI"


class MessageContentPartType(Enum):
    TEXT = "text"
    PNG = "png"


@dataclass
class MessageContentPart:
    """
    Represents a part of the message content.
    """

    type: MessageContentPartType  # e.g., "text", "png", etc.
    data: bytes  # the data of the content part as a byte array


@dataclass
class ToolCall:
    """
    Represents a tool call within an AI message.
    """

    tool_name: str
    id: str
    arguments: Dict[str, Any]  # JSON as dict


@dataclass
class MessageValue:
    """
    Represents a message to or from an AI model.
    """

    message_type: MessageType
    content: List[MessageContentPart]
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class MessageValueFactory(kt.PythonValueFactory):
    """
    Factory for creating MessageValue instances.
    """

    def __init__(self):
        super().__init__(MessageValue)

    def decode(self, storage):
        if storage is None:
            return None

        # storage: dict with keys "0", "1", "2", "3", "4"
        message_type = MessageType(storage["0"])
        content = [
            MessageContentPart(type=MessageContentPartType(part["0"]), data=part["1"])
            for part in storage["1"]
        ]

        tool_calls = None
        if storage["2"] is not None:
            tool_calls = [
                ToolCall(
                    tool_name=tc["1"],
                    id=tc["0"],
                    arguments=json.loads(tc["2"])
                    if isinstance(tc["2"], str)
                    else tc["2"],
                )
                for tc in storage["2"]
            ]
        tool_call_id = storage.get("3")
        name = storage.get("4")
        return MessageValue(
            message_type=message_type,
            content=content,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
            name=name,
        )

    def encode(self, value: "MessageValue"):
        # handle missing values
        if value is None:
            return None

        storage = {
            "0": value.message_type.value,
            "1": [
                {
                    "0": part.type.value,
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
                    "2": json.dumps(tc.arguments)
                    if not isinstance(tc.arguments, str)
                    else tc.arguments,
                }
                for tc in value.tool_calls
            ]
        else:
            storage["2"] = None
        storage["3"] = value.tool_call_id
        storage["4"] = value.name
        return storage


def to_langchain_message(msg: "MessageValue"):
    """
    Convert a MessageValue to a langchain_core.messages.BaseMessage instance.
    """
    from langchain_core.messages.human import HumanMessage
    from langchain_core.messages.ai import AIMessage
    from langchain_core.messages.tool import ToolMessage

    # Convert content parts to string or dict as appropriate
    def _convert_content(content_parts):
        def _decode_safely(text_bytes):
            from charset_normalizer import from_bytes

            best_match = from_bytes(text_bytes).best()
            if best_match:
                return str(best_match)

            return text_bytes.decode("latin-1", errors="replace")

        # If all parts are type "text", join as string, else use list of dicts
        if not content_parts:
            return ""

        if all(part.type == "text" for part in content_parts):
            return "".join(
                _decode_safely(part.data)
                if isinstance(part.data, bytes)
                else str(part.data)
                for part in content_parts
            )

        # Otherwise, return as list of dicts
        result = []
        for part in content_parts:
            if part.type == MessageContentPartType.TEXT:
                result.append(
                    {
                        "type": "text",
                        "text": _decode_safely(part.data)
                        if isinstance(part.data, bytes)
                        else str(part.data),
                    }
                )
            elif part.type == MessageContentPartType.PNG:
                if isinstance(part.data, bytes):
                    import base64

                    base64_image = base64.b64encode(part.data).decode("utf-8")
                    image_url = f"data:image/png;base64,{base64_image}"
                    result.append(
                        {"type": "image_url", "image_url": {"url": image_url}}
                    )
            else:
                result.append({"type": part.type.value, "data": part.data})
        return result

    content = _convert_content(msg.content)

    if msg.message_type == MessageType.USER:
        return HumanMessage(content=content, name=msg.name)
    elif msg.message_type == MessageType.AI:
        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(
                    {
                        "name": tc.tool_name,
                        "args": tc.arguments,
                        "id": tc.id,
                        "type": "tool_call",
                    }
                )
        return AIMessage(
            content=content, tool_calls=tool_calls, id=msg.tool_call_id, name=msg.name
        )
    elif msg.message_type == MessageType.TOOL:
        return ToolMessage(
            content=content, tool_call_id=msg.tool_call_id, name=msg.name
        )
    else:
        raise ValueError(f"Unknown MessageType: {msg.message_type}")


def from_langchain_message(lc_msg) -> MessageValue:
    """
    Convert a langchain_core.messages.BaseMessage instance to a MessageValue.
    """
    from langchain_core.messages.human import HumanMessage
    from langchain_core.messages.ai import AIMessage
    from langchain_core.messages.tool import ToolMessage

    # Helper to convert content to MessageContentPart list
    def _to_content_parts(content):
        parts = []
        if isinstance(content, str):
            parts.append(
                MessageContentPart(
                    type=MessageContentPartType.TEXT, data=content.encode("utf-8")
                )
            )
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, str):
                    parts.append(
                        MessageContentPart(
                            type=MessageContentPartType.TEXT, data=part.encode("utf-8")
                        )
                    )
                elif isinstance(part, dict):
                    if part.get("type") == "image_url":
                        # OpenAI format: {"type": "image_url", "image_url": {"url": ...}}
                        url = part.get("image_url", {}).get("url", b"")
                        if isinstance(url, str):
                            url = url.encode("utf-8")
                        parts.append(
                            MessageContentPart(
                                type=MessageContentPartType.PNG, data=url
                            )
                        )
                    elif part.get("type") == "text":
                        text = part.get("text", "")
                        parts.append(
                            MessageContentPart(
                                type=MessageContentPartType.TEXT,
                                data=text.encode("utf-8"),
                            )
                        )
                    else:
                        raise ValueError(
                            f"Unsupported message part type: {part.get('type')}"
                        )
        return parts

    name = getattr(lc_msg, "name", None)

    if isinstance(lc_msg, HumanMessage):
        msg_type = MessageType.USER
        content = _to_content_parts(lc_msg.content)
        return MessageValue(message_type=msg_type, content=content, name=name)
    elif isinstance(lc_msg, AIMessage):
        msg_type = MessageType.AI
        content = _to_content_parts(lc_msg.content)
        tool_calls = None
        if getattr(lc_msg, "tool_calls", None):
            tool_calls = [
                ToolCall(
                    tool_name=tc.get("name", ""),
                    id=tc.get("id", ""),
                    arguments=tc.get("args", {})
                    if isinstance(tc.get("args", {}), dict)
                    else json.loads(tc.get("args", "{}")),
                )
                for tc in lc_msg.tool_calls
            ]
        return MessageValue(
            message_type=msg_type,
            content=content,
            tool_calls=tool_calls,
            tool_call_id=getattr(lc_msg, "id", None),
            name=name,
        )
    elif isinstance(lc_msg, ToolMessage):
        msg_type = MessageType.TOOL
        content = _to_content_parts(lc_msg.content)
        return MessageValue(
            message_type=msg_type,
            content=content,
            tool_call_id=getattr(lc_msg, "tool_call_id", None),
            name=name,
        )
    else:
        raise ValueError(f"Unsupported langchain message type: {type(lc_msg)}")
