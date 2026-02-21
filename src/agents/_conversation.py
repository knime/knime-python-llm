# -*- coding: utf-8 -*-

from typing import Optional
import re

import knime.extension as knext
import pandas as pd
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
try:
    from knime.types.message import MessageValue, from_langchain_message
except ModuleNotFoundError:
    MessageValue = None
    from_langchain_message = None

from ._agent import validate_ai_message, CancelError
from ._parameters import ErrorHandlingMode


def _message_type() -> knext.LogicalType:
    if MessageValue is None:
        raise ModuleNotFoundError("knime.types.message is required for message output")
    return knext.logical(MessageValue)


class AgentPrompterConversation:
    _MESSAGE_ID_PREFIX = "msg-"
    _MESSAGE_ID_SUFFIX_PATTERN = re.compile(r"\n\n\[message_id:\s*([^\]]+)\]\s*$")

    def __init__(self, error_handling: Optional[str], ctx: knext.ExecutionContext = None):
        self._error_handling = error_handling
        self._message_and_errors = []
        self._is_message = []
        self._ctx = ctx
        self._next_message_id = 1
        self._used_message_ids = set()

    def append_messages(self, messages):
        """Raises a CancelError if the context was canceled."""
        if isinstance(messages, BaseMessage):
            messages = [messages]

        if self._ctx and self._ctx.is_canceled():
            raise CancelError("Execution canceled.")

        for msg in messages:
            self._assign_message_id_and_suffix(
                msg, skip_suffix=isinstance(msg, SystemMessage)
            )
            if isinstance(msg, AIMessage):
                try:
                    validate_ai_message(msg)
                except Exception as e:
                    if self._error_handling == ErrorHandlingMode.FAIL.name:
                        if self._ctx:
                            self._ctx.set_warning(str(e))
                        continue
                    self._append(e)
                    continue
            self._append(msg)

    def _assign_message_id_and_suffix(self, message, skip_suffix: bool = False):
        content = getattr(message, "content", None)
        suffix_id = None
        if isinstance(content, str):
            match = self._MESSAGE_ID_SUFFIX_PATTERN.search(content)
            if match:
                suffix_id = match.group(1).strip()
                content = content[: match.start()]

        existing_id = getattr(message, "id", None)
        if existing_id is not None:
            existing_id = str(existing_id).strip()
            if not existing_id:
                existing_id = None

        if existing_id and existing_id not in self._used_message_ids:
            message_id = existing_id
        elif suffix_id and suffix_id not in self._used_message_ids:
            message_id = suffix_id
        else:
            message_id = self._new_message_id()

        message.id = message_id
        self._used_message_ids.add(message_id)

        if not skip_suffix and isinstance(content, str):
            message.content = f"{content}\n\n[message_id: {message_id}]"

    def _new_message_id(self) -> str:
        while True:
            message_id = f"{self._MESSAGE_ID_PREFIX}{self._next_message_id:04d}"
            self._next_message_id += 1
            if message_id not in self._used_message_ids:
                return message_id

    def append_error(self, error):
        if not isinstance(error, Exception):
            raise error

        if self._error_handling == ErrorHandlingMode.FAIL.name:
            raise error
        self._append(error)

    def get_messages(self):
        return [
            moe
            for is_msg, moe in zip(self._is_message, self._message_and_errors)
            if is_msg
        ]

    def _append(self, message_or_error):
        self._message_and_errors.append(message_or_error)
        self._is_message.append(isinstance(message_or_error, BaseMessage))

    def _construct_output(self):
        return [
            {"message": moe if is_msg else None, "error": moe if not is_msg else None}
            for is_msg, moe in zip(self._is_message, self._message_and_errors)
        ]

    def create_output_table(
        self,
        tool_converter,
        output_column_name: str,
        error_column_name: str = None,
    ) -> knext.Table:
        def to_knime_message_or_none(msg):
            if msg is None:
                return None
            if from_langchain_message is None:
                raise ModuleNotFoundError(
                    "knime.types.message is required for message output"
                )
            desanitized = tool_converter.desanitize_tool_names(msg)
            return from_langchain_message(desanitized)

        if error_column_name is None:
            messages = self.get_messages()
            if messages and isinstance(messages[0], SystemMessage):
                messages = messages[1:]
            result_df = pd.DataFrame(
                {output_column_name: [to_knime_message_or_none(msg) for msg in messages]}
            )
        else:
            messages_and_errors = [
                moe
                for moe in self._construct_output()
                if not isinstance(moe["message"], SystemMessage)
            ]
            messages = [to_knime_message_or_none(moe["message"]) for moe in messages_and_errors]
            errors = [
                str(moe["error"]) if moe["error"] is not None else None
                for moe in messages_and_errors
            ]
            result_df = pd.DataFrame(
                {output_column_name: messages, error_column_name: errors}
            )

            if not any(messages):
                result_df[output_column_name] = result_df[output_column_name].astype(
                    _message_type().to_pandas()
                )
            if not any(errors):
                result_df[error_column_name] = result_df[error_column_name].astype(
                    "string"
                )

        return knext.Table.from_pandas(result_df)
