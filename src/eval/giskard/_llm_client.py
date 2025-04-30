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


import knime.extension as knext
from models.base import LLMPortObject, OutputFormatOptions

from typing import Sequence, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, ChatMessage, SystemMessage
from giskard.llm.client.base import LLMClient


class KnimeLLMClient(LLMClient):
    def __init__(self, port_object: LLMPortObject, ctx: knext.ExecutionContext):
        self._model = port_object
        self._ctx = ctx

        # Initialize attributes if they exist in the model spec
        attributes = ["_temperature", "_max_tokens", "_caller_id", "_seed", "_format"]
        for attr in attributes:
            if hasattr(self._model.spec, attr):
                setattr(self, attr, getattr(self._model.spec, attr))

    def complete(
        self,
        messages: Sequence[ChatMessage],
        temperature: float = 1,
        max_tokens: Optional[int] = None,
        caller_id: Optional[str] = None,
        seed: Optional[int] = None,
        format=None,
    ) -> ChatMessage:
        """Prompts the model to generate domain-specific probes. Uses the parameters of the model instead of
        the function parameters."""

        # Update model spec attributes if provided
        attributes = {
            "_temperature": temperature,
            "_max_tokens": (
                max_tokens
                if max_tokens is not None
                else getattr(self, "_max_tokens", None)
            ),
            "_caller_id": (
                caller_id
                if caller_id is not None
                else getattr(self, "_caller_id", None)
            ),
            "_seed": seed if seed is not None else getattr(self, "_seed", None),
            "_format": format if format is not None else getattr(self, "_format", None),
        }

        for attr, value in attributes.items():
            if hasattr(self._model.spec, attr):
                setattr(self._model.spec, attr, value)

        # Create model
        model = self._model.create_model(
            self._ctx, output_format=OutputFormatOptions.Text
        )

        converted_messages = self._convert_messages(messages)
        answer = model.invoke(converted_messages)
        if isinstance(model, BaseChatModel):
            answer = answer.content
        return ChatMessage(role="assistant", content=answer)

    _role_to_message_type = {
        "ai": AIMessage,
        "assistant": AIMessage,
        "user": HumanMessage,
        "human": HumanMessage,
        "system": SystemMessage,
    }

    def _create_message(self, role: str, content: str):
        if not role:
            raise RuntimeError("Giskard did not specify a message role.")
        message_type = self._role_to_message_type.get(role.lower(), None)
        if message_type:
            return message_type(content=content)
        else:
            # fallback
            return ChatMessage(content=content, role=role)

    def _convert_messages(self, messages: Sequence[ChatMessage]):
        return [self._create_message(msg.role, msg.content) for msg in messages]
