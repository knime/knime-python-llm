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


from typing import List, Optional, Any, Sequence
from langchain_core.language_models import BaseLLM, BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    AIMessage,
)
from langchain_core.callbacks.manager import (
    CallbackManagerForLLMRun,
    AsyncCallbackManagerForLLMRun,
)
from langchain_core.outputs import ChatGeneration, ChatResult


class LLMChatModelAdapter(BaseChatModel):
    """
    This class adapts LLMs as chat models, allowing LLMs that have been fine-tuned for
    chat applications to be used as chat models with the LLM Prompter (Conversation).
    """

    llm: BaseLLM
    system_prompt_template: str
    prompt_template: str

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prediction = self._predict_messages(
            messages=messages,
            system_prompt_template=self.system_prompt_template,
            prompt_template=self.prompt_template,
            stop=stop,
            **kwargs,
        )
        return ChatResult(generations=[ChatGeneration(message=prediction)])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prediction = await self._apredict_messages(
            messages=messages,
            system_prompt_template=self.system_prompt_template,
            prompt_template=self.prompt_template,
            stop=stop,
            **kwargs,
        )
        return ChatResult(generations=[ChatGeneration(message=prediction)])

    def _apply_prompt_templates(
        self,
        messages: Sequence[BaseMessage],
        system_prompt_template: Optional[str] = None,
        prompt_template: Optional[str] = None,
    ) -> str:
        string_messages = []

        message_templates = {
            HumanMessage: prompt_template,
            AIMessage: "%1",
            SystemMessage: system_prompt_template,
        }

        for m in messages:
            if type(m) not in message_templates:
                raise ValueError(f"Got unsupported message type: {m}")

            template = message_templates[type(m)]

            # if the template doesn't include the predefined pattern "%1",
            # the template input will be ignored and only the entered message will be passed
            if "%1" in template:
                message = template.replace(
                    "%1",
                    (
                        m.content
                        if isinstance(m, (HumanMessage, SystemMessage))
                        else m.content
                    ),
                )
            else:
                message = m.content

            string_messages.append(message)

        return "\n".join(string_messages)

    def _predict_messages(
        self,
        messages: List[BaseMessage],
        stop: Optional[Sequence[str]] = None,
        system_prompt_template: Optional[str] = None,
        prompt_template: Optional[str] = None,
        **kwargs: Any,
    ) -> BaseMessage:
        text = self._apply_prompt_templates(
            messages=messages,
            system_prompt_template=system_prompt_template,
            prompt_template=prompt_template,
        )
        if stop is None:
            _stop = None
        else:
            _stop = list(stop)
        content = self.llm(text, stop=_stop, **kwargs)
        return AIMessage(content=content)

    async def _apredict_messages(
        self,
        messages: List[BaseMessage],
        stop: Optional[Sequence[str]] = None,
        system_prompt_template: Optional[str] = None,
        prompt_template: Optional[str] = None,
        **kwargs: Any,
    ) -> BaseMessage:
        text = self._apply_prompt_templates(
            messages=messages,
            system_prompt_template=system_prompt_template,
            prompt_template=prompt_template,
        )
        if stop is None:
            _stop = None
        else:
            _stop = list(stop)
        content = await self.llm._call_async(text, stop=_stop, **kwargs)
        return AIMessage(content=content)

    def _llm_type(self) -> str:
        """Return type of llm."""
        return "LLMChatModelAdapter"
