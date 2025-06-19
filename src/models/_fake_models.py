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


from langchain_core.language_models import LLM, SimpleChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.messages import BaseMessage
from typing import Any, List, Optional, Mapping, Dict
from pydantic import BaseModel
import knime.extension as knext
from util import MissingValueHandlingOptions


def generate_response(
    response_dict: dict[str, str],
    default_response: str,
    prompt: str,
    missing_value_strategy: str,
    node: str,
):
    # LLM Prompter auto-injects this prefix during processing, causing a mismatch with the stored prompts
    if prompt.lower().startswith("human: "):
        prompt = prompt[len("human: ") :]

    response = response_dict.get(prompt)

    if not response:
        if missing_value_strategy == MissingValueHandlingOptions.Fail.name:
            raise knext.InvalidParametersError(
                f"""Could not find matching response for prompt: '{prompt}'. Please ensure that the prompt 
                exactly matches one specified in the prompt column of the {node} upstream."""
            )
        else:
            return default_response

    return response


class TestDictLLM(LLM):
    """Self implemented Test LLM wrapper for testing purposes."""

    response_dict: dict[str, str]
    default_response: str
    missing_value_strategy: str

    @property
    def _llm_type(self) -> str:
        return "test-dict"

    def _call(
        self,
        prompt: str,
        stop: List[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        return generate_response(
            self.response_dict,
            self.default_response,
            prompt,
            self.missing_value_strategy,
            "Test LLM Connector",
        )

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Return next response"""
        return generate_response(
            self.response_dict,
            self.default_response,
            prompt,
            self.missing_value_strategy,
            "Test LLM Connector",
        )

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {}


class TestChatModel(SimpleChatModel):
    """Test ChatModel for testing purposes."""

    response_dict: Dict[str, str]
    default_response: str
    missing_value_strategy: str

    @property
    def _llm_type(self) -> str:
        return "test-chat-model"

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        prompt = messages[len(messages) - 1].content
        return generate_response(
            self.response_dict,
            self.default_response,
            prompt,
            self.missing_value_strategy,
            "Test Chat Model Selector",
        )

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {}


class TestEmbeddings(Embeddings, BaseModel):
    embeddings_dict: dict[str, list[float]]
    fail_on_mismatch: bool

    def embed_documents(self, documents: List[str]) -> List[float]:
        return [self.embed_query(document) for document in documents]

    def embed_query(self, text: str) -> List[float]:
        try:
            return self.embeddings_dict[text]
        except KeyError:
            if self.fail_on_mismatch:
                raise knext.InvalidParametersError(
                    f"""Could not find document '{text}' in the Test Embedding Model. Please ensure that 
                    the query exactly matches one of the embedded documents."""
                )
            else:
                embeddings_dimension = len(next(iter(self.embeddings_dict.values())))
                zero_vector = [0.0 for _ in range(embeddings_dimension)]

                return zero_vector
