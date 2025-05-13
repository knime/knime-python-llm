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


from typing import Any, List, Optional
from langchain_core.language_models import LLM
from langchain_core.embeddings import Embeddings
from pydantic import model_validator, BaseModel
from huggingface_hub import InferenceClient, AsyncInferenceClient
from .hf_base import raise_for
import util


class HFLLM(LLM):
    """Custom implementation backed by huggingface_hub.InferenceClient.
    We can't use the implementation of langchain_community because it always requires an api token (and is
    probably going to be deprecated soon) and we also can't use the langchain_huggingface implementation
    since it has torch as a required dependency."""

    model: str
    """Can be a repo id on hugging face hub or the url of a TGI server."""
    provider: str = None  # None for TGI
    hf_api_token: Optional[str] = None
    max_new_tokens: int = 512
    top_k: Optional[int] = None
    top_p: Optional[float] = 0.95
    typical_p: Optional[float] = 0.95
    temperature: Optional[float] = 0.8
    repetition_penalty: Optional[float] = None
    client: Any
    seed: Optional[int] = None

    def _llm_type(self):
        return "hfllm"

    @model_validator(mode="before")
    @classmethod
    def validate_values(cls, values: dict) -> dict:
        values["client"] = InferenceClient(
            model=values["model"],
            provider=values.get("provider"),
            timeout=120,
            token=values.get("hf_api_token"),
        )
        return values

    def _call(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager=None,
        **kwargs: Any,
    ) -> str:
        client: InferenceClient = self.client
        try:
            return client.text_generation(
                prompt,
                max_new_tokens=self.max_new_tokens,
                repetition_penalty=self.repetition_penalty,
                top_k=self.top_k,
                top_p=self.top_p,
                typical_p=self.typical_p,
                seed=self.seed,
            )
        except Exception as ex:
            raise_for(ex)


class HuggingFaceEmbeddings(BaseModel, Embeddings):
    """Custom implementation backed by huggingface_hub.InferenceClient to avoid the torch dependency of
    langchain_huggingface."""

    model: str
    """Can be a model or a repo id on hugging face hub or the url of a TEI server."""
    provider: str = None  # None for TEI
    hf_api_token: Optional[str] = None
    client: Any
    async_client: Any

    @model_validator(mode="before")
    @classmethod
    def validate_values(cls, values: dict) -> dict:
        values["client"] = InferenceClient(
            model=values["model"],
            provider=values.get("provider"),
            timeout=120,
            token=values.get("hf_api_token"),
        )
        values["async_client"] = AsyncInferenceClient(
            model=values["model"],
            provider=values.get("provider"),
            timeout=120,
            token=values.get("hf_api_token"),
        )
        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = [
            text.replace("\n", " ") for text in texts
        ]  # newlines can negatively affect performance according to langchain_huggingface
        try:
            responses = self.client.feature_extraction(text=texts)
        except Exception as ex:
            raise_for(ex)
        return responses.tolist()

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = [
            text.replace("\n", " ") for text in texts
        ]  # newlines can negatively affect performance according to langchain_huggingface
        try:
            responses = await self.async_client.feature_extraction(text=texts)
        except Exception as ex:
            raise_for(ex)
        return responses.tolist()

    def embed_query(self, text: str) -> list[float]:
        response = self.embed_documents([text])[0]
        return response

    async def aembed_query(self, text: str) -> list[float]:
        response = (await self.aembed_documents([text]))[0]
        return response


class HuggingFaceTEIEmbeddings(HuggingFaceEmbeddings):
    batch_size: int = 32

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Overrides HuggingFaceEmbeddings 'embed_documents' to allow batches
        return util.batched_apply(super().embed_documents, texts, self.batch_size)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        # Overrides HuggingFaceEmbeddings 'aembed_documents' to allow batches
        return util.abatched_apply(super().aembed_documents, texts, self.batch_size)
