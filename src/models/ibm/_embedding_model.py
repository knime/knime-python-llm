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

"""
Custom LangChain Embeddings implementation for IBM watsonx.ai.

This module provides a LangChain-compatible embedding model that uses direct REST API
calls to watsonx.ai, replacing the langchain-ibm package dependency.
"""

import knime.extension as knext
from typing import List, Optional

from langchain_core.embeddings import Embeddings

from ._client import WatsonxClient


class WatsonxEmbeddings(Embeddings):
    """
    LangChain Embeddings implementation for IBM watsonx.ai.

    This implementation uses direct REST API calls instead of the langchain-ibm
    package.
    """

    def __init__(
        self,
        model_id: str,
        apikey: str,
        url: str,
        project_id: Optional[str] = None,
        space_id: Optional[str] = None,
    ):
        """
        Initialize the watsonx.ai embedding model.

        Args:
            model_id: The watsonx.ai embedding model ID to use
            apikey: IBM Cloud API key for authentication
            url: Base URL for the watsonx.ai API
            project_id: Project ID to use for the API calls
            space_id: Space ID to use for the API calls
        """
        self._model_id = model_id
        self._apikey = apikey
        self._url = url
        self._project_id = project_id
        self._space_id = space_id
        self._client: Optional[WatsonxClient] = None

    def _get_client(self) -> WatsonxClient:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = WatsonxClient(
                api_key=self._apikey,
                base_url=self._url,
            )
        return self._client

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        try:
            client = self._get_client()
            return client.embed_documents(
                model_id=self._model_id,
                texts=texts,
                project_id=self._project_id,
                space_id=self._space_id,
            )
        except Exception as e:
            raise knext.InvalidParametersError(
                "Failed to embed texts. If you selected a space to run your model, "
                "make sure that the space has a valid runtime service instance. "
                "You can check this at IBM watsonx.ai Studio under Manage tab in your space."
            ) from e

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.

        Args:
            text: The text to embed

        Returns:
            Embedding vector
        """
        try:
            client = self._get_client()
            return client.embed_query(
                model_id=self._model_id,
                text=text,
                project_id=self._project_id,
                space_id=self._space_id,
            )
        except Exception as e:
            raise knext.InvalidParametersError(
                "Failed to embed query. If you selected a space to run your model, "
                "make sure that the space has a valid runtime service instance. "
                "You can check this at IBM watsonx.ai Studio under Manage tab in your space."
            ) from e
