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


# KNIME / own imports
import knime.extension as knext
from knime.extension.nodes import FilestorePortObject
from ..base import (
    EmbeddingsPortObjectSpec,
    EmbeddingsPortObject,
)

from ._utils import gpt4all_icon, gpt4all_category

from typing import Optional
import shutil
import os

_embeddings4all_model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"


def _create_embedding_model(
    model_name: str, model_path: str, n_threads: int, allow_download: bool
):
    from langchain_community.embeddings import GPT4AllEmbeddings

    return GPT4AllEmbeddings(
        model_name=model_name,
        n_threads=n_threads,
        gpt4all_kwargs={"allow_download": allow_download, "model_path": model_path},
    )


class Embeddings4AllPortObjectSpec(EmbeddingsPortObjectSpec):
    """The Embeddings4All port object spec."""

    def __init__(self, num_threads: int = 0) -> None:
        super().__init__()
        self._num_threads = num_threads

    @property
    def num_threads(self) -> int:
        return self._num_threads

    def serialize(self) -> dict:
        return {
            "num_threads": self._num_threads,
        }

    @classmethod
    def deserialize(cls, data: dict):
        return cls(data["num_threads"])


class Embeddings4AllPortObject(EmbeddingsPortObject, FilestorePortObject):
    """
    The Embeddings4All port object.

    The port object copies the Embeddings4All model into a filestore in order
    to make workflows containing such models portable.
    """

    def __init__(
        self,
        spec: EmbeddingsPortObjectSpec,
        model_name: str = _embeddings4all_model_name,
        model_path: Optional[str] = None,
    ) -> None:
        super().__init__(spec)
        self._model_name = model_name
        self._model_path = model_path

    @property
    def spec(self) -> Embeddings4AllPortObjectSpec:
        return super().spec

    def create_model(self, ctx):
        try:
            return _create_embedding_model(
                model_name=self._model_name,
                n_threads=self.spec.num_threads,
                model_path=self._model_path,
                allow_download=False,
            )
        except Exception as e:
            unsupported_model_exception = (
                "Unable to instantiate model: Unsupported model architecture: bert"
            )
            if str(e) == unsupported_model_exception:
                raise knext.InvalidParametersError(
                    "The current embedding model is incompatible. "
                    "Please run the GPT4All Embedding Model Selector again to download the latest model, "
                    "or update it manually to a newer version. "
                    "For additional details on available models, please refer to: "
                    "https://raw.githubusercontent.com/nomic-ai/gpt4all/main/gpt4all-chat/metadata/models3.json"
                )
            raise ValueError(f"The model at path {self.model_path} is not valid.")

    def write_to(self, file_path: str) -> None:
        from requests.exceptions import (
            ConnectionError,
        )  # The ConnectionError inherits from IOError, so we need the import

        os.makedirs(file_path)
        if self._model_path:
            # should be verified in the connector
            shutil.copy(
                os.path.join(self._model_path, self._model_name),
                os.path.join(file_path, self._model_name),
            )
        else:
            try:
                _create_embedding_model(
                    model_name=_embeddings4all_model_name,
                    model_path=file_path,
                    n_threads=1,
                    allow_download=True,
                )
            except ConnectionError:
                raise knext.InvalidParametersError(
                    "Connection error. Please ensure that your internet connection is enabled to download the model."
                )

    @classmethod
    def read_from(
        cls, spec: Embeddings4AllPortObjectSpec, file_path: str
    ) -> "Embeddings4AllPortObject":
        model_name = os.listdir(file_path)[0]
        return cls(spec, model_name, file_path)


embeddings4all_port_type = knext.port_type(
    "GPT4All Embeddings",
    Embeddings4AllPortObject,
    Embeddings4AllPortObjectSpec,
    id="org.knime.python.llm.models.gpt4all.Embeddings4AllPortObject",
)


class ModelRetrievalOptions(knext.EnumParameterOptions):
    DOWNLOAD = (
        "Download",
        "Downloads the model from GPT4All during execution. Requires an internet connection.",
    )
    READ = ("Read", "Reads the model from the local file system.")


@knext.node(
    "GPT4All Embedding Model Selector",
    knext.NodeType.SOURCE,
    gpt4all_icon,
    gpt4all_category,
    keywords=[
        "GenAI",
        "Gen AI",
        "Generative AI",
        "Local RAG",
        "Local Retrieval Augmented Generation",
    ],
)
@knext.output_port(
    "GPT4All Embedding model",
    "A GPT4All Embedding model that calculates embeddings on the local machine.",
    embeddings4all_port_type,
)
class Embeddings4AllConnector:
    """
    Select an embedding model that runs on the local machine.

    This node connects to an embedding model that runs on the local machine via GPT4All.

    The default model was trained on sentences and short paragraphs of English text.

    **Note**: Unlike the other GPT4All nodes, this node can be used on the KNIME Hub.
    """

    model_retrieval = knext.EnumParameter(
        "Model retrieval",
        "Defines how the model is retrieved during execution.",
        ModelRetrievalOptions.DOWNLOAD.name,
        ModelRetrievalOptions,
        style=knext.EnumParameter.Style.VALUE_SWITCH,
    )

    model_path = knext.LocalPathParameter(
        "Path to model", "The local file system path to the model."
    ).rule(
        knext.OneOf(model_retrieval, [ModelRetrievalOptions.READ.name]),
        knext.Effect.SHOW,
    )

    num_threads = knext.IntParameter(
        "Number of threads",
        """The number of threads the model uses. 
        More threads may reduce the runtime of queries to the model.

        If set to 0, the number of threads is determined automatically.""",
        0,
        min_value=0,
        is_advanced=True,
    )

    def configure(self, ctx) -> Embeddings4AllPortObjectSpec:
        return self._create_spec()

    def _create_spec(self) -> Embeddings4AllPortObjectSpec:
        n_threads = None if self.num_threads == 0 else self.num_threads
        return Embeddings4AllPortObjectSpec(n_threads)

    def execute(self, ctx) -> Embeddings4AllPortObject:
        if self.model_retrieval == ModelRetrievalOptions.DOWNLOAD.name:
            model_path = None
            model_name = _embeddings4all_model_name
        else:
            if not os.path.exists(self.model_path):
                raise ValueError(
                    f"The provided model path {self.model_path} does not exist."
                )
            model_path, model_name = os.path.split(self.model_path)
            try:
                _create_embedding_model(
                    model_name=model_name,
                    model_path=model_path,
                    n_threads=self.num_threads,
                    allow_download=False,
                )
            except Exception as e:
                unsupported_model_exception = (
                    "Unable to instantiate model: Unsupported model architecture: bert"
                )
                if str(e) == unsupported_model_exception:
                    raise knext.InvalidParametersError(
                        "The current embedding model is incompatible. "
                        "Please run the GPT4All Embedding Model Selector again to download the latest model, "
                        "or update it manually to a newer version. "
                        "For additional details on available models, please refer to: "
                        "https://raw.githubusercontent.com/nomic-ai/gpt4all/main/gpt4all-chat/metadata/models3.json"
                    )
                raise ValueError(f"The model at path {self.model_path} is not valid.")

        return Embeddings4AllPortObject(
            self._create_spec(),
            model_name=model_name,
            model_path=model_path,
        )
