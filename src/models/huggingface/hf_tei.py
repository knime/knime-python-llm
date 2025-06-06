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
from ..base import EmbeddingsPortObjectSpec, EmbeddingsPortObject
from .hf_base import hf_category, hf_icon
from .hf_hub import (
    hf_authentication_port_type,
    HFAuthenticationPortObject,
    HFAuthenticationPortObjectSpec,
)


# Other imports
from typing import Optional

hf_tei_category = knext.category(
    path=hf_category,
    level_id="tei",
    name="Text Embeddings Inference (TEI)",
    description="Contains nodes that connect to Hugging Face's text embeddings inference server.",
    icon=hf_icon,
)


class HFTEIEmbeddingsPortObjectSpec(EmbeddingsPortObjectSpec):
    def __init__(
        self,
        inference_server_url: str,
        batch_size: int,
        hf_hub_auth: Optional[HFAuthenticationPortObjectSpec],
    ) -> None:
        super().__init__()
        self._inference_server_url = inference_server_url
        self._batch_size = batch_size
        self._hf_hub_auth = hf_hub_auth

    @property
    def inference_server_url(self):
        return self._inference_server_url

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def hf_hub_auth(self) -> Optional[HFAuthenticationPortObjectSpec]:
        return self._hf_hub_auth

    def validate_context(self, ctx: knext.ConfigurationContext):
        if self.hf_hub_auth:
            self.hf_hub_auth.validate_context(ctx)

    def serialize(self) -> dict:
        return {
            "inference_server_url": self.inference_server_url,
            "batch_size": self.batch_size,
            "hf_hub_auth": self._hf_hub_auth.serialize() if self._hf_hub_auth else None,
        }

    @classmethod
    def deserialize(cls, data: dict):
        hub_auth_data = data.get("hf_hub_auth")
        if hub_auth_data:
            hub_auth = HFAuthenticationPortObjectSpec.deserialize(hub_auth_data)
        else:
            hub_auth = None
        return cls(data["inference_server_url"], data["batch_size"], hub_auth)


class HFTEIEmbeddingsPortObject(EmbeddingsPortObject):
    def __init__(self, spec: HFTEIEmbeddingsPortObjectSpec) -> None:
        super().__init__(spec)

    @property
    def spec(self) -> HFTEIEmbeddingsPortObjectSpec:
        return super().spec

    def create_model(self, ctx: knext.ExecutionContext):
        from ._hf_llm import HuggingFaceTEIEmbeddings

        hub_auth = self.spec.hf_hub_auth
        return HuggingFaceTEIEmbeddings(
            model=self.spec.inference_server_url,
            batch_size=self.spec.batch_size,
            hf_api_token=hub_auth.get_token(ctx) if hub_auth else None,
        )


huggingface_tei_embeddings_port_type = knext.port_type(
    "Hugging Face TEI Embedding Model",
    HFTEIEmbeddingsPortObject,
    HFTEIEmbeddingsPortObjectSpec,
)


@knext.node(
    "HF TEI Embedding Model Selector",
    knext.NodeType.SOURCE,
    hf_icon,
    hf_tei_category,
    keywords=[
        "GenAI",
        "Gen AI",
        "Generative AI",
        "HuggingFace",
        "Hugging Face",
        "Text Embeddings Inference",
        "RAG",
        "Retrieval Augmented Generation",
    ],
)
@knext.input_port(
    name="Hugging Face Hub Connection",
    description="An optional Hugging Face Hub connection that can be used to "
    "access protected Hugging Face inference endpoints.",
    port_type=hf_authentication_port_type,
    optional=True,
)
@knext.output_port(
    "Embedding Model",
    "Connection to an embedding model hosted on a Text Embeddings Inference server.",
    huggingface_tei_embeddings_port_type,
)
class HFTEIEmbeddingsConnector:
    """
    Select a dedicated Text Embeddings Inference Server.

    This node can connect to locally or remotely hosted TEI servers which includes
    [Text Embedding Inference Endpoints](https://huggingface.co/docs/inference-endpoints/) of
    popular embedding models that are deployed via Hugging Face Hub.

    Protected endpoints require a connection with a **HF Hub Authenticator** node in order to authenticate with Hugging Face Hub.

    The [Text Embeddings Inference Server](https://github.com/huggingface/text-embeddings-inference)
    is a toolkit for deploying and serving open source text embeddings and sequence classification models.

    For more details and information about integrating with the Hugging Face Embeddings Inference
    and setting up a server, refer to
    [Text Embeddings Inference GitHub](https://github.com/huggingface/text-embeddings-inference).

    **Note**: If you use the
    [Credentials Configuration node](https://hub.knime.com/knime/extensions/org.knime.features.js.quickforms/latest/org.knime.js.base.node.configuration.input.credentials.CredentialsDialogNodeFactory)
    and do not select the "Save password in configuration (weakly encrypted)" option for passing the API key via the **HF Hub Authenticator** node,
    the Credentials Configuration node will need to be reconfigured upon reopening the workflow, as the credentials flow variable
    was not saved and will therefore not be available to downstream nodes.
    """

    server_url = knext.StringParameter(
        "Text Embeddings Inference server URL",
        "The URL where the Text Embeddings Inference server is hosted e.g. `http://localhost:8080/`.",
    )

    batch_size = knext.IntParameter(
        "Batch size",
        "How many texts should be sent to the embeddings endpoint in one batch.",
        min_value=1,
        default_value=32,
    )

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        hf_hub_auth: Optional[HFAuthenticationPortObjectSpec],
    ) -> HFTEIEmbeddingsPortObjectSpec:
        if not self.server_url:
            raise knext.InvalidParametersError("Server URL missing")

        if hf_hub_auth:
            hf_hub_auth.validate_context(ctx)

        return self.create_spec(hf_hub_auth)

    def execute(
        self,
        ctx: knext.ExecutionContext,
        hf_hub_auth: Optional[HFAuthenticationPortObject],
    ) -> HFTEIEmbeddingsPortObject:
        hf_hub_auth = hf_hub_auth.spec if hf_hub_auth else None
        return HFTEIEmbeddingsPortObject(self.create_spec(hf_hub_auth))

    def create_spec(
        self, hf_hub_auth: Optional[HFAuthenticationPortObjectSpec]
    ) -> HFTEIEmbeddingsPortObjectSpec:
        return HFTEIEmbeddingsPortObjectSpec(
            self.server_url, self.batch_size, hf_hub_auth
        )
