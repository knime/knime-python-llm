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
import knime.api.schema as ks
from knime.extension import ConfigurationContext, ExecutionContext
from ._base import (
    hub_connector_icon,
    knime_category,
    create_authorization_headers,
    extract_api_base,
    create_model_choice_provider,
    list_models,
    validate_auth_spec,
)


from ..base import (
    EmbeddingsPortObjectSpec,
    EmbeddingsPortObject,
)


class KnimeHubEmbeddingsPortObjectSpec(EmbeddingsPortObjectSpec):
    def __init__(
        self,
        auth_spec: ks.HubAuthenticationPortObjectSpec,
        model_name: str,
    ) -> None:
        super().__init__()
        self._auth_spec = auth_spec
        self._model_name = model_name

    @property
    def auth_spec(self) -> ks.HubAuthenticationPortObjectSpec:
        return self._auth_spec

    @property
    def model_name(self) -> str:
        return self._model_name

    def validate_context(self, ctx: ConfigurationContext):
        validate_auth_spec(self.auth_spec)

    def serialize(self) -> dict:
        return {
            "auth": self.auth_spec.serialize(),
            "model_name": self.model_name,
        }

    @classmethod
    def deserialize(cls, data: dict):
        return cls(
            ks.HubAuthenticationPortObjectSpec.deserialize(data["auth"]),
            data["model_name"],
        )


class KnimeHubEmbeddingsPortObject(EmbeddingsPortObject):
    @property
    def spec(self) -> KnimeHubEmbeddingsPortObjectSpec:
        return super().spec

    def create_model(self, ctx: ExecutionContext):
        from ._embeddings_model import OpenAIEmbeddings

        auth_spec = self.spec.auth_spec
        return OpenAIEmbeddings(
            model=self.spec.model_name,
            base_url=extract_api_base(auth_spec),
            api_key="placeholder",
            extra_headers=create_authorization_headers(auth_spec),
        )


knime_embeddings_port_type = knext.port_type(
    "KNIME Hub Embeddings",
    KnimeHubEmbeddingsPortObject,
    KnimeHubEmbeddingsPortObjectSpec,
)


@knext.node(
    name="KNIME Hub Embedding Model Selector",
    node_type=knext.NodeType.SOURCE,
    icon_path=hub_connector_icon,
    category=knime_category,
    keywords=[
        "GenAI",
        "Gen AI",
        "Generative AI",
        "GenAI Gateway",
        "RAG",
        "Retrieval Augmented Generation",
    ],
)
@knext.input_port(
    name="KNIME Hub Credential",
    description="Credential for a KNIME Hub.",
    port_type=knext.PortType.HUB_AUTHENTICATION,
)
@knext.output_port(
    name="KNIME Hub Embeddings",
    description="An embedding model that connects to a KNIME Hub to embed documents.",
    port_type=knime_embeddings_port_type,
)
class KnimeHubEmbeddingsConnector:
    """
    Select an embedding model configured in the GenAI Gateway of the connected KNIME Hub.

    Connects to an embedding model configured in the GenAI Gateway of the connected KNIME Hub using the authentication
    provided via the input port.

    Use this node to generate embeddings, which are dense vector representations of text input data.
    Embeddings are useful for tasks like similarity search, e.g. in a retrieval augmented generation (RAG) system but
    can also be used for clustering, classification and other machine learning applications.
    """

    model_name = knext.StringParameter(
        "Model",
        "Select the model to use.",
        choices=create_model_choice_provider("embedding"),
    )

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        authentication: ks.HubAuthenticationPortObjectSpec,
    ) -> KnimeHubEmbeddingsPortObjectSpec:
        # raises exception if the hub authenticator has not been executed
        validate_auth_spec(authentication)
        if self.model_name == "":
            raise knext.InvalidParametersError("No embedding model selected.")
        return self._create_spec(authentication)

    def execute(
        self, ctx: knext.ExecutionContext, authentication: knext.PortObject
    ) -> KnimeHubEmbeddingsPortObject:
        available_models = list_models(authentication.spec, "embedding")
        if self.model_name not in available_models:
            raise knext.InvalidParametersError(
                f"The selected model {self.model_name} is not served by the connected Hub."
            )
        return KnimeHubEmbeddingsPortObject(self._create_spec(authentication.spec))

    def _create_spec(
        self, authentication: ks.HubAuthenticationPortObjectSpec
    ) -> KnimeHubEmbeddingsPortObjectSpec:
        return KnimeHubEmbeddingsPortObjectSpec(
            authentication,
            self.model_name,
        )
