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

from ..base import EmbeddingsPortObjectSpec, EmbeddingsPortObject
from ._base import (
    ibm_watsonx_icon,
    ibm_watsonx_category,
)
from ._auth import (
    IBMwatsonxAuthenticationPortObjectSpec,
    IBMwatsonxAuthenticationPortObject,
    ibm_watsonx_auth_port_type,
)
from ._util import (
    list_embedding_models,
    check_model,
)


class IBMwatsonxEmbeddingPortObjectSpec(EmbeddingsPortObjectSpec):
    def __init__(self, auth: IBMwatsonxAuthenticationPortObjectSpec, model_id: str):
        super().__init__()
        self._auth = auth
        self._model_id = model_id

    @property
    def auth(self) -> IBMwatsonxAuthenticationPortObjectSpec:
        return self._auth

    @property
    def model_id(self) -> str:
        return self._model_id

    def serialize(self):
        return {
            "auth": self.auth.serialize(),
            "model_id": self.model_id,
        }

    @classmethod
    def deserialize(cls, data: dict):
        auth = IBMwatsonxAuthenticationPortObjectSpec.deserialize(data["auth"])
        return cls(
            auth=auth,
            model_id=data["model_id"],
        )


class IBMwatsonxEmbeddingPortObject(EmbeddingsPortObject):
    @property
    def spec(self) -> IBMwatsonxEmbeddingPortObjectSpec:
        return super().spec

    def create_model(self, ctx):
        from ._embedding_model import WatsonxEmbeddings

        # Retrieve the name-id map for projects or spaces
        project_id, space_id = self.spec.auth.get_project_or_space_ids(ctx)

        model = WatsonxEmbeddings(
            apikey=self.spec.auth.get_api_key(ctx),
            url=self.spec.auth.base_url,
            model_id=self.spec.model_id,
            project_id=project_id,
            space_id=space_id,
        )
        return model


ibm_watsonx_embedding_port_type = knext.port_type(
    "IBM watsonx.ai Embedding Model",
    IBMwatsonxEmbeddingPortObject,
    IBMwatsonxEmbeddingPortObjectSpec,
)


@knext.node(
    name="IBM watsonx.ai Embedding Model Connector",
    category=ibm_watsonx_category,
    icon_path=ibm_watsonx_icon,
    keywords=["IBMwatsonx", "Embedding", "GenAI"],
    node_type=knext.NodeType.SOURCE,
)
@knext.input_port(
    "IBM watsonx.ai Authentication",
    "The authentication for the IBM watsonx.ai API.",
    ibm_watsonx_auth_port_type,
)
@knext.output_port(
    "IBM watsonx.ai Embedding Model",
    "Connection to an embedding model provided by IBM watsonx.ai.",
    ibm_watsonx_embedding_port_type,
)
class IBMwatsonxEmbeddingModelConnector:
    """Connects to an embedding model provided by IBM watsonx.ai.

    After successfully authenticating using the **IBM watsonx.ai Authenticator** node, you can select an embedding model. Refer to
    [IBM watsonx.ai documentation](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-models-embed.html?context=wx#embed) for more information on available embedding models. At the moment, only the embedding models from foundation models are supported.

    **Note**: If you want to use a space, make sure that the space has a valid runtime service instance. You can check this at
    [IBM watsonx.ai Studio](https://dataplatform.cloud.ibm.com/login) under Manage tab in your space.

    """

    model_id = knext.StringParameter(
        "Model",
        description="The model to use for the embedding generation.",
        default_value="",
        choices=list_embedding_models,
    )

    def configure(
        self, ctx, auth: IBMwatsonxAuthenticationPortObjectSpec
    ) -> IBMwatsonxEmbeddingPortObjectSpec:
        check_model(self.model_id)
        auth.validate_context(ctx)
        return self._create_spec(auth)

    def _create_spec(
        self, auth: IBMwatsonxAuthenticationPortObjectSpec
    ) -> IBMwatsonxEmbeddingPortObjectSpec:
        return IBMwatsonxEmbeddingPortObjectSpec(
            auth,
            self.model_id,
        )

    def execute(
        self, ctx, auth: IBMwatsonxAuthenticationPortObject
    ) -> IBMwatsonxEmbeddingPortObject:
        # Check if the model is still available
        if self.model_id not in auth.spec.get_embedding_model_list(ctx):
            raise knext.InvalidParametersError(
                f"The embedding model {self.model_id} is not available."
            )
        return IBMwatsonxEmbeddingPortObject(self._create_spec(auth.spec))
