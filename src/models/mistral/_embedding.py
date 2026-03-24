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

from ._util import (
    MISTRAL_EMBEDDING_MODELS_FALLBACK,
    MISTRAL_EMBEDDING_DEFAULT,
    create_mistral_httpx_clients,
)
from ._base import mistral_icon, mistral_category
from ._auth import (
    MistralAuthenticationPortObjectSpec,
    MistralAuthenticationPortObject,
    mistral_auth_port_type,
)
from ..base import EmbeddingsPortObjectSpec, EmbeddingsPortObject


class MistralEmbeddingsPortObjectSpec(EmbeddingsPortObjectSpec):
    def __init__(
        self,
        auth: MistralAuthenticationPortObjectSpec,
        model: str,
    ) -> None:
        super().__init__()
        self._auth = auth
        self._model = model

    @property
    def auth(self) -> MistralAuthenticationPortObjectSpec:
        return self._auth

    @property
    def model(self) -> str:
        return self._model

    def validate_context(self, ctx):
        self._auth.validate_context(ctx)

    def serialize(self) -> dict:
        return {
            "auth": self._auth.serialize(),
            "model": self._model,
        }

    @classmethod
    def deserialize(cls, data: dict):
        auth = MistralAuthenticationPortObjectSpec.deserialize(data["auth"])
        return cls(auth=auth, model=data["model"])


class MistralEmbeddingsPortObject(EmbeddingsPortObject):
    def __init__(self, spec: MistralEmbeddingsPortObjectSpec):
        super().__init__(spec)

    @property
    def spec(self) -> MistralEmbeddingsPortObjectSpec:
        return super().spec

    def create_model(self, ctx: knext.ExecutionContext):
        from langchain_mistralai import MistralAIEmbeddings

        api_key = ctx.get_credentials(self.spec.auth.credentials).password
        client, async_client = create_mistral_httpx_clients(
            api_key, self.spec.auth.base_url
        )

        return MistralAIEmbeddings(
            api_key=api_key,
            endpoint=self.spec.auth.base_url,
            model=self.spec.model,
            client=client,
            async_client=async_client,
        )


mistral_embeddings_port_type = knext.port_type(
    "Mistral AI Embedding Model",
    MistralEmbeddingsPortObject,
    MistralEmbeddingsPortObjectSpec,
)


class MistralModelSelectionOptions(knext.EnumParameterOptions):
    DEFAULT_MODELS = (
        "Default models",
        "Select from a curated list of known Mistral AI embedding models.",
    )
    ALL_MODELS = (
        "All models",
        """Select from all models available for the provided API key.
        Note that this includes models that may not support embeddings,
        so please make sure to select a compatible model.""",
    )


def _list_all_models(ctx: knext.DialogCreationContext) -> list[str]:
    if (specs := ctx.get_input_specs()) and (auth_spec := specs[0]):
        return auth_spec.get_embedding_model_list(ctx)
    return []


@knext.node(
    name="Mistral AI Embedding Model Selector",
    node_type=knext.NodeType.SOURCE,
    icon_path=mistral_icon,
    category=mistral_category,
    keywords=[
        "Mistral",
        "GenAI",
        "Gen AI",
        "Generative AI",
        "Embeddings",
        "Embedding model",
        "RAG",
        "Retrieval Augmented Generation",
    ],
)
@knext.input_port(
    "Mistral AI Authentication",
    "The authentication for the Mistral AI API.",
    mistral_auth_port_type,
)
@knext.output_port(
    "Mistral AI Embedding Model",
    "The configured Mistral AI embedding model.",
    mistral_embeddings_port_type,
)
class MistralEmbeddingModelConnector:
    """Select a Mistral AI embedding model.

    This node establishes a connection with a Mistral AI embedding model. After successfully
    authenticating using the **Mistral AI Authenticator** node, you can select a model from
    the curated list of known embedding models.

    If Mistral AI releases a new embedding model that is not contained in the predefined list,
    you can also select from a list of all available Mistral AI models.
    """

    selection = knext.EnumParameter(
        "Model selection",
        "Whether to select from the curated list of known embedding models or from all models available for the provided API key.",
        MistralModelSelectionOptions.DEFAULT_MODELS.name,
        MistralModelSelectionOptions,
        style=knext.EnumParameter.Style.VALUE_SWITCH,
    )

    model = knext.StringParameter(
        "Model",
        "Select a Mistral AI embedding model.",
        default_value=MISTRAL_EMBEDDING_DEFAULT,
        choices=MISTRAL_EMBEDDING_MODELS_FALLBACK,
    ).rule(
        knext.OneOf(selection, [MistralModelSelectionOptions.DEFAULT_MODELS.name]),
        knext.Effect.SHOW,
    )

    specific_model = knext.StringParameter(
        "Specific model ID",
        """Select from all models available for the provided API key.
        The selected model must be compatible with the Mistral AI embeddings API.
        This configuration will **overwrite** the default model selection when set.""",
        default_value="",
        choices=_list_all_models,
    ).rule(
        knext.OneOf(selection, [MistralModelSelectionOptions.ALL_MODELS.name]),
        knext.Effect.SHOW,
    )

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        auth: MistralAuthenticationPortObjectSpec,
    ) -> MistralEmbeddingsPortObjectSpec:
        auth.validate_context(ctx)
        return self._create_spec(auth)

    def execute(
        self,
        ctx: knext.ExecutionContext,
        auth: MistralAuthenticationPortObject,
    ) -> MistralEmbeddingsPortObject:
        return MistralEmbeddingsPortObject(self._create_spec(auth.spec))

    def _create_spec(
        self, auth: MistralAuthenticationPortObjectSpec
    ) -> MistralEmbeddingsPortObjectSpec:
        return MistralEmbeddingsPortObjectSpec(
            auth=auth,
            model=self._get_active_model(),
        )

    def _get_active_model(self) -> str:
        if self.selection == MistralModelSelectionOptions.ALL_MODELS.name:
            if not self.specific_model:
                raise knext.InvalidParametersError(
                    "No model selected. Please select a model from the list."
                )
            return self.specific_model
        if not self.model:
            raise knext.InvalidParametersError(
                "No model selected. Please select a model from the list."
            )
        return self.model
