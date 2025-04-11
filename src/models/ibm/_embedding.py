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
from typing import List


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


class WatsonxEmbeddingsWrapper:
    """
    Wrapper for WatsonxEmbeddings to handle exceptions that may occur during embedding generation.
    """

    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            return self.model.embed_documents(texts)
        except Exception:
            raise knext.InvalidParametersError(
                "Failed to embed texts. If you selected a space to run your model, "
                "make sure that the space has a valid runtime service instance. "
                "You can check this at IBM watsonx.ai Studio under Manage tab in your space."
            )


class IBMwatsonxEmbeddingPortObject(EmbeddingsPortObject):
    @property
    def spec(self) -> IBMwatsonxEmbeddingPortObjectSpec:
        return super().spec

    def create_model(self, ctx):
        from langchain_ibm import WatsonxEmbeddings

        # Retrieve the name-id map for projects or spaces
        project_id, space_id = self.spec.auth.get_project_or_space_ids(ctx)

        embedding_model = WatsonxEmbeddingsWrapper(
            model=WatsonxEmbeddings(
                apikey=self.spec.auth.get_api_key(ctx),
                url=self.spec.auth.base_url,
                model_id=self.spec.model_id,
                project_id=project_id,
                space_id=space_id,
            )
        )

        return embedding_model


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
