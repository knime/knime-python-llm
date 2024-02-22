# KNIME / own imports
import knime.extension as knext
from ..base import EmbeddingsPortObjectSpec, EmbeddingsPortObject
from .hf_base import hf_category, hf_icon


# Langchain imports
from langchain_community.embeddings import HuggingFaceHubEmbeddings

hf_tei_category = knext.category(
    path=hf_category,
    level_id="tei",
    name="Text Embeddings Inference (TEI)",
    description="Contains nodes that connect to Hugging Face's text embeddings inference server.",
    icon=hf_icon,
)


class HFTEIEmbeddingsPortObjectSpec(EmbeddingsPortObjectSpec):
    def __init__(self, inference_server_url: str) -> None:
        super().__init__()
        self._inference_server_url = inference_server_url

    @property
    def inference_server_url(self):
        return self._inference_server_url

    def serialize(self) -> dict:
        return {"inference_server_url": self.inference_server_url}

    @classmethod
    def deserialize(cls, data: dict):
        return cls(data["inference_server_url"])


class HFTEIEmbeddingsPortObject(EmbeddingsPortObject):
    def __init__(self, spec: HFTEIEmbeddingsPortObjectSpec) -> None:
        super().__init__(spec)

    @property
    def spec(self) -> HFTEIEmbeddingsPortObjectSpec:
        return super().spec

    def create_model(self, ctx) -> HuggingFaceHubEmbeddings:
        return HuggingFaceHubEmbeddings(model=self.spec.inference_server_url)


huggingface_tei_embeddings_port_type = knext.port_type(
    "Hugging Face TEI Embeddings Model",
    HFTEIEmbeddingsPortObject,
    HFTEIEmbeddingsPortObjectSpec,
)


@knext.node(
    "HF TEI Embeddings Connector",
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
@knext.output_port(
    "Embeddings Model",
    "Connection to an embeddings model hosted on a Text Embeddings Inference server.",
    huggingface_tei_embeddings_port_type,
)
class HFTEIEmbeddingsConnector:
    """
    Connects to a dedicated Text Embeddings Inference Server.

    The [Text Embeddings Inference Server](https://github.com/huggingface/text-embeddings-inference)
    is a toolkit for deploying and serving open source text embeddings and sequence classification models.

    This node can connect to locally or remotely hosted TEI servers which includes public
    [Inference Endpoints](https://huggingface.co/docs/inference-endpoints/) of
    popular embeddings models that are deployed via Hugging Face Hub.

    For more details and information about integrating with the Hugging Face Embeddings Inference
    and setting up a server, refer to
    [Text Embeddings Inference GitHub](https://github.com/huggingface/text-embeddings-inference).
    """

    server_url = knext.StringParameter(
        "Text Embeddings Inference Server URL",
        "The URL where the Text Embeddings Inference server is hosted e.g. `http://localhost:8010/`.",
    )

    def configure(
        self, ctx: knext.ConfigurationContext
    ) -> HFTEIEmbeddingsPortObjectSpec:
        if not self.server_url:
            raise knext.InvalidParametersError("Server URL missing")

        return self.create_spec()

    def execute(self, ctx: knext.ExecutionContext) -> HFTEIEmbeddingsPortObject:
        return HFTEIEmbeddingsPortObject(self.create_spec())

    def create_spec(self) -> HFTEIEmbeddingsPortObjectSpec:
        return HFTEIEmbeddingsPortObjectSpec(self.server_url)
