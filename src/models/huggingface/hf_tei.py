# KNIME / own imports
import knime.extension as knext
from ..base import EmbeddingsPortObjectSpec, EmbeddingsPortObject
from .hf_base import hf_category, hf_icon

# Langchain imports
from langchain_community.embeddings import HuggingFaceHubEmbeddings

# Other imports
import json
from typing import List

hf_tei_category = knext.category(
    path=hf_category,
    level_id="tei",
    name="Text Embeddings Inference (TEI)",
    description="Contains nodes that connect to Hugging Face's text embeddings inference server.",
    icon=hf_icon,
)


class _HuggingFaceEmbeddings(HuggingFaceHubEmbeddings):
    batch_size: int = 32

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Overrides HuggingFaceHubEmbeddings 'embed_documents' method for batch_size

        embeddings_vectors: List[List[float]] = []

        # replace newlines, which can negatively affect performance.
        texts = [text.replace("\n", " ") for text in texts]
        n_texts = len(texts)

        _model_kwargs = self.model_kwargs or {}

        for index in range(0, n_texts, self.batch_size):
            batch_end = index + self.batch_size

            text_batch = texts[index:batch_end]

            responses = self.client.post(
                json={"inputs": text_batch, "parameters": _model_kwargs}, task=self.task
            )

            vectors = json.loads(responses.decode())

            embeddings_vectors.extend(vectors)

        return embeddings_vectors

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        # Overrides HuggingFaceHubEmbeddings 'aembed_documents' method for batch_size

        embeddings_vectors: List[List[float]] = []

        # replace newlines, which can negatively affect performance.
        texts = [text.replace("\n", " ") for text in texts]
        n_texts = len(texts)

        _model_kwargs = self.model_kwargs or {}

        for index in range(0, n_texts, self.batch_size):
            batch_end = index + self.batch_size

            text_batch = texts[index:batch_end]

            responses = await self.client.post(
                json={"inputs": text_batch, "parameters": _model_kwargs}, task=self.task
            )

            embeddings_vectors.extend(json.loads(responses.decode()))

        return embeddings_vectors


class HFTEIEmbeddingsPortObjectSpec(EmbeddingsPortObjectSpec):
    def __init__(self, inference_server_url: str, batch_size: int) -> None:
        super().__init__()
        self._inference_server_url = inference_server_url
        self._batch_size = batch_size

    @property
    def inference_server_url(self):
        return self._inference_server_url

    @property
    def batch_size(self):
        return self._batch_size

    def serialize(self) -> dict:
        return {
            "inference_server_url": self.inference_server_url,
            "batch_size": self.batch_size,
        }

    @classmethod
    def deserialize(cls, data: dict):
        return cls(data["inference_server_url"], data["batch_size"])


class HFTEIEmbeddingsPortObject(EmbeddingsPortObject):
    def __init__(self, spec: HFTEIEmbeddingsPortObjectSpec) -> None:
        super().__init__(spec)

    @property
    def spec(self) -> HFTEIEmbeddingsPortObjectSpec:
        return super().spec

    def create_model(self, ctx) -> HuggingFaceHubEmbeddings:
        return _HuggingFaceEmbeddings(
            model=self.spec.inference_server_url, batch_size=self.spec.batch_size
        )


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
        "The URL where the Text Embeddings Inference server is hosted e.g. `http://localhost:8080/`.",
    )

    batch_size = knext.IntParameter(
        "Batch size",
        "How many texts will be send to the embeddings endpoint in one batch.",
        min_value=1,
        default_value=32,
        since_version="5.3.0",
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
        batch_size = getattr(self, "batch_size", 32)

        return HFTEIEmbeddingsPortObjectSpec(self.server_url, batch_size)
