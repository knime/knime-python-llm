import knime.extension as knext
from models.base import (
    EmbeddingsPortObjectSpec,
    EmbeddingsPortObject,
    embeddings_port_type,
)
from .base import (
    VectorStorePortObjectSpec,
    VectorStorePortObject,
    vectorstore_port_type,
)

from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader

vectorstore_category = ""
chroma_icon = "./icons/chroma.png"


@knext.node(
    "Vectorstore Loader",
    knext.NodeType.SOURCE,
    chroma_icon,
    category=vectorstore_category,
)
@knext.input_port(
    "Embeddings", "The embeddings to use in the vectorstore.", embeddings_port_type
)
@knext.output_port("Vectorstore", "The loaded vectorstore.", vectorstore_port_type)
class VectorStoreLoader:
    def configure(
        self, ctx: knext.ConfigurationContext, embeddings: EmbeddingsPortObjectSpec
    ) -> VectorStorePortObjectSpec:
        return VectorStorePortObjectSpec()

    def execute(
        self, ctx: knext.ExecutionContext, embeddings: EmbeddingsPortObject
    ) -> VectorStorePortObject:
        return VectorStorePortObject(VectorStorePortObjectSpec)
