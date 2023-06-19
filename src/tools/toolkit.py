import knime.extension as knext
import pandas as pd

from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.document_loaders import YoutubeLoader

from models.base import (
    EmbeddingsPortObjectSpec,
    EmbeddingsPortObject,
    embeddings_port_type,
)

from .base import (
    VectorStorePortObject,
    VectorStorePortObjectSpec,
    VectorStorePortObjectSpecContent,
    VectorStorePortObjectContent,
    vector_store_port_type,
)
import logging

LOGGER = logging.getLogger(__name__)


@knext.node(
    "Vectorstore Combiner",
    knext.NodeType.PREDICTOR,
    langchain_icon,
    category=agent_category,
)
@knext.input_port(
    "Vector Store",
    "The vector store to get context for the agent.",
    vector_store_port_type,
)
@knext.input_port(
    "Vector Store",
    "The vector store to get context for the agent.",
    vector_store_port_type,
)
@knext.output_port("Vectorstore", "The loaded vectorstore.", vector_store_port_type)
class VectorstoreCombiner:
    def configure(
        self,
        ctx: knext.ConfigurationContext,
        vectorstore_spec_first: VectorStorePortObjectSpec,
        vectorstore_spec_second: VectorStorePortObjectSpec,
    ) -> VectorStorePortObjectSpec:
        return VectorStorePortObjectSpec(self.create_spec_content(None))

    def execute(
        self,
        ctx: knext.ExecutionContext,
        vectorstore_first: VectorStorePortObject,
        vectorstore_second: VectorStorePortObject,
    ):
        LOGGER.info("in there")

        persist_directories = [
            vectorstore_first.spec._persist_directory,
            vectorstore_second.spec._persist_directory,
        ]

        LOGGER.info(vars(vectorstore_first._content._embeddings_model))

        return VectorStorePortObject(
            spec=VectorStorePortObjectSpec(
                self.create_spec_content(persist_directories)
            ),
            content=FAISSVectorstorePortObjectContent(
                self.create_spec_content(persist_directories),
                vectorstore_first._content._embeddings_model,
            ),
        )

    def create_spec_content(self, persist_directories):
        return FAISSVectorstorePortObjectSpecContent(persist_directories)
