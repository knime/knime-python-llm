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
from typing import Any, Optional

from models.base import (
    EmbeddingsPortObjectSpec,
    EmbeddingsPortObject,
    embeddings_model_port_type,
)

from .base import (
    BaseVectorStoreCreator,
    FilestoreVectorstorePortObjectSpec,
    FilestoreVectorstorePortObject,
    store_category,
)

from numpy import ndarray

faiss_icon = "icons/embeddings/store.png"
faiss_category = knext.category(
    path=store_category,
    level_id="faiss",
    name="FAISS",
    description="Contains nodes for working with FAISS vector stores.",
    icon=faiss_icon,
)


class FAISSVectorstorePortObjectSpec(FilestoreVectorstorePortObjectSpec):
    # placeholder to enable use to add FAISS specific stuff later on
    pass


class FAISSVectorstorePortObject(FilestoreVectorstorePortObject):
    def __init__(
        self,
        spec: FAISSVectorstorePortObjectSpec,
        embeddings_port_object: EmbeddingsPortObject,
        folder_path: Optional[str] = None,
        vectorstore: Optional[Any] = None,
    ) -> None:
        super().__init__(
            spec, embeddings_port_object, folder_path, vectorstore=vectorstore
        )

    def load_vectorstore(self, embeddings, vectorstore_path, ctx):
        from langchain_community.vectorstores import FAISS

        return FAISS.load_local(
            embeddings=embeddings,
            folder_path=vectorstore_path,
            allow_dangerous_deserialization=True,
        )

    def save_vectorstore(self, vectorstore_folder, vectorstore):
        vectorstore.save_local(vectorstore_folder)

    def get_documents(self, ctx: knext.ExecutionContext) -> tuple[list, ndarray]:
        from langchain_community.vectorstores import FAISS

        store: FAISS = self.load_store(ctx)
        docs = [store.docstore.search(id) for id in store.index_to_docstore_id.values()]
        embeddings = [
            store.index.reconstruct(id) for id in store.index_to_docstore_id.keys()
        ]
        return docs, embeddings

    def get_metadata_filter(self, filter_parameter) -> dict:
        return {
            group.metadata_column: group.metadata_value for group in filter_parameter
        }


faiss_vector_store_port_type = knext.port_type(
    "FAISS Vector Store", FAISSVectorstorePortObject, FAISSVectorstorePortObjectSpec
)


@knext.node(
    "FAISS Vector Store Creator",
    knext.NodeType.SOURCE,
    faiss_icon,
    category=faiss_category,
    keywords=[
        "RAG",
        "Retrieval Augmented Generation",
        "Embeddings",
    ],
)
@knext.input_port(
    "Embedding Model",
    "The embedding model to use for the vector store.",
    embeddings_model_port_type,
)
@knext.input_table(
    name="Documents",
    description="""Table containing a string column representing documents that will be used in the vector store.""",
)
@knext.output_port(
    "FAISS Vector Store",
    "The created FAISS vector store.",
    faiss_vector_store_port_type,
)
class FAISSVectorStoreCreator(BaseVectorStoreCreator):
    """
    Creates a FAISS vector store from a string column and an embedding model.

    The node generates a FAISS vector store that uses the given embedding model to map documents to a numerical vector that captures
    the semantic meaning of the document.

    By default, the node embeds the selected documents using the embedding model, but it is also possible to create the vector store
    from existing embeddings by specifying the corresponding embeddings column in the node dialog.

    Downstream nodes, such as the **Vector Store Retriever**, utilize the vector store to find documents with similar
    semantic meaning when given a query.

    **Note**: If you use the
    [Credentials Configuration node](https://hub.knime.com/knime/extensions/org.knime.features.js.quickforms/latest/org.knime.js.base.node.configuration.input.credentials.CredentialsDialogNodeFactory)
    and do not select the "Save password in configuration (weakly encrypted)" option for passing the API key for the Embedding Model Selector node,
    the Credentials Configuration node will need to be reconfigured upon reopening the workflow, as the credentials flow variable
    was not saved and will therefore not be available to downstream nodes.
    """

    def _configure(
        self,
        embeddings_spec: EmbeddingsPortObjectSpec,
        metadata_column_names: list[str],
    ) -> FAISSVectorstorePortObjectSpec:
        return FAISSVectorstorePortObjectSpec(
            embeddings_spec=embeddings_spec,
            metadata_column_names=metadata_column_names,
        )

    def _create_port_object(
        self,
        ctx: knext.ExecutionContext,
        embeddings_obj: EmbeddingsPortObject,
        documents: list,
        metadata_column_names: list[str],
        embeddings,
    ) -> FAISSVectorstorePortObject:
        from langchain_community.vectorstores import FAISS

        embeddings_model = embeddings_obj.create_model(ctx)
        if embeddings is None:
            db = FAISS.from_documents(
                documents=documents,
                embedding=embeddings_model,
            )
        else:
            docs = [doc.page_content for doc in documents]
            text_embeddings = zip(docs, embeddings)
            metadatas = [doc.metadata for doc in documents]
            db = FAISS.from_embeddings(
                embedding=embeddings_model,
                text_embeddings=text_embeddings,
                metadatas=metadatas,
            )

        return FAISSVectorstorePortObject(
            FAISSVectorstorePortObjectSpec(embeddings_obj.spec, metadata_column_names),
            embeddings_obj,
            vectorstore=db,
        )


@knext.node(
    "FAISS Vector Store Reader",
    knext.NodeType.SOURCE,
    faiss_icon,
    category=faiss_category,
    keywords=[
        "RAG",
        "Retrieval Augmented Generation",
        "Embeddings",
    ],
)
@knext.input_port(
    "Embeddings",
    "The embedding model that the vector store uses for embedding documents.",
    embeddings_model_port_type,
)
@knext.output_port(
    "FAISS Vector Store", "The loaded FAISS vector store.", faiss_vector_store_port_type
)
class FAISSVectorStoreReader:
    """
    Reads a FAISS vector store created with LangChain from a local path.

    This node reads a FAISS vector store create with [LangChain](https://python.langchain.com/docs/integrations/vectorstores/faiss#saving-and-loading) from a local path.
    If you want to create a new vector store, use the **FAISS Vector Store Creator** instead.

    A *vector store* is a data structure or storage mechanism that stores a collection of numerical vectors
    along with their corresponding documents. The vector store enables efficient storage, retrieval, and similarity
    search operations on these vectors and their associated data.

    If the vector store was created with LangChain in Python, the embedding model is not stored with the vectorstore, so it has to be provided separately via the matching **Embedding Model Selector** node.

    On execution, the node will extract a document from the store to obtain information about the document's metadata. This assumes that each document in the vector store has the same kind of metadata attached to it.

    **Note**: If you use the
    [Credentials Configuration node](https://hub.knime.com/knime/extensions/org.knime.features.js.quickforms/latest/org.knime.js.base.node.configuration.input.credentials.CredentialsDialogNodeFactory)
    and do not select the "Save password in configuration (weakly encrypted)" option for passing the API key for the Embedding Model Selector node,
    the Credentials Configuration node will need to be reconfigured upon reopening the workflow, as the credentials flow variable
    was not saved and will therefore not be available to downstream nodes.
    """

    persist_directory = knext.StringParameter(
        "Vector store directory",
        "The local directory in which the vector store is stored.",
    )

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        embeddings_spec: EmbeddingsPortObjectSpec,
    ) -> FAISSVectorstorePortObjectSpec:
        embeddings_spec.validate_context(ctx)
        if not self.persist_directory:
            raise knext.InvalidParametersError("Select the vector store directory.")
        return FAISSVectorstorePortObjectSpec(embeddings_spec)

    def execute(
        self,
        ctx: knext.ExecutionContext,
        embeddings_port_object: EmbeddingsPortObject,
    ) -> FAISSVectorstorePortObject:
        from langchain_community.vectorstores import FAISS

        model = embeddings_port_object.create_model(ctx)

        # TODO: Add check if .fiass and .pkl files are in the directory instead of instatiating as check
        db = FAISS.load_local(
            self.persist_directory,
            model,
            allow_dangerous_deserialization=True,
        )

        embedding = model.embed_query("a")
        if len(embedding) != db.index.d:
            raise knext.InvalidParametersError(
                "The vector store has a different embedding dimensionality than the embedding model. "
                "This could be because the vector store was created with a different model than the one "
                "connected to the FAISS Vector Store Reader."
            )
        document_list = db.similarity_search_by_vector(embedding, k=1)

        metadata_keys = (
            [key for key in document_list[0].metadata] if len(document_list) > 0 else []
        )

        return FAISSVectorstorePortObject(
            FAISSVectorstorePortObjectSpec(embeddings_port_object.spec, metadata_keys),
            embeddings_port_object,
            vectorstore=db,
        )
