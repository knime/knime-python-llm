from typing import List
from langchain.schema import Document, BaseRetriever


class AdapterRetriever(BaseRetriever):
    """Langchain expects the sources of documents to always be stored in the source metadata field
    but we would like to give the users the choice of which metadata contains the sources.
    This class ensures that what the user chooses is put into the source metadata for LangChain.
    """

    retriever: BaseRetriever
    source_metadata: str

    def _get_relevant_documents(self, query: str, *, run_manager) -> List[Document]:
        docs = self.retriever.get_relevant_documents(query)
        return [self._adapt_document(doc) for doc in docs]

    def _adapt_document(self, doc: Document) -> Document:
        return Document(
            page_content=doc.page_content,
            metadata={"source": doc.metadata[self.source_metadata]},
        )
