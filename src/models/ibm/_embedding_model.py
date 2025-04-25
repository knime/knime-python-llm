import knime.extension as knext
from typing import List

from langchain_ibm import WatsonxEmbeddings as _WatsonxEmbeddings

class WatsonxEmbeddings(_WatsonxEmbeddings):
    """
    A subclass that wraps embed_documents in a try/except
    to handle exceptions that may occur during embedding generation.
    """

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            return super().embed_documents(texts)
        except Exception:
            raise knext.InvalidParametersError(
                "Failed to embed texts. If you selected a space to run your model, "
                "make sure that the space has a valid runtime service instance. "
                "You can check this at IBM watsonx.ai Studio under Manage tab in your space."
            )