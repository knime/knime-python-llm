from typing import Dict, Optional
from langchain_community.embeddings import GPT4AllEmbeddings
from pydantic import model_validator
import knime.extension as knext
from gpt4all import Embed4All


# TODO: Delete the wrapper if Langchain is > 0.2.x and instantiate Embedd4All class
class _GPT4ALLEmbeddings(GPT4AllEmbeddings):
    model_name: str
    model_path: str
    num_threads: Optional[int] = None
    allow_download: bool = False

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that GPT4All library is installed."""

        try:
            values["client"] = Embed4All(
                model_name=values["model_name"],
                model_path=values["model_path"],
                n_threads=values.get("num_threads"),
                allow_download=values["allow_download"],
            )
            return values
        except ConnectionError:
            raise knext.InvalidParametersError(
                "Connection error. Please ensure that your internet connection is enabled to download the model."
            )
