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


from typing import Any, Dict, List, Optional, Set

from langchain_core.language_models.llms import LLM
from pydantic import Field
from langchain_core.utils import pre_init
from langchain_community.llms.utils import enforce_stop_tokens


class GPT4All(LLM):
    """
    Custom implementation backed by Nomic's GPT4All.
    We decided to implement our own GPT4All class instead of the one provided by langchain_community
    because it did not allow configuration of some parameters (e.g. n_ctx).
    In the future, this class will also be useful for migrating to GPT4All Jinja templates.
    """

    model: str
    n_ctx: int = Field(2048, alias="n_ctx")
    n_threads: Optional[int] = Field(4, alias="n_threads")
    n_predict: Optional[int] = 256
    temp: Optional[float] = 0.7
    top_p: Optional[float] = 0.1
    top_k: Optional[int] = 40
    n_batch: int = Field(8, alias="n_batch")
    device: Optional[str] = Field("cpu", alias="device")
    client: Any

    @staticmethod
    def _model_param_names() -> Set[str]:
        return {
            "n_predict",
            "top_k",
            "top_p",
            "temp",
            "n_batch",
        }

    def _default_params(self) -> Dict[str, Any]:
        return {
            "n_predict": self.n_predict,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temp": self.temp,
            "n_batch": self.n_batch,
        }

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the python package exists in the environment."""
        try:
            from gpt4all import GPT4All as GPT4AllModel
        except ImportError:
            raise ImportError(
                "Could not import gpt4all python package. "
                "Please install it with `pip install gpt4all`."
            )

        full_path = values["model"]
        model_path, delimiter, model_name = full_path.rpartition("/")
        model_path += delimiter

        values["client"] = GPT4AllModel(
            model_name,
            model_path=model_path or None,
            device=values["device"],
            n_ctx=values["n_ctx"],
            allow_download=False,
        )

        if values["n_threads"] is not None:
            values["client"].model.set_thread_count(values["n_threads"])

        return values

    @property
    def _llm_type(self) -> str:
        """Return the type of llm."""
        return "gpt4all"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs: Any,
    ) -> str:
        r"""Call out to GPT4All's generate method.

        Args:
            prompt: The prompt to pass into the model.
            stop: A list of strings to stop generation when encountered.

        Returns:
            The string generated by the model.

        Example:
            .. code-block:: python

                prompt = "Once upon a time, "
                response = model.invoke(prompt, n_predict=55)
        """
        text = ""
        params = {**self._default_params(), **kwargs}

        for token in self.client.generate(prompt, **params):
            text += token
        if stop is not None:
            text = enforce_stop_tokens(text, stop)

        if (
            text
            == "ERROR: The prompt size exceeds the context window size and cannot be processed."
        ):
            raise ValueError(
                "The prompt size exceeds the context window size and cannot be processed."
            )
        return text
