from langchain_openai import ChatOpenAI


class DatabricksChatOpenAI(ChatOpenAI):
    """DatabricksChatOpenAI is a subclass of ChatOpenAI that overrides the default parameters to be compatible with Databricks."""

    @property
    def _default_params(self) -> dict:
        params = super()._default_params
        return self._revert_max_completion_tokens(params)

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        return self._revert_max_completion_tokens(payload)

    def _revert_max_completion_tokens(self, params: dict):
        """Revert the conversion to max_completion_tokens which is not supported by Databricks."""
        if "max_completion_tokens" in params:
            # Databricks does not support max_completion_tokens parameter
            # so we remove it from the parameters dictionary.
            params["max_tokens"] = params.pop("max_completion_tokens")
        return params
