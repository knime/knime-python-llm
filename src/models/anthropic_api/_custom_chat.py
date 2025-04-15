from langchain_anthropic import ChatAnthropic
from typing import Any


class _ChatAnthropic(ChatAnthropic):
    def _format_output(self, data: Any, **kwargs: Any):
        from langchain_core.outputs import ChatGeneration, ChatResult

        response = super()._format_output(data, **kwargs)

        message = response.generations[0].message
        if isinstance(message.content, list):
            text = response.generations[0].text
            message.content = text
            return ChatResult(
                generations=[ChatGeneration(message=message)],
                llm_output=response.llm_output,
            )

        return response
