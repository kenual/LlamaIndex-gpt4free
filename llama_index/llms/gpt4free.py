from typing import (
    Any,
    Sequence
)

from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.base.llms.generic_utils import (
    completion_response_to_chat_response,
    stream_completion_response_to_chat_response,
)
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse
)

from llama_index.core.llms.callbacks import (
    llm_chat_callback,
    llm_completion_callback
)

import g4f


class GPT4Free(CustomLLM):
    context_window: int = 3900
    num_output: int = 256
    model_name: str = "g4f"
    g4f_model: str = "gpt-4"

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
            g4f_model=self.g4f_model,
        )

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        chat_messages: g4f.typing.Messages = [
            {'role': chat_message.role, 'content': chat_message.content} for chat_message in messages
        ]
        response = g4f.ChatCompletion.create(
            model=self.g4f_model,
            messages=chat_messages,
        )
        return completion_response_to_chat_response(CompletionResponse(text=response))

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        response = g4f.ChatCompletion.create(
            model=self.g4f_model,
            messages=[{"role": "user", "content": prompt}],
        )
        return CompletionResponse(text=response)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        streaming_tokens = g4f.ChatCompletion.create(
            model=self.g4f_model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        response = ""
        for token in streaming_tokens:
            response += token
            yield CompletionResponse(text=response, delta=token)

    def get_provider(self):
        return g4f.get_last_provider().__name__
